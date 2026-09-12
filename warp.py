"""One resampling per frame. Final pixel q on the padded reference canvas -> u = D^-1(H_i^-1 q) ->
original o = L(u) -> sample the decoded frame (ground and clouds blanked, blanks filled with the local
sky level first so interpolation sees continuous data; a validity channel rides along so the no-data
rim falls out of the same resampling). Writes warped/NNNNN.npy (3, H+2P, W+2P) float16 in 0..1 with
0 = no data, count_map.npy (frames per pixel, reference geometry), and the static layers in reference
geometry: ground_d.fit, ground_mean_d.fit, mask_sky_d.npy. The whole per-frame path runs on the GPU.
"""
import os, glob, numpy as np, cv2
from astropy.io import fits
import register as R
import optics, gpu
from fitmodel import R0

W = R.W
PAD = int(round(0.085 * R.WIDTH))          # padded-canvas ring: enough for the sky rotation over a night at 16mm
_cache = {}


def lens_map():
    """(H, W, 2) source pixel for each lens-corrected pixel, u -> o. Computed once per project (lensfun
    takes a minute on 45 MP) and kept in lens_map.npy."""
    if "L" not in _cache:
        path = f"{W}/lens_map.npy"
        if not os.path.exists(path):
            np.save(path, optics.undist_map(R.WIDTH, R.HEIGHT, R.P.LENS, R.P.FOCAL))
        _cache["L"] = np.load(path)
    return _cache["L"]


def total_map(Hi, pad=0, scale=1.0, k1=None, k2=None):
    """Map from final grid (optionally padded by `pad` px on every side, optionally at `scale` of full
    resolution) to original-frame coordinates (always full-resolution units): q -> D^-1(H^-1 q) -> L.
    On the GPU when there is one (a 45 MP grid costs 35 s on the CPU, 3 s there); -1e6 = outside."""
    if k1 is None:
        m = np.load(f"{W}/model.npz"); k1, k2 = float(m["k1"]), float(m["k2"])
    h, w = int(round((R.HEIGHT + 2 * pad) * scale)), int(round((R.WIDTH + 2 * pad) * scale))
    Hinv = np.linalg.inv(Hi)
    L = lens_map()
    if gpu.CUDA:
        import torch
        o = _total_map_gpu(Hinv, h, w, pad, scale, k1, k2, L)
        ox, oy = o[0].cpu().numpy(), o[1].cpu().numpy()
        del o
        torch.cuda.empty_cache()
        return ox, oy
    yy, xx = (np.mgrid[0:h, 0:w].astype(np.float32) / scale) - pad
    den = Hinv[2, 0] * xx + Hinv[2, 1] * yy + Hinv[2, 2]
    ux = (Hinv[0, 0] * xx + Hinv[0, 1] * yy + Hinv[0, 2]) / den
    uy = (Hinv[1, 0] * xx + Hinv[1, 1] * yy + Hinv[1, 2]) / den
    dx, dy = ux - R.CX, uy - R.CY
    rd = np.sqrt(dx * dx + dy * dy) / R0
    rho = np.linspace(0, 1.2, 4001)                                       # D^-1 as a lookup: r_d/R0 = rho (1 + k1 rho^2 + k2 rho^4)
    s = np.interp(rd, rho * (1 + k1 * rho ** 2 + k2 * rho ** 4), rho).astype(np.float32)
    s = np.where(rd > 0, s / np.maximum(rd, 1e-9), 1.0).astype(np.float32)
    ux, uy = (R.CX + dx * s).astype(np.float32), (R.CY + dy * s).astype(np.float32)
    ox = cv2.remap(np.ascontiguousarray(L[..., 0]), ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)
    oy = cv2.remap(np.ascontiguousarray(L[..., 1]), ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)
    return ox, oy


def _total_map_gpu(Hinv, h, w, pad, scale, k1, k2, L):
    """(2, h, w) float32 tensor of source coordinates on the GPU, -1e6 outside the source frame."""
    import torch
    dev = "cuda"
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32) / scale - pad,
                            torch.arange(w, device=dev, dtype=torch.float32) / scale - pad, indexing="ij")
    den = Hinv[2, 0] * xx + Hinv[2, 1] * yy + Hinv[2, 2]
    dx = (Hinv[0, 0] * xx + Hinv[0, 1] * yy + Hinv[0, 2]) / den - R.CX
    dy = (Hinv[1, 0] * xx + Hinv[1, 1] * yy + Hinv[1, 2]) / den - R.CY
    del xx, yy, den
    rd = torch.sqrt(dx * dx + dy * dy) / R0
    rho = rd.clone()
    for _ in range(6):                                                    # D^-1 by fixed point: rho = rd / (1 + k1 rho^2 + k2 rho^4)
        rho = rd / (1 + k1 * rho * rho + k2 * rho ** 4)
    sc = torch.where(rd > 0, rho / rd.clamp(min=1e-9), torch.ones_like(rd))
    del rd, rho
    grid = torch.stack([(R.CX + dx * sc) * (2.0 / (R.WIDTH - 1)) - 1.0, (R.CY + dy * sc) * (2.0 / (R.HEIGHT - 1)) - 1.0], -1)[None]
    del dx, dy, sc
    outside = (grid[0, ..., 0].abs() > 1) | (grid[0, ..., 1].abs() > 1)
    if "Lt" not in _cache:
        _cache["Lt"] = torch.from_numpy(np.ascontiguousarray(np.moveaxis(L, -1, 0))).to(dev)[None]   # (1,2,H,W), kept on the GPU
    o = torch.nn.functional.grid_sample(_cache["Lt"], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    o[:, outside] = -1e6
    del grid, outside
    return o


def filled(d, zero):
    """Blanked pixels take the local sky level so interpolation sees continuous data (no dark ring)."""
    from scipy import ndimage as ndi
    v = (1 - zero).astype(np.float32)[::8, ::8]
    out = np.empty_like(d)
    for c in range(3):
        num = ndi.gaussian_filter(d[c][::8, ::8] * v, 6); den = ndi.gaussian_filter(v, 6)
        fill = cv2.resize(num / np.maximum(den, 1e-6), (d.shape[2], d.shape[1]), interpolation=cv2.INTER_LINEAR)
        out[c] = np.where(zero > 0, fill, d[c])
    return out


def warp(d, ox, oy):
    """d: (3,H,W) float32 with 0 = no-data. Returns the resampled frame with a 1 px zero rim (CPU path)."""
    zero = (d == 0).any(axis=0).astype(np.uint8)
    out = gpu.remap(filled(d, zero), ox, oy)
    zw = gpu.remap(np.stack([zero.astype(np.float32)]), ox, oy, nearest=True)[0] > 0.5
    zw |= (ox < -1e5)
    zw = cv2.dilate(zw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[:, zw] = 0
    return out


def warp_gpu(d, Hi, pad):
    """d: (3,H,W) float32 with 0 = no-data, uploaded once; fill, map and resampling all on the GPU.
    A validity channel is resampled with the data, so any output pixel that touched no-data is 0."""
    import torch, torch.nn.functional as F
    dev = "cuda"
    src = torch.from_numpy(d).to(dev)
    valid = (src > 0).all(dim=0, keepdim=True).float()                    # (1,H,W)
    small_v = F.avg_pool2d(valid[None], 8)[0]                              # local sky level on a 1/8 grid ...
    small_d = F.avg_pool2d((src * valid)[None], 8)[0]
    k = _gauss_kernel(6, dev)
    num = F.conv2d(small_d[:, None], k, padding=k.shape[-1] // 2)[:, 0]
    den = F.conv2d(small_v[None], k, padding=k.shape[-1] // 2)[0]
    fill = F.interpolate((num / den.clamp(min=1e-6))[None], size=src.shape[1:], mode="bilinear", align_corners=False)[0]
    src = torch.where(valid > 0, src, fill)                                # ... fills the blanks
    del small_v, small_d, num, den, fill
    m = np.load(f"{W}/model.npz"); k1, k2 = float(m["k1"]), float(m["k2"])
    h, w = R.HEIGHT + 2 * pad, R.WIDTH + 2 * pad
    o = _total_map_gpu(np.linalg.inv(Hi), h, w, pad, 1.0, k1, k2, lens_map())
    grid = torch.stack([o[0] * (2.0 / (R.WIDTH - 1)) - 1.0, o[1] * (2.0 / (R.HEIGHT - 1)) - 1.0], -1)[None]
    del o
    out = torch.empty((3, h, w), device=dev, dtype=torch.float16)
    for c in range(3):
        out[c] = F.grid_sample(src[c][None, None], grid, mode="bicubic", padding_mode="zeros", align_corners=True)[0, 0].clamp_(min=0).half()
    v = F.grid_sample(valid[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    out[:, v < 0.999] = 0                                                 # touched no-data or the outside: a natural 1 px rim
    res = out.cpu().numpy()
    del src, valid, grid, out, v
    torch.cuda.empty_cache()
    return res


def _gauss_kernel(sigma, dev):
    import torch
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    g = torch.exp(-x * x / (2 * sigma * sigma)); g /= g.sum()
    return (g[:, None] * g[None, :])[None, None]


def main():
    m = np.load(f"{W}/model.npz"); H_all, have = m["H"], m["have"]
    ox, oy = total_map(np.eye(3))
    for name in ("ground", "ground_mean"):
        fits.PrimaryHDU(gpu.remap(fits.getdata(f"{W}/{name}.fit").astype(np.float32), ox, oy)).writeto(f"{W}/{name}_d.fit", overwrite=True)
    sky = np.load(f"{W}/mask_sky.npy")
    np.save(f"{W}/mask_sky_d.npy", gpu.remap(np.stack([sky.astype(np.float32)]), ox, oy, nearest=True)[0] > 0.5)
    print("ground_d, ground_mean_d, mask_sky_d written", flush=True)
    out_dir = f"{W}/warped"
    for f in glob.glob(f"{out_dir}/*.npy"):
        os.remove(f)
    os.makedirs(out_dir, exist_ok=True)
    count = np.zeros((R.HEIGHT, R.WIDTH), np.uint8)
    for i in range(1, R.P.N + 1):
        if not have[i]:
            continue
        d = fits.getdata(R.P.full(i)).astype(np.float32) / 65535.0
        d[:, R.blank(i)] = 0
        if gpu.CUDA:
            o = warp_gpu(d, H_all[i], PAD)
        else:
            o = warp(d, *total_map(H_all[i], pad=PAD)).astype(np.float16)
        np.save(f"{out_dir}/{i:05d}.npy", o)
        count += (o[1, PAD:PAD + R.HEIGHT, PAD:PAD + R.WIDTH] > 0)
        print(f"frame {i:3d} warped", flush=True)
    np.save(f"{W}/count_map.npy", count)
    md = np.load(f"{W}/mask_sky_d.npy")
    print("done; frames per sky pixel (median):", np.median(count[md]) if md.any() else 0)


if __name__ == "__main__":
    main()
