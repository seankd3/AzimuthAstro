"""One resampling per frame. Final pixel q on the padded reference canvas -> u = D^-1(H_i^-1 q) ->
original o = L(u) -> sample the decoded frame (ground and clouds blanked, blanks filled with the local
sky level first so interpolation sees continuous data). Writes warped/NNNNN.npy (3, H+2P, W+2P) float16
in 0..1 with 0 = no data (1 px rim), count_map.npy (frames per pixel, reference geometry), and the
static layers in reference geometry: ground_d.fit, ground_mean_d.fit, mask_sky_d.npy.
"""
import os, glob, numpy as np, cv2
from astropy.io import fits
import register as R
import optics, gpu
from fitmodel import R0

W = R.W
RIM = 1
PAD = int(round(0.085 * R.WIDTH))          # padded-canvas ring: enough for the sky rotation over a night at 16mm
_L = None


def lens_map():
    """(H, W, 2) source pixel for each lens-corrected pixel, u -> o (cached)."""
    global _L
    if _L is None:
        _L = optics.undist_map(R.WIDTH, R.HEIGHT, R.P.LENS, R.P.FOCAL)
    return _L


def total_map(Hi, pad=0, scale=1.0, k1=None, k2=None):
    """Map from final grid (optionally padded by `pad` px on every side, optionally at `scale` of full
    resolution) to original-frame coordinates (always full-resolution units)."""
    if k1 is None:
        m = np.load(f"{W}/model.npz"); k1, k2 = float(m["k1"]), float(m["k2"])
    if scale != 1.0:
        h, w = int(round((R.HEIGHT + 2 * pad) * scale)), int(round((R.WIDTH + 2 * pad) * scale))
        yy, xx = (np.mgrid[0:h, 0:w].astype(np.float32) / scale) - pad
    else:
        yy, xx = np.mgrid[-pad:R.HEIGHT + pad, -pad:R.WIDTH + pad].astype(np.float32)
    Hinv = np.linalg.inv(Hi)
    den = Hinv[2, 0] * xx + Hinv[2, 1] * yy + Hinv[2, 2]
    ux = (Hinv[0, 0] * xx + Hinv[0, 1] * yy + Hinv[0, 2]) / den
    uy = (Hinv[1, 0] * xx + Hinv[1, 1] * yy + Hinv[1, 2]) / den
    dx, dy = ux - R.CX, uy - R.CY
    rd = np.sqrt(dx * dx + dy * dy) / R0
    rho = np.linspace(0, 1.2, 4001)                                       # D^-1 as a lookup: r_d/R0 = rho (1 + k1 rho^2 + k2 rho^4)
    s = np.interp(rd, rho * (1 + k1 * rho ** 2 + k2 * rho ** 4), rho).astype(np.float32)
    s = np.where(rd > 0, s / np.maximum(rd, 1e-9), 1.0).astype(np.float32)
    ux, uy = (R.CX + dx * s).astype(np.float32), (R.CY + dy * s).astype(np.float32)
    L = lens_map()
    ox = cv2.remap(np.ascontiguousarray(L[..., 0]), ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)
    oy = cv2.remap(np.ascontiguousarray(L[..., 1]), ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)
    return ox, oy


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
    """d: (3,H,W) float32 with 0 = no-data. Returns the resampled frame with a RIM px zero rim."""
    zero = (d == 0).any(axis=0).astype(np.uint8)
    out = gpu.remap(filled(d, zero), ox, oy)
    zw = gpu.remap(np.stack([zero.astype(np.float32)] * 3)[:1], ox, oy, nearest=True)[0] > 0.5
    zw |= (ox < -1e5)                                                     # outside the source
    zw = cv2.dilate(zw.astype(np.uint8), np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)) > 0
    out[:, zw] = 0
    return out


def main():
    m = np.load(f"{W}/model.npz"); H_all, have = m["H"], m["have"]
    ox, oy = total_map(np.eye(3))
    for name in ("ground", "ground_mean"):
        fits.PrimaryHDU(gpu.remap(fits.getdata(f"{W}/{name}.fit").astype(np.float32), ox, oy)).writeto(f"{W}/{name}_d.fit", overwrite=True)
    sky = np.load(f"{W}/mask_sky.npy")
    np.save(f"{W}/mask_sky_d.npy", gpu.remap(np.stack([sky.astype(np.float32)] * 3)[:1], ox, oy, nearest=True)[0] > 0.5)
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
        ox, oy = total_map(H_all[i], pad=PAD)
        o = warp(d, ox, oy)
        np.save(f"{out_dir}/{i:05d}.npy", o.astype(np.float16))
        count += (o[1, PAD:PAD + R.HEIGHT, PAD:PAD + R.WIDTH] > 0)
        print(f"frame {i:3d} warped", flush=True)
    np.save(f"{W}/count_map.npy", count)
    md = np.load(f"{W}/mask_sky_d.npy")
    print("done; frames per sky pixel (median):", np.median(count[md]) if md.any() else 0)


if __name__ == "__main__":
    main()
