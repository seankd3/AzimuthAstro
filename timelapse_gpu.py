"""GPU timelapse: same three videos as timelapse.py (locked, standard, clouds), rendered at half
resolution with the per-frame resampling on the GPU. Per real frame: two remaps of the full frame
through the lens map (undistorted view, sky-locked view), then the sub-steps are small homography
warps of those half-resolution views. Requires torch with CUDA.
"""
import os, sys, time, numpy as np, cv2, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from PIL import Image
import render as Rn
import warp

W = Rn.W
SCALE, SUB = 0.5, 4
dev = torch.device("cuda")
N = Rn.P.N
h, w = int(round(Rn.HEIGHT * SCALE)), int(round(Rn.WIDTH * SCALE))


def save(name, k, img8):
    Image.fromarray(Rn.to_display(img8)).save(f"{W}/timelapse_{name}/{k:04d}.jpg", quality=93)


def to_grid(mapx, mapy, src_w, src_h):
    g = torch.empty((1,) + mapx.shape + (2,), dtype=torch.float32, device=dev)
    g[0, ..., 0] = torch.from_numpy(mapx).to(dev) * (2.0 / (src_w - 1)) - 1.0
    g[0, ..., 1] = torch.from_numpy(mapy).to(dev) * (2.0 / (src_h - 1)) - 1.0
    return g


def sample3(src, g):
    """src (3,1,H,W) float32 on device -> (3,h,w) numpy float32"""
    return F.grid_sample(src, g.expand(3, -1, -1, -1), mode="bicubic", padding_mode="zeros", align_corners=True)[:, 0].clamp_(min=0)


class Renders:
    """Half-res undistorted and sky-locked views of a frame, computed once each on the GPU."""

    def __init__(self):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / SCALE           # half-res output -> full-res coords
        self.grid_pts = (xx, yy)
        ox, oy = warp.total_map(np.eye(3), scale=SCALE)
        self.ident = to_grid(ox, oy, Rn.WIDTH, Rn.HEIGHT)

    def frame(self, i):
        full = torch.from_numpy(Rn.load_full(i)).to(dev)[:, None]
        und = sample3(full, self.ident)
        ox, oy = warp.total_map(Rn.H_FRAME[i], scale=SCALE)
        g = to_grid(ox, oy, Rn.WIDTH, Rn.HEIGHT)
        locked = sample3(full, g)
        del full, g
        return und, locked


def rim(img3, r=2):
    """no-data mask of a warped view, grown r px so the interpolation ring at the border goes with it."""
    z = (img3[1] <= 0).float()[None, None]
    return F.max_pool2d(z, 2 * r + 1, 1, r)[0, 0] > 0


def warp_half(img3, Hm):
    """warp a device (3,h,w) tensor by a full-res-units homography (output -> source via inverse)."""
    Hs = Rn.scale_H(Hm, SCALE)
    Hi = np.linalg.inv(Hs)
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    den = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    g = torch.empty((1, h, w, 2), dtype=torch.float32, device=dev)
    g[0, ..., 0] = ((Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / den) * (2.0 / (w - 1)) - 1.0
    g[0, ..., 1] = ((Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / den) * (2.0 / (h - 1)) - 1.0
    return sample3(img3[:, None], g)


class ToneGPU:
    """Rn.Tone.apply on the device: linear (3,h,w) tensor -> (3,h,w) uint8 numpy."""

    def __init__(self, tone):
        self.black = torch.tensor(tone.black, device=dev)[:, None, None]
        self.scale = torch.tensor(Rn.WB / (tone.white * Rn.WB[1]), device=dev)[:, None, None]
        self.a = float(tone.a); self.norm = float(np.arcsinh(tone.a))

    def apply(self, lin):
        x = ((lin - self.black) * self.scale).clamp_(0, 1)
        y = (torch.asinh(x * self.a) / self.norm).clamp_(0, 1).pow_(1 / 1.15)
        return (y * 255).to(torch.uint8).cpu().numpy()


def main(which):
    for n in which:
        os.makedirs(f"{W}/timelapse_{n}", exist_ok=True)
    tone = Rn.Tone(); tg = ToneGPU(tone)
    mask = np.load(f"{W}/mask_sky_d.npy")
    m = torch.from_numpy(cv2.resize(Rn.feathered_mask(mask, np.ones_like(mask), 0, 1), (w, h), interpolation=cv2.INTER_AREA)).to(dev)[None]
    core = (m[0] > 0.999)
    ground = Rn.downscale(Rn.load_fits("ground_d"), SCALE)
    q = 4                                                                       # cloud layer at quarter of the video size
    ground_q = Rn.downscale(ground, 1 / q)
    sky_mask_q = cv2.resize(mask.astype(np.uint8), (w // q, h // q), interpolation=cv2.INTER_NEAREST) > 0
    black = float(Rn.BLACK)
    R = Renders()
    k_locked = k_std = k_cloud = 0
    prev = R.frame(1); tstart = time.time()
    for i in range(1, N + 1):
        cur = prev
        nxt = R.frame(i + 1) if i < N else cur
        ti, tn = Rn.TIMES[i], (Rn.TIMES[i + 1] if i < N else Rn.TIMES[i] + 33)
        for s in range(SUB if i < N else 1):
            a = s / SUB
            t = ti + a * (tn - ti)
            if "locked" in which:
                fa = warp_half(cur[1], Rn.H_at(t) @ np.linalg.inv(Rn.H_FRAME[i]))
                fa[:, rim(fa)] = 0
                if a > 0:
                    fb = warp_half(nxt[1], Rn.H_at(t) @ np.linalg.inv(Rn.H_FRAME[i + 1]))
                    fb[:, rim(fb)] = 0
                    fa = fa * (1 - a) + fb * a
                save("locked", k_locked, tg.apply(fa - black)); k_locked += 1
            if "standard" in which:
                skyt = warp_half(cur[1], np.linalg.inv(Rn.H_at(t)))
                gnd = cur[0] if a == 0 else cur[0] * (1 - a) + nxt[0] * a
                nodata = rim(skyt)
                skyt = skyt - black; gndl = gnd - black
                skyt[:, nodata] = gndl[:, nodata]
                if a == 0:                                                  # background match once per real frame
                    off = torch.stack([gndl[c][core][::9].median() - skyt[c][core][::9].median() for c in range(3)])[:, None, None]
                lin = (skyt + off) * m + gndl * (1 - m)
                save("standard", k_std, tg.apply(lin)); k_std += 1
            if "clouds" in which and s == 0:
                d = Rn.downscale(cur[0].cpu().numpy(), 1 / q) - ground_q
                d[:, ~sky_mask_q] = 0
                layer = np.clip(np.stack([ndi.gaussian_filter(ndi.median_filter(c, 3), 1) for c in d]), 0, None)
                layer = torch.from_numpy(np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC) for c in layer])).to(dev)
                save("clouds", k_cloud, tg.apply(layer * 6 + tg.black + 0.0006)); k_cloud += 1
        prev = nxt
        if i % 10 == 0:
            print(f"frame {i}/{N}  {(time.time() - tstart) / i:.1f} s/frame", flush=True)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:] or ["locked", "standard", "clouds"])
