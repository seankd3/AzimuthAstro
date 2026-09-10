"""GPU trail engine: same outputs as trails.py, with the star canvas, sampling grids and accumulators
resident on the GPU (one upload, one download). fp16 canvases and accumulators, float32 grid math
built in place, one channel at a time: ~2 GB peak on a 4 GB card. Requires torch with CUDA.
"""
import os, time, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
import render as Rn
import register as R

W = Rn.W
STEP, TAU, SNAP_EVERY, SCALE_VIDEO, PAD = 8.0, 900.0, 4, 0.5, Rn.PAD
MAXSTEPS = int(os.environ.get("MAXSTEPS", "0"))
dev = torch.device("cuda")


class Grid:
    """Reusable sampling grid for outputs (h, w) sampling a padded source (src_h, src_w)."""

    def __init__(self, w, h, src_w, src_h, offset):
        self.w, self.h, self.offset = w, h, offset
        self.sx, self.sy = 2.0 / (src_w - 1), 2.0 / (src_h - 1)
        yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        self.xx, self.yy = xx, yy
        self.g = torch.empty((1, h, w, 2), dtype=torch.float32, device=dev)
        self.den = torch.empty((h, w), dtype=torch.float32, device=dev)

    def set(self, Hm):
        Hi = np.linalg.inv(Hm)
        torch.add(self.xx * float(Hi[2, 0]), self.yy * float(Hi[2, 1]), out=self.den); self.den += float(Hi[2, 2])
        gx = self.g[0, ..., 0]; gy = self.g[0, ..., 1]
        torch.add(self.xx * float(Hi[0, 0]), self.yy * float(Hi[0, 1]), out=gx); gx += float(Hi[0, 2]); gx /= self.den
        torch.add(self.xx * float(Hi[1, 0]), self.yy * float(Hi[1, 1]), out=gy); gy += float(Hi[1, 2]); gy /= self.den
        gx += self.offset; gy += self.offset
        gx *= self.sx; gx -= 1.0
        gy *= self.sy; gy -= 1.0
        return self.g


def sample(src_c, grid):
    """src_c (1,1,H,W) float32 -> (h,w) float32."""
    return F.grid_sample(src_c, grid, mode="bicubic", padding_mode="zeros", align_corners=True)[0, 0].clamp_(min=0)


def main():
    padded = np.load(f"{W}/padded_stack.npy")
    valid = padded[1] > 0
    stars, _ = Rn.star_layer(padded, valid, k=4.0)
    del padded
    Hp, Wp = stars.shape[1:]
    sv = SCALE_VIDEO
    hv, wv = int(round(R.HEIGHT * sv)), int(round(R.WIDTH * sv))
    stars_v = Rn.downscale(stars, sv)
    S = torch.from_numpy(stars).to(dev)[:, None]                          # (3,1,Hp,Wp) float32: grid_sample wants source and grid alike
    Sv = torch.from_numpy(stars_v).to(dev)[:, None]
    del stars
    gap = torch.zeros((3, R.HEIGHT, R.WIDTH), device=dev, dtype=torch.float16); comet = torch.zeros_like(gap)
    grow = torch.zeros((3, hv, wv), device=dev, dtype=torch.float16)
    G = Grid(R.WIDTH, R.HEIGHT, Wp, Hp, PAD)
    Gv = Grid(wv, hv, stars_v.shape[2], stars_v.shape[1], PAD * sv)
    bg = np.load(f"{W}/bg_layer.npy"); ground = Rn.load_fits("ground_d"); resid = np.load(f"{W}/resid_layer.npy"); vl = np.load(f"{W}/valid_layer.npy")
    mask = np.load(f"{W}/mask_sky_d.npy")
    ground_v, bg_v, resid_v = (Rn.downscale(a, sv) for a in (ground, bg, resid))
    mask_v = cv2.resize(mask.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
    valid_v = cv2.resize(vl.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
    del ground, bg, resid
    t0 = min(Rn.TIMES.values()); t1 = max(Rn.TIMES.values()) + Rn.EXPOSURE
    ts = np.arange(t0, t1 + 1e-6, STEP)
    if MAXSTEPS:
        ts = ts[:MAXSTEPS]
    print(f"{len(ts)} positions, GPU resident, {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated", flush=True)
    os.makedirs(f"{W}/trails_frames", exist_ok=True)
    tone = Rn.Tone(); snap = 0; tstart = time.time()
    for k, t in enumerate(ts):
        Hm = np.linalg.inv(Rn.H_at(t))                           # reference -> sky at time t
        w = float(np.exp(-(t1 - t) / TAU))
        g = G.set(Hm)
        for c in range(3):
            r = sample(S[c:c + 1], g).half()
            torch.maximum(gap[c], r, out=gap[c])
            r *= w
            torch.maximum(comet[c], r, out=comet[c])
            del r
        gv = Gv.set(Rn.scale_H(Hm, sv))
        for c in range(3):
            r = sample(Sv[c:c + 1], gv).half()
            torch.maximum(grow[c], r, out=grow[c])
            del r
        if k % SNAP_EVERY == 0 or k == len(ts) - 1:
            lin = Rn.compose(grow.float().cpu().numpy(), bg_v, ground_v, mask_v, resid_v, valid_v)
            Image.fromarray(Rn.to_display(tone.apply(lin))).save(f"{W}/trails_frames/{snap:04d}.jpg", quality=93)
            snap += 1
        if k % 20 == 0:
            torch.cuda.synchronize()
            print(f"step {k}/{len(ts)}  {(time.time() - tstart) / (k + 1):.2f} s/step  {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak", flush=True)
    np.save(f"{W}/trails_gapless.npy", gap.float().cpu().numpy()); np.save(f"{W}/trails_comet.npy", comet.float().cpu().numpy())
    print("done", snap, "snapshots", f"{time.time() - tstart:.0f}s")


if __name__ == "__main__":
    main()
