"""Trails from the sharp stack. Sentence: a star trail is the star layer of the (padded) stack swept
through every rotation the night covered; lighten-accumulate it over the stationary background and
put the static ground back.

One pass computes:
  gapless   max over t of rotate(stars, t)                       -> trails_gapless.npy
  comet     same with a weight that fades with age; heads sharp  -> trails_comet.npy
  growing   snapshots of the gapless accumulation every N steps  -> trails_frames/NNNN.jpg (half-res)
"""
import os, numpy as np, cv2
from PIL import Image
import render as Rn
import register as R

W = Rn.W
STEP = 8.0                  # seconds between rendered positions (= one exposure: no gaps, no beads)
TAU = 900.0                 # comet tail e-folding time, seconds
SNAP_EVERY = 4
SCALE_VIDEO = 0.5
PAD = Rn.PAD


def T(pad, s=1.0):
    return np.array([[1, 0, pad * s], [0, 1, pad * s], [0, 0, 1.0]])


def main():
    padded = np.load(f"{W}/padded_stack.npy")
    valid = padded[1] > 0
    stars, _ = Rn.star_layer(padded, valid, k=4.0)
    del padded
    bg = np.load(f"{W}/bg_layer.npy"); ground = Rn.load_fits("ground_d"); resid = np.load(f"{W}/resid_layer.npy"); valid = np.load(f"{W}/valid_layer.npy")
    mask = np.load(f"{W}/mask_sky_d.npy")
    t0 = min(Rn.TIMES.values()); t1 = max(Rn.TIMES.values()) + Rn.EXPOSURE
    ts = np.arange(t0, t1 + 1e-6, STEP)
    print(f"{len(ts)} positions from {t0:.0f}s to {t1:.0f}s", flush=True)
    sv = SCALE_VIDEO
    stars_v = Rn.downscale(stars, sv)
    hv, wv = int(round(R.HEIGHT * sv)), int(round(R.WIDTH * sv)); pv = int(round(PAD * sv))
    gap = np.zeros((3, R.HEIGHT, R.WIDTH), np.float32); comet = np.zeros_like(gap)
    grow = np.zeros((3, hv, wv), np.float32)
    ground_v = Rn.downscale(ground, sv); bg_v = Rn.downscale(bg, sv); resid_v = Rn.downscale(resid, sv); valid_v = cv2.resize(valid.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
    mask_v = cv2.resize(mask.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST) > 0
    os.makedirs(f"{W}/trails_frames", exist_ok=True)
    tone = Rn.Tone()
    snap = 0
    for k, t in enumerate(ts):
        Hm = np.linalg.inv(Rn.H_at(t))                 # reference -> sky as seen at time t
        # full-res: warp the padded star canvas, keep the frame window
        Hp = T(PAD) @ Hm @ T(-PAD)
        r = np.stack([cv2.warpPerspective(c, Hp, (R.WIDTH + 2 * PAD, R.HEIGHT + 2 * PAD), flags=cv2.INTER_LANCZOS4,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)[PAD:PAD + R.HEIGHT, PAD:PAD + R.WIDTH] for c in stars])
        np.maximum(gap, r, out=gap)
        w = np.exp(-(t1 - t) / TAU)
        np.maximum(comet, r * w, out=comet)
        Hs = Rn.scale_H(Hp, sv)
        rv = np.stack([cv2.warpPerspective(c, Hs, (wv + 2 * pv, hv + 2 * pv), flags=cv2.INTER_LANCZOS4,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)[pv:pv + hv, pv:pv + wv] for c in stars_v])
        np.maximum(grow, rv, out=grow)
        if k % SNAP_EVERY == 0 or k == len(ts) - 1:
            lin = Rn.compose(grow, bg_v, ground_v, mask_v, resid_v, valid_v)
            Image.fromarray(Rn.to_display(tone.apply(lin))).save(f"{W}/trails_frames/{snap:04d}.jpg", quality=93)
            snap += 1
        if k % 50 == 0:
            print(f"step {k}/{len(ts)} t={t:.0f}s", flush=True)
    np.save(f"{W}/trails_gapless.npy", gap); np.save(f"{W}/trails_comet.npy", comet)
    print("done", snap, "snapshots")


if __name__ == "__main__":
    main()
