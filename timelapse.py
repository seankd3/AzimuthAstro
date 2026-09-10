"""Three videos from the same frames, all at half resolution (4096 x 2732 -> 4K UHD crop-free).

sky-locked   each sharp frame warped so the stars stay put and the ground turns around Polaris;
             sub-steps rotate the frame continuously, neighbours cross-fade for clouds/lights.
standard     stars move as they did; sub-steps come from the sky-locked frame rotated back to time t,
             ground from the undistorted frame (static), composited through the sky mask.
clouds       the cloud layer alone: frame minus the clear-sky stack, stars removed, in the sky mask.

Frames are written as JPEG into timelapse_<name>/ and encoded by encode.sh.
"""
import os, sys, numpy as np, cv2
from scipy import ndimage as ndi
from PIL import Image
import render as Rn

W = Rn.W
SCALE = 0.5
SUB = 4                     # rendered sub-steps per real frame
N = Rn.P.N


def save(name, k, img8):
    Image.fromarray(Rn.to_display(img8)).save(f"{W}/timelapse_{name}/{k:04d}.jpg", quality=93)


def main(which):
    for n in which:
        os.makedirs(f"{W}/timelapse_{n}", exist_ok=True)
    tone = Rn.Tone()
    mask = np.load(f"{W}/mask_sky_d.npy")
    sky = Rn.load_fits("sky")
    m_full = Rn.feathered_mask(mask, sky[1] > 0)
    h, w = int(round(Rn.HEIGHT * SCALE)), int(round(Rn.WIDTH * SCALE))
    m = cv2.resize(m_full, (w, h), interpolation=cv2.INTER_AREA)
    ground = Rn.downscale(Rn.load_fits("ground_d"), SCALE)
    sky_mask_v = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    # per-frame renders, cached one frame ahead
    def renders(i):
        full = Rn.load_full(i)
        und = Rn.to_d(full, np.eye(3), SCALE)              # original view, undistorted
        locked = Rn.to_d(full, Rn.H_FRAME[i], SCALE)       # stars fixed
        return und, locked
    k_locked = k_std = k_cloud = 0
    prev = renders(1)
    for i in range(1, N + 1):
        cur = prev
        nxt = renders(i + 1) if i < N else cur
        ti, tn = Rn.TIMES[i], (Rn.TIMES[i + 1] if i < N else Rn.TIMES[i] + 33)
        for s in range(SUB if i < N else 1):
            a = s / SUB
            t = ti + a * (tn - ti)
            if "locked" in which:
                # rotate the sky-locked view continuously: the ground turns, stars stay
                Hm = Rn.H_at(t) @ np.linalg.inv(Rn.H_FRAME[i])
                fa = Rn.warp_d(cur[1], Hm)
                if a > 0:
                    Hn = Rn.H_at(t) @ np.linalg.inv(Rn.H_FRAME[i + 1])
                    fb = Rn.warp_d(nxt[1], Hn)
                    fa = fa * (1 - a) + fb * a
                save("locked", k_locked, tone.apply(fa - Rn.BLACK)); k_locked += 1
            if "standard" in which:
                # sky at time t from the locked view, ground static from the undistorted view
                skyt = Rn.warp_d(cur[1], np.linalg.inv(Rn.H_at(t)))
                gnd = cur[0] if a == 0 else cur[0] * (1 - a) + nxt[0] * a
                lin = Rn.blend(skyt, gnd, m)
                save("standard", k_std, tone.apply(lin)); k_std += 1
            if "clouds" in which and s == 0:
                d = cur[0] - ground
                d[:, ~sky_mask_v] = 0
                layer = np.stack([ndi.gaussian_filter(ndi.median_filter(c, 5), 2) for c in d])
                layer = np.clip(layer, 0, None)
                save("clouds", k_cloud, tone.apply(layer * 6 + tone.black[:, None, None] + 0.0006)); k_cloud += 1
        prev = nxt
        print(f"frame {i}/{N}", flush=True)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:] or ["locked", "standard", "clouds"])
