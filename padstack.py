"""Padded stack: the reference-geometry stack extended PAD px on every side so the trail engine can
sweep stars that entered or left the frame during the night. Inside the frame the Siril stack is
used as is; the ring is a per-pixel median of the 90 blanked, hot-patched frames warped onto the
padded canvas. Writes padded_stack.npy (3, H+2P, W+2P) float32 raw-with-black, 0 = no data.
"""
import numpy as np, cv2
from astropy.io import fits
import register as R
import render as Rn
import warp2

W = Rn.W
PAD = Rn.PAD
H, Wd = R.HEIGHT, R.WIDTH
HP, WP = H + 2 * PAD, Wd + 2 * PAD
N = R.P.N
RIM = 1


def main():
    top = np.zeros((N, PAD, WP, 3), np.float16); bot = np.zeros_like(top)
    left = np.zeros((N, H, PAD, 3), np.float16); right = np.zeros_like(left)
    kernel = np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)
    for i in range(1, N + 1):
        if not Rn._m["have"][i]:
            continue
        d = fits.getdata(R.P.light(i)).astype(np.float32) / 65535.0
        ox, oy = warp2.total_map(Rn.H_FRAME[i], pad=PAD)
        zero = (d == 0).any(axis=0).astype(np.uint8)
        d = warp2.filled(d, zero)
        out = np.stack([np.clip(cv2.remap(c, ox, oy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, None) for c in d])
        zw = cv2.dilate(cv2.remap(zero, ox, oy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1), kernel)
        out[:, zw > 0] = 0
        o = np.moveaxis(out, 0, -1)
        top[i - 1] = o[HP - PAD:, :]; bot[i - 1] = o[:PAD, :]
        left[i - 1] = o[PAD:PAD + H, :PAD]; right[i - 1] = o[PAD:PAD + H, WP - PAD:]
        print(f"frame {i}", flush=True)
    padded = np.zeros((3, HP, WP), np.float32)
    sky = Rn.load_fits("sky")
    padded[:, PAD:PAD + H, PAD:PAD + Wd] = sky

    def med(strip):                                         # median of non-zero values, row blocks
        n, h, w, _ = strip.shape
        res = np.zeros((h, w, 3), np.float32)
        for r0 in range(0, h, 64):
            blk = strip[:, r0:r0 + 64].astype(np.float32)
            blk[blk == 0] = np.nan
            with np.errstate(all="ignore"):
                m = np.nanmedian(blk, axis=0)
            res[r0:r0 + 64] = np.nan_to_num(m, nan=0.0)
        return np.moveaxis(res, -1, 0)
    padded[:, HP - PAD:, :] = med(top)
    padded[:, :PAD, :] = med(bot)
    padded[:, PAD:PAD + H, :PAD] = med(left)
    padded[:, PAD:PAD + H, WP - PAD:] = med(right)
    np.save(f"{W}/padded_stack.npy", padded)
    print("done", (padded[1] > 0).mean())


if __name__ == "__main__":
    main()
