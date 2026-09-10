"""Per-frame cloud masks. Sentence: a cloud is sky that is brighter than that same
patch of sky is in the clear frames (the light-pollution gradient is fixed in camera
coordinates, clouds are not).

Writes clouds.npz: masks (N, H/8, W/8) bool True = cloud, frames (N,) frame numbers,
and preview_clouds.jpg.
"""
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from PIL import Image

import project as P
W = P.W
B = 8                       # block size
THR = 1.5                   # ADU above baseline
DILATE = 3                  # blocks
MIN_BLOCKS = 200            # smaller patches are not clouds
FRAMES = list(P.IDS)


def block_mean(a):
    H, Wd = a.shape
    H2, W2 = H // B * B, Wd // B * B
    return a[:H2, :W2].reshape(H2 // B, B, W2 // B, B).mean(axis=(1, 3))


def main():
    small = []
    for i, n in enumerate(FRAMES, start=1):
        g = fits.getdata(P.full(i))[1].astype(np.float32)
        b = ndi.median_filter(block_mean(g), size=5)          # stars span 1-2 blocks; clouds are wide
        small.append(ndi.gaussian_filter(b, 2))
    small = np.array(small)                                    # (N, h, w), FITS orientation
    base = np.percentile(small, 25, axis=0)
    excess = small - base
    masks = excess > THR
    sky = np.load(W + "/mask_sky.npy")[::B, ::B]
    sky = sky[: masks.shape[1], : masks.shape[2]]
    masks &= sky[None]
    for i in range(len(masks)):                                 # keep only extended patches
        lab, n = ndi.label(masks[i])
        if n:
            sizes = ndi.sum(masks[i], lab, index=np.arange(1, n + 1))
            masks[i] = np.isin(lab, np.arange(1, n + 1)[sizes >= MIN_BLOCKS])
    sky_rows = slice((P.HEIGHT - P.SKY_ROWS) // B, None)                          # rows that are certainly sky
    fr = masks[:, sky_rows].mean(axis=(1, 2))
    masks = np.array([ndi.binary_dilation(m, iterations=DILATE) for m in masks])
    np.savez_compressed(f"{W}\\clouds.npz", masks=masks, frames=np.array(FRAMES), excess=excess.astype(np.float16))
    for n, f in zip(FRAMES, fr):
        print(f"{n:3d} cloud {f * 100:5.1f}%", end="  |" if (n // 2) % 6 != 5 else "\n")
    print()
    print("mean cloud fraction over sky", fr.mean())
    tiles = []
    for i in np.linspace(0, len(FRAMES) - 1, 8).astype(int):
        e = excess[i]; img = np.clip((e + 2) / 8, 0, 1)
        rgb = np.stack([img] * 3, -1)
        rgb[..., 0] = np.where(masks[i], 1.0, rgb[..., 0])
        tiles.append((rgb[::-1] * 255).astype(np.uint8))
    rows = [np.hstack(tiles[:4]), np.hstack(tiles[4:])]
    Image.fromarray(np.vstack(rows)).save(f"{W}\\preview_clouds.jpg", quality=85)


if __name__ == "__main__":
    main()
