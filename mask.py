"""Sky mask from the static (ground) stack alone. Sentence: the sky is the smooth region
that touches the top of the frame; the ground is everything with edges. Stars are small
edge blobs and are dropped by size; the lake is smooth but is cut off from the sky by the
treeline.

Arrays stay in FITS orientation (row 0 = bottom). Writes mask_sky.npy (True = sky) and previews.
"""
import sys, numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from PIL import Image

import project as P
W = P.W
S = 2                       # work scale
TIGHTEN = 8                 # full-res px of sky given up at the boundary
K = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55   # of local sky level
TMAX = 2.0                  # ADU texture; above this it is lit ground
HI = 2.5                    # of local sky level; brighter than this is a lamp, not sky
OPEN = 3                    # work-scale px; outline opening
COARSE = 8                  # work-scale px; region opening that cuts bridges into the lake


def stretch(img, lo=0.5, hi=99.9, gamma=0.35):
    a, b = np.percentile(img, [lo, hi])
    return (np.clip((img - a) / (b - a), 0, 1) ** gamma * 255).astype(np.uint8)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def main():
    d = fits.getdata(f"{W}\\ground.fit").astype(np.float32)
    L = (d[0] * 0.25 + d[1] * 0.5 + d[2] * 0.25)
    H, Wd = L.shape
    g = L[: H // S * S, : Wd // S * S].reshape(H // S, S, Wd // S, S).mean(axis=(1, 3))
    g = g * 65535 - 2047                                   # ADU above black
    hp = g - ndi.gaussian_filter(g, 4)
    T = np.sqrt(ndi.gaussian_filter(hp * hp, 3))
    Lm = ndi.median_filter(g, 9)
    hy = (H - P.SKY_ROWS) // S                                    # lowest row that is certainly sky
    ref = ndi.uniform_filter1d(np.median(Lm[hy:hy + 40], axis=0), 41)
    cand = (Lm > FRAC * ref[None, :]) & (Lm < HI * ref[None, :]) & (T < TMAX)   # lamp glow is far brighter than sky
    cand[hy:] = True
    cand = ndi.binary_closing(cand, disk(2))
    lab, _ = ndi.label(cand)
    top = np.unique(lab[-10]); top = top[top > 0]
    sky = np.isin(lab, top)
    sky = ndi.binary_fill_holes(sky)
    def top_component(b):
        lab, _ = ndi.label(b)
        top = np.unique(lab[-10]); top = top[top > 0]
        return np.isin(lab, top)
    coarse = top_component(ndi.binary_opening(sky, disk(COARSE)))          # region: bridges into the lake cut
    fine = top_component(ndi.binary_opening(sky, disk(OPEN)))              # outline: tree tips kept
    sky = top_component(fine & ndi.binary_dilation(coarse, disk(COARSE + OPEN)))
    sky = ndi.binary_erosion(sky, disk(TIGHTEN // S))
    n = 0; sig = float(np.median(T[hy:]))
    full = np.repeat(np.repeat(sky, S, axis=0), S, axis=1)
    out = np.zeros((H, Wd), bool); out[:full.shape[0], :full.shape[1]] = full
    np.save(f"{W}\\mask_sky.npy", out)
    print(f"T sky sigma {sig:.6f}; edge components {n}; sky fraction {out.mean():.3f}")
    ov = np.stack([stretch(g)] * 3, -1).astype(np.float32)
    ov[..., 0] = np.where(sky, ov[..., 0] * 0.5 + 128, ov[..., 0])
    ov = ov[::-1].astype(np.uint8)
    Image.fromarray(ov[::2, ::2]).save(f"{W}/preview_mask_overlay.jpg", quality=90)
    h, w = ov.shape[:2]
    band = (~sky[::-1][:, 30:-30]).any(axis=1).nonzero()[0]; band = band[band > 30]   # display rows with ground (not the eroded border)
    r0 = max(0, (int(band.min()) if len(band) else h // 2) - h // 8); r1 = min(h, r0 + h // 3)
    for name, c0, c1 in (("left", 0, w // 3), ("mid", w // 3, 2 * w // 3), ("right", 2 * w // 3, w)):
        Image.fromarray(ov[r0:r1, c0:c1]).save(f"{W}/preview_mask_crop_{name}.jpg", quality=90)


if __name__ == "__main__":
    main()
