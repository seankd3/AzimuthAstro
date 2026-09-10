"""Pixel-accurate treeline. Sentence: near the coarse boundary a pixel is sky if it is closer to the
local sky level than to the local tree level; any pixel a star trail crossed is sky regardless.

Inputs: mask_sky.npy (coarse, from mask.py; saved as mask_coarse.npy), ground.fit (median stack),
ground_mean.fit (sigma-clipped mean: trail residue marks where stars passed).
Output: mask_sky.npy (refined), preview crops.
"""
import numpy as np, cv2
from astropy.io import fits
from scipy import ndimage as ndi
from PIL import Image
import project as P

W = P.W
OUT = 120            # px outside the coarse sky (the coarse mask rounds concavities off by up to ~100 px)
IN = 10              # px inside it
SIG = 30             # px scale of the local tree level
SIG_SKY = 200        # px scale of the local sky level
DEEP = 25            # px: how far inside each region the level samples come from


def nc(img, weight, sigma):
    """normalized convolution: local mean of img over weight>0."""
    num = ndi.gaussian_filter(img * weight, sigma)
    den = ndi.gaussian_filter(weight, sigma)
    return num / np.maximum(den, 1e-6), den


def main():
    coarse = np.load(f"{W}/mask_sky.npy")
    np.save(f"{W}/mask_coarse.npy", coarse)
    med = fits.getdata(f"{W}/ground.fit").astype(np.float32)
    mean = fits.getdata(f"{W}/ground_mean.fit").astype(np.float32)
    L = (med[0] * 0.25 + med[1] * 0.5 + med[2] * 0.25) * 65535
    H, Wd = L.shape
    S = 4
    Ls = L[::S, ::S]
    cs = coarse[::S, ::S]
    sky_far = ndi.binary_erosion(cs, iterations=DEEP // S).astype(np.float32)
    tree_far = ndi.binary_erosion(~cs, iterations=DEEP // S).astype(np.float32)
    sky_lvl, sky_w = nc(Ls, sky_far, SIG_SKY / S)      # sky brightness varies slowly: reach far into pockets
    tree_lvl, tree_w = nc(Ls, tree_far, SIG / S)
    up = lambda a: cv2.resize(a.astype(np.float32), (Wd, H), interpolation=cv2.INTER_LINEAR)
    sky_lvl, tree_lvl, sky_w, tree_w = up(sky_lvl), up(tree_lvl), up(sky_w), up(tree_w)
    noise = 1.4826 * np.median(np.abs((L - ndi.median_filter(L[::S, ::S], 5).repeat(S, 0).repeat(S, 1)[:H, :Wd]))[coarse][::53])
    edge = ndi.binary_dilation(coarse, iterations=OUT) & ~ndi.binary_erosion(coarse, iterations=IN)
    Lm = ndi.median_filter(L, 3)                       # kill single-pixel noise before deciding
    near_tree = tree_w > 0.02
    two_sided = near_tree & (np.abs(sky_lvl - tree_lvl) > 4 * noise)
    closer_sky = np.where(two_sided, np.abs(Lm - sky_lvl) < np.abs(Lm - tree_lvl),   # between the two levels
                          np.abs(Lm - sky_lvl) < 5 * noise)                            # no tree nearby: sky-only test
    refined = coarse.copy()
    band = edge & (two_sided | ~near_tree)
    refined[band] = closer_sky[band]
    # star trails prove sky: the mean stack carries trail residue the median does not
    trail = (mean[1] - med[1]) * 65535 > 6 * noise
    trail = ndi.binary_dilation(trail, iterations=1)
    refined |= trail & ndi.binary_dilation(coarse, iterations=OUT)
    # clean: no islands, no pinholes, then give up the 1 px mixed edge
    lab, n = ndi.label(refined)
    top = np.unique(lab[-40]); top = top[top > 0]                    # sky is what reaches the top of the frame (the border rows are eroded)
    refined = np.isin(lab, top)
    holes = ~refined
    lab, n = ndi.label(holes)
    sizes = ndi.sum(holes, lab, index=np.arange(1, n + 1))
    refined |= np.isin(lab, np.arange(1, n + 1)[sizes < 400])
    refined = ndi.binary_erosion(refined, iterations=1)
    np.save(f"{W}/mask_sky.npy", refined)
    print(f"noise {noise:.2f} ADU; band px {band.sum()}; changed {np.count_nonzero(refined != coarse)}; sky fraction {refined.mean():.4f} (coarse {coarse.mean():.4f})")
    # previews: three 1:1 crops along the treeline where the boundary is busiest
    a, b = np.percentile(L[coarse], 1), np.percentile(L, 99.7)
    v = (np.clip((L - a) / (b - a), 0, 1) ** 0.5 * 255).astype(np.uint8)
    rows = np.nonzero(edge.any(axis=1))[0]
    r0, r1 = rows.min(), rows.max()
    cols = np.linspace(Wd * 0.1, Wd * 0.9, 3).astype(int)
    tiles = []
    for c in cols:
        rr = edge[:, c - 300:c + 300].any(axis=1).nonzero()[0]
        rc = int(np.median(rr)) if len(rr) else (r0 + r1) // 2
        rs, cs2 = slice(max(0, rc - 200), rc + 200), slice(c - 300, c + 300)
        rgb = np.stack([v[rs, cs2]] * 3, -1).astype(np.float32)
        rgb[..., 0] = np.where(refined[rs, cs2], rgb[..., 0] * 0.5 + 128, rgb[..., 0])
        rgb[..., 2] = np.where(coarse[rs, cs2] & ~refined[rs, cs2], 200, rgb[..., 2])
        tiles.append(rgb[::-1].astype(np.uint8))
    Image.fromarray(np.hstack(tiles)).save(f"{W}/preview_mask_refined.jpg", quality=92)


if __name__ == "__main__":
    main()
