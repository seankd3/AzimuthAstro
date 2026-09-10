"""Hot pixels: bright in the minimum over six frames spread through the night (a star never stays put),
replaced by the median of their 24 neighbours. decode.py patches in memory before a frame is written;
this stage re-checks the written frames and patches whatever is left, writing hot_mask.npy."""
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi


def find(mn):
    """mask (H, W) of hot pixels in a per-pixel minimum image (3, H, W), noise-scaled, blobs of <= 25 px only
    (a bright star that never leaves its pixel would be bigger), grown by one pixel."""
    hot = np.zeros(mn.shape[1:], bool)
    for c in range(3):
        diff = mn[c] - ndi.median_filter(ndi.median_filter(mn[c], size=(1, 5)), size=(5, 1))   # separable: 6x faster than a 5x5 median
        sig = 1.4826 * np.median(np.abs(diff[::7, ::7]))          # noise of the minimum image
        hot |= diff > max(8 * sig, 25)
    lab, n = ndi.label(hot)
    sizes = ndi.sum(hot, lab, index=np.arange(1, n + 1))
    return ndi.binary_dilation(np.isin(lab, np.arange(1, n + 1)[sizes <= 25]), iterations=1)


def neighbours(hot):
    """index arrays: the hot pixels (hy, hx) and their 24 neighbours (ny, nx), each (24, n)."""
    hy, hx = np.nonzero(hot)
    H, W = hot.shape
    offs = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)]
    ny = np.stack([np.clip(hy + dy, 0, H - 1) for dy, dx in offs]); nx = np.stack([np.clip(hx + dx, 0, W - 1) for dy, dx in offs])
    return hy, hx, ny, nx


def patch(d, idx):
    """in place on a (3, H, W) array: each hot pixel becomes the median of its neighbours."""
    hy, hx, ny, nx = idx
    for c in range(3):
        d[c][hy, hx] = np.median(d[c][ny, nx].astype(np.float32), axis=0).astype(d.dtype)


if __name__ == "__main__":
    import project as P
    picks = [max(1, round(x)) for x in np.linspace(1, P.N, 6)]
    mn = None
    for i in picks:
        d = fits.getdata(P.full(i)).astype(np.float32)
        mn = d if mn is None else np.minimum(mn, d)
    hot = find(mn)
    np.save(f"{P.W}/hot_mask.npy", hot)
    print("hot pixels left after decode:", int(hot.sum()), flush=True)
    if hot.any():
        idx = neighbours(hot)
        for i in range(1, P.N + 1):                    # read, patch, write back whole: a memory map flushes each touched page alone (5x slower)
            arr = fits.getdata(P.full(i))
            patch(arr, idx)
            fits.PrimaryHDU(arr).writeto(P.full(i), overwrite=True)
    print("done")
