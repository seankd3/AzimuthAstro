"""Hot pixels: bright in the minimum over six frames spread through the night (a star never stays put),
median-patched in place in every full_NNNNN.fit. Writes hot_mask.npy."""
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
import project as P

picks = [max(1, round(x)) for x in np.linspace(1, P.N, 6)]
mn = None
for i in picks:
    d = fits.getdata(P.full(i)).astype(np.float32)
    mn = d if mn is None else np.minimum(mn, d)
hot = np.zeros(mn.shape[1:], bool)
for c in range(3):
    diff = mn[c] - ndi.median_filter(mn[c], 5)
    sig = 1.4826 * np.median(np.abs(diff[::7, ::7]))          # noise of the minimum image
    hot |= diff > max(8 * sig, 25)
lab, n = ndi.label(hot)
sizes = ndi.sum(hot, lab, index=np.arange(1, n + 1))
hot = ndi.binary_dilation(np.isin(lab, np.arange(1, n + 1)[sizes <= 25]), iterations=1)
np.save(f"{P.W}/hot_mask.npy", hot)
print("hot pixels", int(hot.sum()), flush=True)
hy, hx = np.nonzero(hot)
H, Wd = hot.shape
offs = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)]
ny = np.stack([np.clip(hy + dy, 0, H - 1) for dy, dx in offs]); nx = np.stack([np.clip(hx + dx, 0, Wd - 1) for dy, dx in offs])
for i in range(1, P.N + 1):
    # unscaled memory map: only the hot pixels' bytes change, the file is not rewritten
    with fits.open(P.full(i), mode="update", memmap=True, do_not_scale_image_data=True) as h:
        d = h[0].data
        for c in range(3):                                           # median of the 24 neighbours, hot pixels only
            d[c][hy, hx] = np.median(d[c][ny, nx].astype(np.int32), axis=0).astype(d.dtype)
        h.flush()
    if i % 50 == 0:
        print("patched", i, flush=True)
print("done")
