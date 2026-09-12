"""Rejection stacking of a frame set, in row bands so any number of frames fits in memory. Zero pixels are
no-data (blanked ground, cloud, warp rim) and never take part. Inputs are 16-bit FITS frames (scaled to
0..1) or float16 .npy frames; the output goes by its extension: a 32-bit float FITS or a float32 .npy.

  python stack.py <glob> <out> median|sigma     e.g.  stack.py "full_[0-9]*.fit" ground.fit median
                                                       stack.py "warped/*.npy" padded_stack.npy sigma
sigma: mean of the frames within 3 robust sigmas of the per-pixel median (two passes).
"""
import sys, os, glob, warnings, numpy as np
from astropy.io import fits
from multiprocessing import Pool
import project as P

warnings.filterwarnings("ignore", "All-NaN")                                # a pixel no frame covers is simply 0
pattern, out, mode = sys.argv[1:4]
files = sorted(glob.glob(f"{P.W}/{pattern}"))
BYTES = 3e8                                                                  # per band, all frames in float32


def shape_of(f):
    if f.endswith(".npy"):
        return np.load(f, mmap_mode="r").shape
    h = fits.getheader(f)
    return (h["NAXIS3"], h["NAXIS2"], h["NAXIS1"])


def band(f, r0, r1):
    if f.endswith(".npy"):
        return np.load(f, mmap_mode="r")[:, r0:r1, :].astype(np.float32)
    with fits.open(f, memmap=True, do_not_scale_image_data=True) as h:
        d = h[0].data[:, r0:r1, :].astype(np.float32)
        hd = h[0].header
        if hd["BITPIX"] == 16:
            d = (d * hd.get("BSCALE", 1) + hd.get("BZERO", 0)) / 65535.0
    return d


def median(x):
    x = np.where(x > 0, x, np.nan)
    return np.nan_to_num(np.nanmedian(x, axis=0))


def sigma(x, k=3.0):
    valid = x > 0
    xn = np.where(valid, x, np.nan)
    m = np.nanmedian(xn, axis=0)
    s = 1.4826 * np.nanmedian(np.abs(xn - m), axis=0) + 1e-6
    for _ in range(2):
        keep = valid & (np.abs(x - m) <= k * s)
        n = keep.sum(axis=0)
        m = np.where(n > 0, (x * keep).sum(axis=0) / np.maximum(n, 1), m)
        s = np.sqrt(np.where(n > 1, (((x - m) ** 2) * keep).sum(axis=0) / np.maximum(n - 1, 1), s * s)) + 1e-6
    return np.nan_to_num(m)


def one(rows):
    r0, r1 = rows
    x = np.stack([band(f, r0, r1) for f in files])                          # (N, 3, rows, W)
    return r0, (median(x) if mode == "median" else sigma(x))


if __name__ == "__main__":
    shape = shape_of(files[0])
    B = max(4, int(BYTES // (len(files) * shape[0] * shape[2] * 4)))
    bands = [(r0, min(r0 + B, shape[1])) for r0 in range(0, shape[1], B)]
    result = np.zeros(shape, np.float32)
    with Pool(max(2, (os.cpu_count() or 4) // 2)) as p:                     # each worker holds one band of every frame (~1.5 GB)
        for k, (r0, res) in enumerate(p.imap_unordered(one, bands)):
            result[:, r0:r0 + res.shape[1]] = res
            print(f"{k + 1}/{len(bands)}", end=" ", flush=True)
    path = f"{P.W}/{out}"
    if out.endswith(".npy"):
        np.save(path, result)
    else:
        fits.PrimaryHDU(result).writeto(path, overwrite=True)
    print(f"\n{out}: {len(files)} frames, {mode}, covered {(result[1] > 0).mean():.3f}", flush=True)
