"""Rejection stacking of a frame set, in row bands so any number of frames fits in memory. Zero pixels are
no-data (blanked ground, cloud, warp rim) and never take part. Inputs are 16-bit FITS frames (scaled to
0..1) or float16 .npy frames; the output goes by its extension: a 32-bit float FITS or a float32 .npy.

  python stack.py <glob> <out> median|sigma     e.g.  stack.py "full_[0-9]*.fit" ground.fit median
                                                       stack.py "warped/*.npy" padded_stack.npy sigma
sigma: mean of the frames within 3 robust sigmas of the per-pixel median (two passes).
A sample at the sensor ceiling (project clip) is not a measurement. A pixel whose channel is clipped in
more than a quarter of its frames is saturated: the few unclipped frames are the dim ones and would bias
that channel low (green first, so magenta cores); it comes out at the ceiling in a daylight-white colour.
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


CLIP = P.CLIP * 0.98                                                         # resampling rounds a plateau off a little
WHITE = P.CLIP / (np.array(P.DAYLIGHT_WB, np.float32) / P.DAYLIGHT_WB[1])    # the ceiling, in a colour that renders white


def unclipped(x):
    """(valid, saturated): samples that are data and not at the ceiling; pixels with a channel clipped in
    more than a quarter of its samples."""
    data = x > 0
    clipped = x >= CLIP
    valid = data & ~clipped
    saturated = (clipped.sum(axis=0) > 0.25 * np.maximum(data.sum(axis=0), 1)).any(axis=0)
    return valid, saturated


def whiten(result, saturated):
    result[:, saturated] = np.maximum(result[:, saturated], WHITE[:, None])
    return result


def median(x):
    valid, saturated = unclipped(x)
    x = np.where(valid, x, np.nan)
    return whiten(np.nan_to_num(np.nanmedian(x, axis=0)), saturated)


def sigma(x, k=3.0):
    valid, saturated = unclipped(x)
    xn = np.where(valid, x, np.nan)
    m = np.nanmedian(xn, axis=0)
    s = 1.4826 * np.nanmedian(np.abs(xn - m), axis=0) + 1e-6
    for _ in range(2):
        keep = valid & (np.abs(x - m) <= k * s)
        n = keep.sum(axis=0)
        m = np.where(n > 0, (x * keep).sum(axis=0) / np.maximum(n, 1), m)
        s = np.sqrt(np.where(n > 1, (((x - m) ** 2) * keep).sum(axis=0) / np.maximum(n - 1, 1), s * s)) + 1e-6
    return whiten(np.nan_to_num(m), saturated)


def one(rows):
    r0, r1 = rows
    x = np.stack([band(f, r0, r1) for f in files])                          # (N, 3, rows, W)
    return r0, (median(x) if mode == "median" else sigma(x))


if __name__ == "__main__":
    shape = shape_of(files[0])
    B = max(4, int(BYTES // (len(files) * shape[0] * shape[2] * 4)))
    bands = [(r0, min(r0 + B, shape[1])) for r0 in range(0, shape[1], B)]
    result = np.zeros(shape, np.float32)
    todo = {r0: (r0, r1) for r0, r1 in bands}
    try:
        with Pool(max(2, (os.cpu_count() or 4) // 2)) as p:                 # each worker holds one band of every frame (~1.5 GB)
            for k, (r0, res) in enumerate(p.imap_unordered(one, bands)):
                result[:, r0:r0 + res.shape[1]] = res; todo.pop(r0)
                print(f"{k + 1}/{len(bands)}", end=" ", flush=True)
    except Exception as e:                                                  # a worker respawn in a detached run can fail
        print(f"pool failed ({type(e).__name__}); {len(todo)} bands finished in-process", flush=True)
    for r0, r1 in sorted(todo.values()):                                    # (Windows: DuplicateHandle denied); the rest is done here
        _, res = one((r0, r1)); result[:, r0:r1] = res
        print(f"{r0}", end=" ", flush=True)
    path = f"{P.W}/{out}"
    if out.endswith(".npy"):
        np.save(path, result)
    else:
        fits.PrimaryHDU(result).writeto(path, overwrite=True)
    print(f"\n{out}: {len(files)} frames, {mode}, covered {(result[1] > 0).mean():.3f}", flush=True)
