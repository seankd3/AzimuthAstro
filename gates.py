"""Mechanical gates after stages. Each check returns (ok, message); `azastro run` calls check(stage)
after the stage and fails the run when a gate fails. Every gate here is a bug that actually
happened once; keep it that way.
"""
import os, json, glob, numpy as np
from astropy.io import fits
import project as P

W = P.W


def _sky_lum():
    d = fits.getdata(f"{W}/ground.fit").astype(np.float32)
    return d[0] * 0.25 + d[1] * 0.5 + d[2] * 0.25


def convert():
    """orientation: the top third of frame 1 must be sky-bright relative to the bottom third (lake/ground
    is darker than sky at night); hot pixels must be rare."""
    d = fits.getdata(P.full(1)).astype(np.float32)[1]
    H = d.shape[0]
    top, bottom = np.median(d[-H // 3:]), np.median(d[: H // 3])          # FITS rows run bottom-up
    ok1 = top > bottom
    hot = np.load(f"{W}/hot_mask.npy")
    frac = hot.mean()
    ok2 = frac < 0.002
    return ok1 and ok2, f"top/bottom median {top:.0f}/{bottom:.0f} ({'sky up' if ok1 else 'UPSIDE DOWN?'}); hot pixels {frac*100:.3f}% ({'ok' if ok2 else 'too many: threshold misfire'})"


def refine():
    """mask: sky only above the treeline. Bottom 15% of display rows must hold no sky; sky fraction sane."""
    m = np.load(f"{W}/mask_sky.npy")
    H = m.shape[0]
    low = m[: int(H * 0.15)].mean()                                          # FITS bottom = display bottom
    frac = m.mean()
    ok = low < 0.005 and 0.2 < frac < 0.95
    return ok, f"sky fraction {frac:.3f}; sky in bottom 15% {low*100:.2f}% ({'ok' if low < 0.005 else 'LEAK into ground/lake'})"


def stack():
    """the aligned stack has data over most of the sky mask and is not empty."""
    s = fits.getdata(f"{W}/sky.fit").astype(np.float32)[1]
    m = np.load(f"{W}/mask_sky_d.npy")
    cover = (s[m] > 0).mean()
    return cover > 0.9, f"stack covers {cover*100:.1f}% of the sky mask"


def tone():
    """composite must keep the stack's stars: count bright peaks in both over the sky mask."""
    from scipy import ndimage as ndi
    lin = np.load(f"{W}/composite_lin.npy", mmap_mode="r")[1]
    s = fits.getdata(f"{W}/sky.fit").astype(np.float32)[1]
    m = np.load(f"{W}/mask_sky_d.npy")

    def peaks(img):
        g = np.asarray(img[::2, ::2], np.float32)
        hp = g - ndi.median_filter(g, 9)
        sig = 1.4826 * np.median(np.abs(hp[m[::2, ::2]]))
        return int(((hp == ndi.maximum_filter(hp, 5)) & (hp > 8 * sig) & m[::2, ::2]).sum())
    a, b = peaks(s), peaks(lin)
    ratio = b / max(a, 1)
    return ratio > 0.7, f"stars: stack {a}, composite {b} ({ratio:.2f} of stack; {'ok' if ratio > 0.7 else 'STARS LOST in compose'})"


def trails():
    g = np.load(f"{W}/trails_gapless.npy", mmap_mode="r")[1]
    m = np.load(f"{W}/mask_sky_d.npy")
    frac = (np.asarray(g[::4, ::4]) > 0)[m[::4, ::4]].mean()
    return frac > 0.5, f"trail layer covers {frac*100:.0f}% of the sky"


GATES = {"convert": convert, "refine": refine, "stack": stack, "tone": tone, "trails": trails}


def check(stage):
    fn = GATES.get(stage)
    if fn is None:
        return True, ""
    try:
        return fn()
    except Exception as e:                       # a gate that cannot run is a failed gate
        return False, f"gate error: {e!r}"


if __name__ == "__main__":
    import sys
    for st in (sys.argv[1:] or GATES):
        ok, msg = check(st)
        print(f"{'PASS' if ok else 'FAIL'} {st}: {msg}")
