"""Mechanical gates after stages. Each check returns (ok, message); `azastro run` calls check(stage)
after the stage and fails the run when a gate fails. Every gate here is a bug that actually
happened once; keep it that way.
"""
import os, glob, numpy as np
from astropy.io import fits
import project as P

W = P.W


def _sky_lum():
    d = fits.getdata(f"{W}/ground.fit").astype(np.float32)
    return d[0] * 0.25 + d[1] * 0.5 + d[2] * 0.25


def convert():
    """orientation: the top third of frame 1 must be sky-bright relative to the bottom third (lake/ground
    is darker than sky at night)."""
    d = fits.getdata(P.full(1)).astype(np.float32)[1]
    H = d.shape[0]
    top, bottom = np.median(d[-H // 3:]), np.median(d[: H // 3])          # FITS rows run bottom-up
    ok = top > bottom
    return ok, f"top/bottom median {top:.0f}/{bottom:.0f} ({'sky up' if ok else 'UPSIDE DOWN?'})"


def hot():
    """hot pixels must be rare (a fixed threshold once flagged 58% of an ISO 3200 frame; a real R5 night at 27 C is 0.3%)."""
    frac = np.load(f"{W}/hot_mask.npy").mean()
    return frac < 0.01, f"hot pixels {frac*100:.3f}% ({'ok' if frac < 0.01 else 'too many: threshold misfire'})"


def encode():
    """every video is real (a killed encode once delivered a 48-byte mp4)."""
    vids = glob.glob(f"{W}/{P.NAME}_*_4K.mp4")
    small = [os.path.basename(v) for v in vids if os.path.getsize(v) < 1e6]
    return bool(vids) and not small, f"{len(vids)} videos" + (f"; too small: {small}" if small else "")


def refine():
    """mask: sky only above the treeline. Bottom 15% of display rows must hold no sky; sky fraction sane."""
    m = np.load(f"{W}/mask_sky.npy")
    H = m.shape[0]
    low = m[: int(H * 0.15)].mean()                                          # FITS bottom = display bottom
    frac = m.mean()
    below = m[: H - P.SKY_ROWS].any(axis=0).mean()                           # the certainly-sky row sits above the trees, so sky must go on below it
    ok = low < 0.005 and 0.2 < frac < 0.95 and below > 0.5
    return ok, f"sky fraction {frac:.3f}; sky in bottom 15% {low*100:.2f}% ({'ok' if low < 0.005 else 'LEAK into ground/lake'}); sky below the sky row in {below*100:.0f}% of columns{'' if below > 0.5 else ' (CUT AT THE SKY ROW: smoothness test failed)'}"


def stack():
    """the aligned stack has data over most of the sky mask and is not empty."""
    import render as Rn
    s = Rn.load_sky()[1]
    m = np.load(f"{W}/mask_sky_d.npy")
    cover = (s[m] > 0).mean()
    return cover > 0.9, f"stack covers {cover*100:.1f}% of the sky mask"


def clouds():
    """a clear night is mostly clear: the mean cloud fraction over the sky stays small (an absolute
    threshold once flagged 60% of a clear night and left 41 of 144 frames in the stack)."""
    z = np.load(f"{W}/clouds.npz")
    m = z["masks"]
    sky = np.load(f"{W}/mask_sky.npy")[::8, ::8][: m.shape[1], : m.shape[2]]
    fr = np.array([mm[sky].mean() for mm in m])
    hi = [int(z["frames"][i]) for i in np.nonzero(fr > 0.05)[0]]
    return fr.mean() < 0.25, f"cloud {fr.mean()*100:.1f}% of the sky on average; frames over 5%: {hi[:20]}{' ...' if len(hi) > 20 else ''}"


def warp():
    """most of the night reaches most of the sky: median frames per sky pixel."""
    c = np.load(f"{W}/count_map.npy")
    m = np.load(f"{W}/mask_sky_d.npy")
    med = float(np.median(c[m])) if m.any() else 0.0
    return med >= 0.3 * P.N, f"median {med:.0f} frames per sky pixel of {P.N}"


def tone():
    """composite must keep the stack's stars (count bright peaks in both over the sky mask), and the
    black point is a real black (a black point on empty wedges once gave -2047)."""
    import json
    t = json.load(open(f"{W}/tone.json"))
    black = np.array(t["black"]) * 65535
    if not (np.abs(black) < 300).all():
        return False, f"black point {black.round(0).tolist()} ADU: not on data"
    from scipy import ndimage as ndi
    lin = np.load(f"{W}/composite_lin.npy", mmap_mode="r")[1]
    import render as Rn
    s = Rn.load_sky()[1]
    m = np.load(f"{W}/mask_sky_d.npy")

    def peaks(img):
        g = np.asarray(img[::2, ::2], np.float32)
        hp = g - ndi.median_filter(g, 9)
        sig = 1.4826 * np.median(np.abs(hp[m[::2, ::2]]))
        return int(((hp == ndi.maximum_filter(hp, 5)) & (hp > 8 * sig) & m[::2, ::2]).sum())
    a, b = peaks(s), peaks(lin)
    ratio = b / max(a, 1)
    return ratio > 0.7, f"stars: stack {a}, composite {b} ({ratio:.2f} of stack; {'ok' if ratio > 0.7 else 'STARS LOST in compose'})"


def colour():
    """the delivered sky is neutral: the tone curve balances on the sky and carries camera RGB to sRGB."""
    from PIL import Image
    import glob as _g
    out = sorted(_g.glob(f"{W}/{P.NAME}_Print.jpg")) or sorted(_g.glob(f"{W}/{P.NAME}_*.jpg"))
    if not out:
        return True, "no stills yet"
    a = np.asarray(Image.open(out[0]))[::8, ::8].astype(np.float32)
    m = np.median(a[: a.shape[0] * 3 // 10].reshape(-1, 3), axis=0)
    rg, bg = m[0] / max(m[1], 1e-6), m[2] / max(m[1], 1e-6)
    ok = 0.8 < rg < 1.25 and 0.8 < bg < 1.25
    return ok, f"{os.path.basename(out[0])} sky R/G {rg:.2f} B/G {bg:.2f} ({'ok' if ok else 'COLOUR CAST'})"


def export():
    """the bright trails are not magenta: green clipping in single frames once made every dense trail core
    lavender. Bright pixels of the gapless still must not have both R and B above G."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    p = f"{W}/{P.NAME}_Trails_gapless.jpg"
    if not os.path.exists(p):
        return False, "no trails still"
    a = np.asarray(Image.open(p))[::6, ::6].astype(np.float32)
    v = a[: a.shape[0] * 3 // 4].reshape(-1, 3)
    bright = v[v.max(axis=1) > 140]
    if len(bright) < 100:
        return True, "too few bright trail pixels to judge"
    m = np.median(bright, axis=0)
    rg, bg = m[0] / max(m[1], 1e-6), m[2] / max(m[1], 1e-6)
    magenta = min(rg, bg)
    ok = magenta < 1.08
    return ok, f"bright trails R/G {rg:.2f} B/G {bg:.2f} ({'ok' if ok else 'MAGENTA: clipped green'})"


def trails():
    g = np.load(f"{W}/trails_gapless.npy", mmap_mode="r")[1]
    m = np.load(f"{W}/mask_sky_d.npy")
    frac = (np.asarray(g[::4, ::4]) > 0)[m[::4, ::4]].mean()
    return frac > 0.5, f"trail layer covers {frac*100:.0f}% of the sky"


GATES = {"convert": convert, "hot": hot, "refine": refine, "clouds": clouds, "warp": warp, "print": colour, "encode": encode, "export": export, "stack": stack, "tone": tone, "trails": trails}


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
