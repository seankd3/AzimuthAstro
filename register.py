"""Star registration with a physical prior. Sentence: between two frames the sky is a camera
rotation about the celestial pole by an angle known from the timestamps, so star pairs can be
predicted, then a homography is fitted to the pairs.

python register.py diag 45 1        -> match statistics for one pair
python register.py all              -> H for every frame (H_all.npy)
All coordinates are FITS orientation (row 0 = bottom), x = column, y = row.
"""
import sys, numpy as np, cv2, optics
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import project as P
W = P.W
WIDTH, HEIGHT = P.WIDTH, P.HEIGHT
CX, CY = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
F_PX = optics.focal_px(P.FOCAL, max(WIDTH, HEIGHT), P.CROP)
OMEGA = np.deg2rad(360.0 / 86164.0905)   # sidereal rate, rad/s
REF = P.REF
NSTARS = 1500


def frame_times():
    """seconds of each frame relative to the reference frame, indexed 1..N."""
    return dict(P.TIMES)


_masks = {}


def blank(i):
    """(H, W) True where frame i holds no sky: ground (mask) and that frame's clouds."""
    if "sky" not in _masks:
        _masks["sky"] = np.load(f"{W}/mask_sky.npy")
        cl = np.load(f"{W}/clouds.npz"); _masks["clouds"] = dict(zip(cl["frames"].tolist(), cl["masks"]))
    sky = _masks["sky"]; cm = _masks["clouds"][P.IDS[i - 1]]
    full = np.repeat(np.repeat(cm, 8, axis=0), 8, axis=1)
    cloud = np.zeros(sky.shape, bool); h, w = min(sky.shape[0], full.shape[0]), min(sky.shape[1], full.shape[1])
    cloud[:h, :w] = full[:h, :w]
    return ~sky | cloud


def detect(i):
    """Star list (x, y, flux) of frame i in lens-corrected coordinates: the brightest NSTARS peaks of the
    decoded frame's sky (clouds out), centroided, then the positions undistorted (never the image)."""
    g = fits.getdata(P.full(i))[1].astype(np.float32)
    valid = ~blank(i)
    g[~valid] = 0
    bg = ndi.median_filter(g[::4, ::4], 15)
    bg = np.repeat(np.repeat(bg, 4, axis=0), 4, axis=1)[: g.shape[0], : g.shape[1]]
    hp = np.where(valid, g - bg, 0)
    sig = 1.4826 * np.median(np.abs(hp[valid][::97]))
    sm = ndi.gaussian_filter(hp, 1.0)
    peak = (sm == ndi.maximum_filter(sm, size=7)) & (sm > 6 * sig) & valid
    peak &= ~ndi.binary_dilation(~valid, iterations=8)                    # stay away from blanked edges
    ys, xs = np.nonzero(peak)
    flux = sm[ys, xs]
    order = np.argsort(-flux)[:NSTARS]
    ys, xs, flux = ys[order], xs[order], flux[order]
    out = []
    for y, x, f in zip(ys, xs, flux):                                     # centroid refinement in a 7x7 window
        y0, y1, x0, x1 = max(0, y - 3), min(HEIGHT, y + 4), max(0, x - 3), min(WIDTH, x + 4)
        w = np.clip(hp[y0:y1, x0:x1], 0, None)
        s = w.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out.append(((w * xx).sum() / s, (w * yy).sum() / s, f))
    out = np.array(out, np.float64).reshape(-1, 3)
    if "lens" not in _masks:
        _masks["lens"] = optics.undist_map(WIDTH, HEIGHT, P.LENS, P.FOCAL)
    out[:, :2] = optics.undistort_points(out[:, :2], _masks["lens"])
    return out


def K():
    return np.array([[F_PX, 0, CX], [0, F_PX, CY], [0, 0, 1.0]])


def H_model(theta, pole_xy, f=None):
    """Homography mapping a frame rotated by theta (about the pole axis) back to the reference."""
    return optics.sky_homography(theta, F_PX if f is None else f, pole_xy, CX, CY)


apply = optics.apply


def match(ref, cur, Hm, tol):
    pred = apply(Hm, cur)
    tree = cKDTree(ref[:, :2])
    d, j = tree.query(pred, distance_upper_bound=tol)
    ok = np.isfinite(d)
    return cur[ok, :2], ref[j[ok], :2], d[ok]


def fit(ref, cur, H0, tols=(60, 25, 8, 3)):
    Hm = H0
    for tol in tols:
        src, dst, _ = match(ref, cur, Hm, tol)
        if len(src) < 12:
            return None, 0, np.inf
        Hn, inl = cv2.findHomography(src, dst, cv2.RANSAC, tol)
        if Hn is None:
            return None, 0, np.inf
        Hm = Hn
    src, dst, d = match(ref, cur, Hm, 3)
    return Hm, len(src), float(np.median(d)) if len(d) else np.inf


def one(args):
    i, theta, pole, ref = args
    if i == REF:
        Hm, n, res = np.eye(3), len(ref), 0.0
    else:
        cur = detect(i)
        Hm, n, res = fit(ref, cur, H_model(-theta, pole))
        if Hm is None or n < 60:
            return i, None, n, res
    return i, Hm, n, res


def run_all(pole_display=P.POLE_DISPLAY):
    from multiprocessing import Pool
    times = frame_times()
    pole = (pole_display[0], HEIGHT - pole_display[1])
    ref = detect(REF)
    # rotation sense from one nearby frame: the sign whose prior pairs more stars wins (southern sky turns the other way)
    probe = REF + 5 if REF + 5 <= P.N else REF - 5
    cur = detect(probe)
    sign = max((+1, -1), key=lambda sg: len(match(ref, cur, H_model(-sg * OMEGA * times[probe], pole), 25)[0]))
    print(f"rotation sense {'+' if sign > 0 else '-'} (frame {probe})", flush=True)
    jobs = [(i, sign * OMEGA * times[i], pole, ref) for i in range(1, P.N + 1)]
    Hs = {}
    with Pool(6) as p:
        for i, Hm, n, res in p.imap_unordered(one, jobs):
            print(f"frame {i:3d}: matches {n:4d} resid {res:.2f} {'OK' if Hm is not None else 'FAILED'}", flush=True)
            if Hm is not None:
                Hs[i] = Hm
    np.save(f"{W}/H_all.npy", Hs, allow_pickle=True)
    print("registered", len(Hs), "of", P.N)


if __name__ == "__main__" and sys.argv[1] == "all":
    run_all()
