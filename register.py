"""Star registration with a physical prior. Sentence: between two frames the sky is a camera
rotation about the celestial pole by an angle known from the timestamps, so star pairs can be
predicted, then a homography is fitted to the pairs.

python register.py diag 45 1        -> match statistics for one pair
python register.py all              -> H for every frame (H_all.npy) and warped r_u_sky_NNNNN.fit
All coordinates are FITS orientation (row 0 = bottom), x = column, y = row.
"""
import sys, glob, numpy as np, cv2, optics
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
RIM = 5
WRITE = False               # the fit only needs H; warp2 does the real resampling


def frame_times():
    """seconds of each frame relative to the reference frame, indexed 1..N."""
    return dict(P.TIMES)


def detect(i):
    """Star list (x, y, flux) for u_sky_i, brightest NSTARS, FITS orientation."""
    g = fits.getdata(f"{W}\\u_sky_{i:05d}.fit")[1].astype(np.float32)
    valid = g > 0
    bg = ndi.median_filter(g[::4, ::4], 15)
    bg = np.repeat(np.repeat(bg, 4, axis=0), 4, axis=1)[: g.shape[0], : g.shape[1]]
    hp = np.where(valid, g - bg, 0)
    sig = 1.4826 * np.median(np.abs(hp[valid][::97]))
    sm = ndi.gaussian_filter(hp, 1.0)
    peak = (sm == ndi.maximum_filter(sm, size=7)) & (sm > 6 * sig) & valid
    # stay away from blanked edges
    peak &= ~ndi.binary_dilation(~valid, iterations=8)
    ys, xs = np.nonzero(peak)
    flux = sm[ys, xs]
    order = np.argsort(-flux)[:NSTARS]
    ys, xs, flux = ys[order], xs[order], flux[order]
    # centroid refinement in a 7x7 window
    out = []
    for y, x, f in zip(ys, xs, flux):
        y0, y1, x0, x1 = max(0, y - 3), min(HEIGHT, y + 4), max(0, x - 3), min(WIDTH, x + 4)
        w = np.clip(hp[y0:y1, x0:x1], 0, None)
        s = w.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out.append(((w * xx).sum() / s, (w * yy).sum() / s, f))
    return np.array(out, np.float64)


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


def diag(i, j, pole_display=P.POLE_DISPLAY):
    times = frame_times()
    theta = OMEGA * (times[j] - times[i])
    ref, cur = detect(i), detect(j)
    print(f"stars: ref {len(ref)} cur {len(cur)}; dt {times[j]-times[i]:.0f}s theta {np.degrees(theta):.2f} deg")
    pole = (pole_display[0], HEIGHT - pole_display[1])
    for sgn in (+1, -1):
        for f in (F_PX * 0.9, F_PX, F_PX * 1.1):
            H0 = H_model(sgn * theta, pole, f)
            src, dst, d = match(ref, cur, H0, 25)
            Hm, n, res = fit(ref, cur, H0)
            print(f"sign {sgn:+d} f {f:6.0f}: prior matches@25px {len(src):4d}  -> fitted matches@3px {n:4d} median resid {res:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "diag":
        diag(int(sys.argv[2]), int(sys.argv[3]))


def one(args):
    i, theta, pole, ref = args
    if i == REF:
        Hm, n, res = np.eye(3), len(ref), 0.0
    else:
        cur = detect(i)
        Hm, n, res = fit(ref, cur, H_model(-theta, pole))
        if Hm is None or n < 60:
            return i, None, n, res
    if not WRITE:
        return i, Hm, n, res
    d = fits.getdata(f"{W}/u_sky_{i:05d}.fit").astype(np.float32)
    zero = (d == 0).any(axis=0).astype(np.uint8)
    out = np.stack([np.clip(cv2.warpPerspective(c, Hm, (WIDTH, HEIGHT), flags=cv2.INTER_LANCZOS4,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, None) for c in d])
    zw = cv2.warpPerspective(zero, Hm, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    zw = cv2.dilate(zw, np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8))
    out[:, zw > 0] = 0
    fits.PrimaryHDU(out).writeto(f"{W}/r_u_sky_{i:05d}.fit", overwrite=True)
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
