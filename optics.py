"""How the sky reaches the sensor: the pinhole focal length in pixels, the homography of a sky turned by
theta about the celestial pole (K R K^-1), and the lens's distortion from the bundled lensfun profile."""
import os, re, numpy as np, lensfunpy

_DB = lensfunpy.Database(paths=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "lensfun-mil-canon.xml")], load_common=False)


def focal_px(focal_mm, long_side_px, crop=1.0):
    """focal length in pixels: the sensor's long side is 36 mm / crop."""
    return focal_mm * long_side_px / (36.0 / crop)


def rotation(axis, theta):
    """Rodrigues: 3x3 rotation by theta about a unit axis."""
    x, y, z = axis
    Kx = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * Kx @ Kx


def sky_homography(theta, f, pole, cx, cy):
    """Homography of the sky turned by theta about the celestial pole seen at image point `pole`."""
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    a = np.linalg.inv(k) @ np.array([pole[0], pole[1], 1.0]); a /= np.linalg.norm(a)
    return k @ rotation(a, theta) @ np.linalg.inv(k)


def apply(H, pts):
    p = np.c_[pts[:, :2], np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


# ----------------------------------------------------------------------------- lens distortion
def _key(model):
    return re.sub(r"canon|\s|/", "", model.lower())              # "RF16mm F2.8 STM" and "Canon RF 16mm f/2.8 STM" agree


def find_lens(model):
    """the lensfun lens for an EXIF LensModel, or None when the profile is missing (then no correction)."""
    k = _key(model)
    hits = [l for l in _DB.lenses if _key(l.model) == k] or [l for l in _DB.lenses if k in _key(l.model)]
    return hits[0] if hits else None


def undistort_points(pts, coords):
    """source (distorted) positions -> corrected positions, by inverting the radial map `coords`
    ((h, w, 2), source pixel of each corrected pixel) along the diagonal."""
    h, w = coords.shape[:2]
    c = np.array([(w - 1) / 2, (h - 1) / 2], np.float32)
    t = np.linspace(0, 1, 2000)
    ux, uy = c[0] + t * (w - 1 - c[0]), c[1] + t * (h - 1 - c[1])           # corrected radii along the diagonal ...
    src = coords[np.round(uy).astype(int), np.round(ux).astype(int)]
    r_u = np.hypot(ux - c[0], uy - c[1]); r_s = np.hypot(src[:, 0] - c[0], src[:, 1] - c[1])   # ... and their source radii
    r = np.linalg.norm(pts - c, axis=1)
    scale = np.interp(r, r_s, r_u) / np.maximum(r, 1e-6)
    return c + (pts - c) * scale[:, None]


def undist_map(w, h, model, focal):
    """(h, w, 2): for each corrected pixel the source pixel (x, y); identity when the lens is unknown."""
    lens = find_lens(model)
    if lens is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return np.stack([xx, yy], -1)
    mod = lensfunpy.Modifier(lens, 1.0, w, h)
    mod.initialize(float(focal), 4.0, 1000.0, pixel_format=np.float32, flags=lensfunpy.ModifyFlags.DISTORTION)
    return mod.apply_geometry_distortion().astype(np.float32)
