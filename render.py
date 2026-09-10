"""The one engine every output uses. Sentence: the sky at any time t is the reference
stack rotated by the fitted model, and any original frame reaches the reference geometry
through lensfun-map ∘ fitted-D ∘ H.

Coordinates: FITS orientation everywhere (row 0 = bottom of picture) until export, where
frames are flipped for display.
"""
import json, numpy as np, cv2
from astropy.io import fits
import register as R
import fitmodel as F
import warp2

W = R.W
NAME = R.P.NAME
WIDTH, HEIGHT = R.WIDTH, R.HEIGHT
BLACK = 2047.0 / 65535.0
WB = np.array(R.P.WB, np.float32) / 1024.0
_m = np.load(f"{W}/model.npz")
K1, K2, FPX, POLE = float(_m["k1"]), float(_m["k2"]), float(_m["f"]), tuple(_m["pole"])
H_FRAME = _m["H"]                       # index 1..90 (odd CR3 frames), d-space, frame -> reference
TIMES = R.frame_times()                 # seconds relative to the reference frame (45)
EXPOSURE = R.P.EXPOSURE
P = R.P


def _angle(Hm):
    """Rotation angle of a fitted frame homography about the pole axis."""
    k = np.array([[FPX, 0, R.CX], [0, FPX, R.CY], [0, 0, 1.0]])
    Rm = np.linalg.inv(k) @ Hm @ k
    Rm /= np.cbrt(np.linalg.det(Rm))
    rv, _ = cv2.Rodrigues(Rm)
    a = np.linalg.inv(k) @ np.array([POLE[0], POLE[1], 1.0]); a /= np.linalg.norm(a)
    return float(rv.ravel() @ a)


_idx = [i for i in range(1, R.P.N + 1) if _m["have"][i]]
_ts = np.array([TIMES[i] for i in _idx]); _th = np.array([_angle(H_FRAME[i]) for i in _idx])
RATE, TH0 = np.polyfit(_ts, _th, 1)          # rad/s and offset, fitted from the frames themselves


def theta(t):
    """Sky rotation angle (rad) at time t seconds after the reference frame."""
    return RATE * t + TH0


def H_at(t):
    """Homography taking the sky as seen at time t into the reference (stack) geometry."""
    return F.Hmat(theta(t), FPX, POLE)


def load_fits(name):
    return fits.getdata(f"{W}/{name}.fit").astype(np.float32)          # (3,H,W)


def load_full(i):
    """Original-geometry sharp frame i (1..90), float 0..1."""
    return fits.getdata(R.P.full(i)).astype(np.float32) / 65535.0


def scale_H(Hm, s):
    S = np.diag([s, s, 1.0])
    return S @ Hm @ np.linalg.inv(S)


def to_d(img, Hm=np.eye(3), scale=1.0, interp=cv2.INTER_LANCZOS4):
    """Original-geometry (3,H,W) -> reference geometry at `scale` (3, H*s, W*s)."""
    ox, oy = warp2.total_map(Hm)
    if scale != 1.0:
        h, w = int(round(HEIGHT * scale)), int(round(WIDTH * scale))
        ox = cv2.resize(ox, (w, h), interpolation=cv2.INTER_LINEAR) * scale
        oy = cv2.resize(oy, (w, h), interpolation=cv2.INTER_LINEAR) * scale
        img = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA) for c in img])
    return np.stack([cv2.remap(c, ox, oy, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in img])


def warp_d(img_d, Hm, interp=cv2.INTER_LANCZOS4):
    """Warp an image already in reference geometry (any scale) by homography Hm (full-res units)."""
    h, w = img_d.shape[-2:]
    s = w / WIDTH
    Hs = scale_H(Hm, s)
    return np.stack([cv2.warpPerspective(c, Hs, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in img_d])


def downscale(img, scale):
    if scale == 1.0:
        return img
    h, w = img.shape[-2:]
    return np.stack([cv2.resize(c, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) for c in img])


def blend(sky, ground, m):
    """sky/ground (3,H,W) raw-with-black, m (H,W) in [0,1]. Returns black-subtracted linear signal."""
    nodata = sky[1] <= 0
    sky = sky - BLACK; ground = ground - BLACK
    sky[:, nodata] = ground[:, nodata]
    core = m > 0.999
    for c in range(3):
        sky[c] += np.median(ground[c][core]) - np.median(sky[c][core])
    return sky * m[None] + ground * (1 - m[None])


def feathered_mask(mask, valid, shrink=20, feather=8):
    from scipy import ndimage as ndi
    m = mask & valid
    if shrink > 0:                                   # scipy reads iterations=0 as "until nothing changes"
        m = ndi.binary_erosion(m, iterations=shrink)
    return ndi.gaussian_filter(m.astype(np.float32), feather) if feather > 0 else m.astype(np.float32)


# ---- the stationary-background model: only stars rotate, everything else is fixed to the camera
def large_scale(img, valid, scale=4, size=31):
    """Smooth background of a (3,H,W) image over `valid` pixels, ~size*scale px wide."""
    from scipy import ndimage as ndi
    out = np.empty_like(img)
    for c in range(3):
        s = img[c][::scale, ::scale]; v = valid[::scale, ::scale]
        fill = float(np.median(s[v])) if v.any() else 0.0
        b = ndi.median_filter(np.where(v, s, fill), size=size)
        out[c] = cv2.resize(b, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_LINEAR)
    return out


def star_layer(sky, valid=None, count=None, k=3.0, full=60.0):
    """Stars only: stack minus its large-scale background, thresholded above the local noise
    (noise grows as sqrt(full / frames contributing) where `count` is given)."""
    valid = (sky[1] > 0) if valid is None else valid
    bg = large_scale(sky, valid)
    d = sky - bg
    noise = 1.4826 * np.median(np.abs(d[1][valid][::37]))
    thr = k * noise
    if count is not None:
        thr = thr * np.sqrt(full / np.maximum(count, 1.0))[None]
    ratio = d / np.maximum(thr, 1e-9)
    is_star = (ratio.max(axis=0) > 1.0) & (ratio.min(axis=0) > 0.35)   # broadband point, not one-channel noise
    if count is not None:
        is_star &= count >= 8                              # too few frames: background steps masquerade as stars
    stars = np.where(is_star[None], np.clip(d, 0, None), 0)
    stars[:, ~valid] = 0
    return stars, bg


def backgrounds(sky, ground, mask, count, reach=400):
    """Sky background for the composite: the static stack's background near the treeline (the
    horizon glow is fixed to the ground), the aligned stack's background higher up (the Milky Way
    rotates with the stars), blended over `reach` px of distance from the treeline."""
    from scipy import ndimage as ndi
    valid = sky[1] > 0
    stars, bg_sky = star_layer(sky, valid, count)
    bg_ground = large_scale(ground, mask)
    dist = ndi.distance_transform_edt(mask[::4, ::4]) * 4
    w = np.clip(dist / reach, 0, 1).astype(np.float32)
    w = cv2.resize(w, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    bg = bg_ground * (1 - w[None]) + bg_sky * w[None]
    resid_sky = np.where(valid[None], sky - bg_sky - stars, 0)      # aligned stack's own grain
    resid_ground = np.where(mask[None], ground - bg_ground, 0)      # static median stack's grain
    wc = np.clip((count.astype(np.float32) - 4) / 26, 0, 1)[None]    # trust the aligned stack from ~30 frames up
    resid = wc * resid_sky + (1 - wc) * resid_ground
    return stars, bg, resid


def compose(layer, bg, ground, mask, resid, valid, shrink=0, feather=1):
    """Composite in black-subtracted linear signal. Inside the sky: stationary background + the star
    layer (or trails) + the aligned stack's own residual, so the sky keeps its natural grain; outside:
    the static stack. Both sides share the same background, so there is no seam to hide."""
    m = feathered_mask(mask, valid, shrink, feather)[None]
    skyimg = bg + layer + resid
    return (skyimg * m + ground * (1 - m)) - BLACK


class Tone:
    """One fixed tone curve for every still and every video frame: linear signal -> 8-bit sRGB-ish."""
    path = f"{W}/tone.json"

    def __init__(self, black=None, white=None, a=None):
        if black is None:
            p = json.load(open(self.path))
            black, white, a = p["black"], p["white"], p["a"]
        self.black, self.white, self.a = np.array(black, np.float32), np.array(white, np.float32), a

    @classmethod
    def fit(cls, lin, a=60.0, lo=0.2, sky_target=0.16, sky_mask=None, path=None):
        """lin: (3,H,W) black-subtracted linear signal. Black at a low percentile; white chosen so the
        sky background lands at `sky_target` of the output range."""
        black = np.array([np.percentile(lin[c][::7, ::7], lo) for c in range(3)], np.float32)
        g = lin[1][::7, ::7]
        g = g if sky_mask is None else g[sky_mask]
        sky = float(np.median(g)) - black[1]
        x_sky = np.sinh(sky_target * np.arcsinh(a)) / a
        white = sky / x_sky
        t = cls(black.tolist(), [white] * 3, a)
        json.dump({"black": black.tolist(), "white": [white] * 3, "a": a}, open(path or cls.path, "w"))
        return t

    def apply(self, lin):
        x = (lin - self.black[:, None, None]) * WB[:, None, None] / (self.white[:, None, None] * WB[1])
        x = np.clip(x, 0, 1)
        y = np.arcsinh(self.a * x) / np.arcsinh(self.a)
        return (np.clip(y, 0, 1) ** (1 / 1.15) * 255).astype(np.uint8)


def to_display(img8):
    """(3,H,W) uint8 FITS orientation -> (H,W,3) top-down RGB."""
    return np.moveaxis(img8, 0, -1)[::-1]
