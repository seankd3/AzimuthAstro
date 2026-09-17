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
import warp

W = R.W
NAME = R.P.NAME
WIDTH, HEIGHT = R.WIDTH, R.HEIGHT
BLACK = 2047.0 / 65535.0
WB = np.array(R.P.WB, np.float32) / 1024.0
CAM = np.array(R.P.CAM_TO_SRGB, np.float32)      # white-balanced camera RGB -> linear sRGB
DAY = np.array(R.P.DAYLIGHT_WB, np.float32)      # the balance real light is rendered under
_m = np.load(f"{W}/model.npz")
K1, K2, FPX, POLE = float(_m["k1"]), float(_m["k2"]), float(_m["f"]), tuple(_m["pole"])
H_FRAME = _m["H"]                       # index 1..90 (odd CR3 frames), d-space, frame -> reference
TIMES = R.frame_times()                 # seconds relative to the reference frame (45)
EXPOSURE = R.P.EXPOSURE
PAD = warp.PAD
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


def load_warped(i):
    """warped frame i in reference geometry (the centre of its padded canvas), float32 0..1."""
    p = np.load(f"{W}/warped/{i:05d}.npy", mmap_mode="r")
    return np.array(p[:, PAD:PAD + HEIGHT, PAD:PAD + WIDTH], np.float32)


def load_sky():
    """the aligned sky stack in reference geometry: the centre of the padded stack."""
    p = np.load(f"{W}/padded_stack.npy", mmap_mode="r")
    return np.array(p[:, PAD:PAD + HEIGHT, PAD:PAD + WIDTH], np.float32)


def load_full(i):
    """Original-geometry sharp frame i (1..90), float 0..1."""
    return fits.getdata(R.P.full(i)).astype(np.float32) / 65535.0


def scale_H(Hm, s):
    S = np.diag([s, s, 1.0])
    return S @ Hm @ np.linalg.inv(S)


def to_d(img, Hm=np.eye(3), scale=1.0, interp=cv2.INTER_LANCZOS4):
    """Original-geometry (3,H,W) -> reference geometry at `scale` (3, H*s, W*s)."""
    ox, oy = warp.total_map(Hm)
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
    stars = np.where(is_star[None], np.clip(d - thr, 0, None), 0)   # what stands above the noise, not the whole residual:
                                                                    # kept in full, a 4-sigma noise floor accumulates into a magenta wash
    stars[:, ~valid] = 0
    return one_colour(stars), bg


def star_white(stars=None):
    """the integrated flux of the star layer over the sky: the camera-space colour of the average star."""
    if stars is None:
        stars = np.load(f"{W}/stars_layer.npy", mmap_mode="r")[:, ::5, ::5]
    v = np.asarray(stars).reshape(3, -1)
    on = v[1] > 0
    return v[:, on].sum(axis=1) if on.any() else 1.0 / DAY


def one_colour(stars, sigma=2.5):
    """A star has one colour. Lateral chromatic aberration and the demosaic paint a bright core's halo red and
    blue (R/G 0.67 in single frames against a daylight-neutral 0.46), and trails accumulate halos into a lavender
    wash; so the layer keeps its luminance at full resolution and takes its colour from a blurred copy, which is
    each star's flux-weighted colour."""
    from scipy import ndimage as ndi
    lum = stars.sum(axis=0)
    blur = np.stack([ndi.gaussian_filter(c, sigma) for c in stars])
    blum = ndi.gaussian_filter(lum, sigma)
    ratio = blur / np.maximum(blum, 1e-12)[None]
    return np.where(lum[None] > 0, lum[None] * ratio, 0).astype(np.float32)


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
    """One fixed tone curve for every still and every video frame: linear signal -> sRGB.

    Airglow and light pollution are emissions laid over the scene, so the sky's own colour comes off
    as a per-channel black point, not as a white balance. What is left is starlight, and the average
    star defines white: the camera's daylight multipliers left B and A stars magenta (30% too much red
    in point sources), the as-shot balance (chosen for the warm ground light) made every star violet,
    and balancing on the sky divided out its green airglow, which did the same.
    """
    path = f"{W}/tone.json"

    def __init__(self, black=None, white=None, a=None, wb=None):
        if black is None:
            p = json.load(open(self.path))
            black, white, a, wb = p["black"], p["white"], p["a"], p.get("wb")
        self.black, self.white, self.a = np.array(black, np.float32), np.array(white, np.float32), a
        self.wb = np.array(wb if wb is not None else DAY, np.float32)

    @classmethod
    def fit(cls, lin, a=60.0, sky_target=0.16, sky_mask=None, path=None, white_ref=None):
        """lin: (3,H,W) linear camera signal. `white_ref`: the camera-space colour that renders neutral (the
        integrated star flux, see star_white); daylight when absent. The black point is the sky background,
        offset per channel so the sky lands neutral at `sky_target` of the output range."""
        s = lin[:, ::7, ::7]
        s = s[:, sky_mask] if sky_mask is not None else s.reshape(3, -1)
        sky = np.median(s, axis=1)
        wb = DAY if white_ref is None else np.float32(white_ref[1]) / np.maximum(np.asarray(white_ref, np.float32), 1e-12)
        white = float(sky[1]) / (np.sinh(sky_target * np.arcsinh(a)) / a)
        black = sky - float(sky[1]) / wb
        p = {"black": black.tolist(), "white": [white] * 3, "a": a, "wb": wb.tolist()}
        json.dump(p, open(path or cls.path, "w"))
        return cls(**p)

    def apply(self, lin, bits=8):
        """black-subtracted camera-space linear -> sRGB, 8 or 16 bit."""
        x = (lin - self.black[:, None, None]) * self.wb[:, None, None] / self.white[:, None, None]
        x = np.clip(np.einsum("ij,jhw->ihw", CAM, x, optimize=True), 0, 1)     # camera RGB is not sRGB: the sky goes purple without this
        y = np.arcsinh(self.a * x) / np.arcsinh(self.a)
        peak, dt = (255, np.uint8) if bits == 8 else (65535, np.uint16)
        return (np.clip(y, 0, 1) ** (1 / 1.15) * peak).astype(dt)


def to_display(img8):
    """(3,H,W) uint8 FITS orientation -> (H,W,3) top-down RGB."""
    return np.moveaxis(img8, 0, -1)[::-1]


def footprint():
    """(r0, r1, c0, c1): the largest axis-aligned box of the reference geometry that holds data on every
    border line. The undistortion and the fitted model leave empty wedges at the frame edges."""
    g = load_fits("ground_d")[1] > 0
    r0, r1, c0, c1 = 0, g.shape[0], 0, g.shape[1]
    while r1 - r0 > 2 and c1 - c0 > 2:
        empty = [(~g[r0, c0:c1]).sum(), (~g[r1 - 1, c0:c1]).sum(), (~g[r0:r1, c0]).sum(), (~g[r0:r1, c1 - 1]).sum()]
        if max(empty) == 0:
            break
        k = int(np.argmax(empty))                                        # shrink the side with the most empty pixels
        r0, r1, c0, c1 = r0 + (k == 0), r1 - (k == 1), c0 + (k == 2), c1 - (k == 3)
    return r0, r1, c0, c1


def crop(a, box=None):
    r0, r1, c0, c1 = box or footprint()
    return a[..., r0:r1, c0:c1]


def save_still(name, tone, lin, box=None):
    """{NAME}_{name}.jpg and .tif: the same sRGB image at 8 and 16 bits, cropped to the data footprint."""
    import tifffile
    from PIL import Image
    lin = crop(lin, box or footprint())
    Image.fromarray(to_display(tone.apply(lin))).save(f"{W}/{NAME}_{name}.jpg", quality=94)
    tifffile.imwrite(f"{W}/{NAME}_{name}.tif", to_display(tone.apply(lin, bits=16)), photometric="rgb", compression="zlib")
