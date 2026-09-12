"""Full-frame astrometry and annotation. Sentence: the frame is a pinhole camera (fitted focal,
centre, residual radial term) whose orientation is fixed by the pole pixel and one ASTAP-solved
sky point; catalog stars then refine everything by least squares across all 97 degrees.

Writes astrometry.json and {Rn.NAME}_Annotated.jpg (half resolution).
"""
import os, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
import register as R
import render as Rn

W = Rn.W
SKY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sky_data")      # d3-celestial catalogues
DATE = Rn.P.DATE
ASTAP_CENTER = Rn.P.ASTAP_CENTER      # RA, Dec deg of pixel (1000.5, 800.5) of the crop = frame centre
R0 = np.hypot(R.CX, R.CY)


def radec_vec(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def precess(ra, dec):
    from astropy.coordinates import SkyCoord, FK5
    import astropy.units as u
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(FK5(equinox="J2026.68"))
    return c.ra.deg, c.dec.deg


def kabsch(a, b):
    """Rotation R with R a_i ~ b_i (rows unit vectors)."""
    Hm = a.T @ b
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


class Camera:
    def __init__(self, Rm, f, cx, cy, k1, k2):
        self.Rm, self.f, self.cx, self.cy, self.k1, self.k2 = Rm, f, cx, cy, k1, k2

    def project(self, v):
        c = v @ self.Rm.T
        z = -c[:, 2]                          # right-handed camera: x right, y up, looking down -z
        ok = z > 0.05
        x = np.where(ok, c[:, 0] / np.where(ok, z, 1), 0) * self.f
        y = np.where(ok, c[:, 1] / np.where(ok, z, 1), 0) * self.f
        rho2 = (x * x + y * y) / R0 ** 2
        s = 1 + self.k1 * rho2 + self.k2 * rho2 * rho2
        return np.c_[self.cx + x * s, self.cy + y * s], ok

    def params(self):
        rv, _ = cv2.Rodrigues(self.Rm)
        return np.r_[rv.ravel(), self.f, self.cx, self.cy, self.k1, self.k2]

    @classmethod
    def from_params(cls, p):
        Rm, _ = cv2.Rodrigues(np.array(p[:3]))
        return cls(Rm, p[3], p[4], p[5], p[6], p[7])


def initial_camera():
    pole = np.array([Rn.POLE[0] - R.CX, Rn.POLE[1] - R.CY, -Rn.FPX]); pole /= np.linalg.norm(pole)
    centre = np.array([0.0, 0.0, -1.0])
    ra, dec = precess(*ASTAP_CENTER)
    sky = np.stack([radec_vec(0, 90), radec_vec(ra, dec)])
    cam = np.stack([pole, centre])
    return Camera(kabsch(sky, cam), Rn.FPX, R.CX, R.CY, 0.0, 0.0)


def detect_stack(sky):
    """Star list from the stack (x, y, flux), FITS orientation."""
    g = sky[1]; valid = g > 0
    bg = ndi.median_filter(g[::4, ::4], 15)
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    hp = np.where(valid, g - bg, 0)
    sig = 1.4826 * np.median(np.abs(hp[valid][::97]))
    sm = ndi.gaussian_filter(hp, 1.0)
    peak = (sm == ndi.maximum_filter(sm, size=7)) & (sm > 5 * sig) & valid
    ys, xs = np.nonzero(peak); flux = sm[ys, xs]
    order = np.argsort(-flux)[:6000]
    out = []
    for y, x, f in zip(ys[order], xs[order], flux[order]):
        y0, y1, x0, x1 = max(0, y - 3), min(R.HEIGHT, y + 4), max(0, x - 3), min(R.WIDTH, x + 4)
        w = np.clip(hp[y0:y1, x0:x1], 0, None); s = w.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out.append(((w * xx).sum() / s, (w * yy).sum() / s, f))
    return np.array(out)


def load_catalog():
    st = json.load(open(f"{SKY}/stars.6.json", encoding="utf-8"))
    ra = np.array([f["geometry"]["coordinates"][0] for f in st["features"]], float) % 360
    dec = np.array([f["geometry"]["coordinates"][1] for f in st["features"]], float)
    mag = np.array([f["properties"]["mag"] for f in st["features"]], float)
    ra, dec = precess(ra, dec)
    return ra, dec, mag


def refine(cam, cat_v, cat_mag, det, tols=(25, 10, 5, 3)):
    tree = cKDTree(det[:, :2])
    p = cam.params()
    for tol in tols:
        c = Camera.from_params(p)
        proj, ok = c.project(cat_v)
        inside = ok & (proj[:, 0] > 0) & (proj[:, 0] < R.WIDTH) & (proj[:, 1] > 0) & (proj[:, 1] < R.HEIGHT)
        d, j = tree.query(proj, distance_upper_bound=tol)
        m = inside & np.isfinite(d)
        src = cat_v[m]; dst = det[j[m], :2]
        def resid(q):
            pr, _ = Camera.from_params(q).project(src)
            return (pr - dst).ravel()
        res = least_squares(resid, p, loss="soft_l1", f_scale=2.0, max_nfev=300)
        p = res.x
        e = np.hypot(*(resid(p).reshape(-1, 2).T))
        print(f"tol {tol}: matches {m.sum()} median resid {np.median(e):.2f} px", flush=True)
    return Camera.from_params(p), m.sum(), float(np.median(e))


def main():
    sky = Rn.load_fits("sky")
    det = detect_stack(sky)
    print("detected stars", len(det), flush=True)
    ra, dec, mag = load_catalog()
    cat_v = radec_vec(ra, dec)
    cam = initial_camera()
    bright = mag < 5.5
    cam, n, med = refine(cam, cat_v[bright], mag[bright], det)
    cam, n, med = refine(cam, cat_v, mag, det, tols=(4, 3))
    p = cam.params()
    json.dump({"rodrigues": p[:3].tolist(), "f": p[3], "cx": p[4], "cy": p[5], "k1": p[6], "k2": p[7],
               "matches": int(n), "median_resid_px": med, "astap_center": ASTAP_CENTER}, open(f"{W}/astrometry.json", "w"), indent=1)
    annotate(cam)


def annotate(cam, scale=0.5):
    lin = np.load(f"{W}/composite_lin.npy")
    img8 = Rn.Tone().apply(lin)
    base = Image.fromarray(Rn.to_display(img8)).resize((int(R.WIDTH * scale), int(R.HEIGHT * scale)), Image.LANCZOS)
    im = Image.new("RGBA", base.size, (0, 0, 0, 0))          # overlay layer, clipped to the sky at the end
    draw = ImageDraw.Draw(im, "RGBA")
    H = im.height
    def px(ra, dec):
        pr, ok = cam.project(radec_vec(np.atleast_1d(ra), np.atleast_1d(dec)))
        return np.c_[pr[:, 0] * scale, H - 1 - pr[:, 1] * scale], ok
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 30)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    lines = json.load(open(f"{SKY}/constellations.lines.json", encoding="utf-8"))
    for f in lines["features"]:
        for seg in f["geometry"]["coordinates"]:
            seg = np.array(seg, float)
            ra, dec = precess(seg[:, 0] % 360, seg[:, 1])
            pts, ok = px(ra, dec)
            near = ok & (pts[:, 0] > -0.15 * im.width) & (pts[:, 0] < 1.15 * im.width) & (pts[:, 1] > -0.15 * H) & (pts[:, 1] < 1.15 * H)
            for a, b in zip(range(len(pts) - 1), range(1, len(pts))):
                if near[a] and near[b]:
                    draw.line([tuple(pts[a]), tuple(pts[b])], fill=(120, 200, 255, 120), width=2)
                    for q in (a, b):
                        draw.ellipse([pts[q][0] - 4, pts[q][1] - 4, pts[q][0] + 4, pts[q][1] + 4], outline=(120, 200, 255, 160), width=1)
    names = json.load(open(f"{SKY}/constellations.json", encoding="utf-8"))
    for f in names["features"]:
        ra, dec = precess(f["geometry"]["coordinates"][0] % 360, f["geometry"]["coordinates"][1])
        pts, ok = px(ra, dec)
        x, y = pts[0]
        if ok[0] and 0 < x < im.width and 0 < y < H:
            draw.text((x, y), f["properties"]["name"], fill=(160, 220, 255, 200), font=font, anchor="mm")
    dsos = json.load(open(f"{SKY}/dsos.bright.json", encoding="utf-8"))
    for f in dsos["features"]:
        ra, dec = precess(f["geometry"]["coordinates"][0] % 360, f["geometry"]["coordinates"][1])
        pts, ok = px(ra, dec)
        x, y = pts[0]
        if ok[0] and 0 < x < im.width and 0 < y < H:
            name = f["properties"].get("name") or f["properties"].get("desig") or f["id"]
            draw.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 210, 120, 220), width=2)
            draw.text((x + 18, y), name, fill=(255, 220, 140, 230), font=small, anchor="lm")
    # Polaris and the pole
    for nm, (ra, dec), col in (("NCP", (0.0, 90.0), (255, 120, 120, 230)),):
        pts, ok = px(*precess(ra, dec))
        x, y = pts[0]
        draw.line([(x - 16, y), (x + 16, y)], fill=col, width=2); draw.line([(x, y - 16), (x, y + 16)], fill=col, width=2)
        draw.text((x + 20, y - 20), nm, fill=col, font=small, anchor="lm")
    # bright star names
    names_map = json.load(open(f"{SKY}/starnames.json", encoding="utf-8"))
    st = json.load(open(f"{SKY}/stars.6.json", encoding="utf-8"))
    for f in st["features"]:
        if f["properties"]["mag"] > 2.6:
            continue
        nm = (names_map.get(str(f["id"])) or {}).get("name")
        if not nm:
            continue
        ra, dec = precess(f["geometry"]["coordinates"][0] % 360, f["geometry"]["coordinates"][1])
        pts, ok = px(ra, dec)
        x, y = pts[0]
        if ok[0] and 0 < x < im.width and 0 < y < H:
            draw.text((x + 12, y + 12), nm, fill=(255, 255, 255, 215), font=small, anchor="lm")
    skym = np.load(f"{W}/mask_sky_d.npy")[::-1]
    skym = cv2.resize(skym.astype(np.uint8), im.size, interpolation=cv2.INTER_NEAREST) > 0
    a = np.array(im); a[..., 3] = np.where(skym, a[..., 3], 0)
    out = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(a)).convert("RGB")
    out.save(f"{W}/{Rn.NAME}_Annotated.jpg", quality=93)
    print("annotated")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "annotate":
        a = json.load(open(f"{W}/astrometry.json"))
        annotate(Camera.from_params(np.array(a["rodrigues"] + [a[k] for k in ("f", "cx", "cy", "k1", "k2")])))
    else:
        main()
