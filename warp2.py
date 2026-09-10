"""Single-resampling warp: final pixel q -> u = D^-1(H_i^-1 q) -> original o = L(u) -> sample.
Writes r2_sky_NNNNN.fit (float 0..1, zero = no-data with an 8 px rim), ground_d.fit, mask_sky_d.npy.
"""
import numpy as np, cv2
from multiprocessing import Pool
from astropy.io import fits
import register as R
from fitmodel import R0

W = R.W
RIM = 1
m = np.load(f"{W}/model.npz")
k1, k2, H_all, have = float(m["k1"]), float(m["k2"]), m["H"], m["have"]
L = np.load(f"{W}/undist_coords.npy")           # u -> o, (H, W, 2)
Lx, Ly = np.ascontiguousarray(L[..., 0]), np.ascontiguousarray(L[..., 1])

# D^-1 as a lookup on rho: r_d/R0 = rho (1 + k1 rho^2 + k2 rho^4)
_rho = np.linspace(0, 1.2, 4001)
_rd = _rho * (1 + k1 * _rho ** 2 + k2 * _rho ** 4)


def total_map(Hi, pad=0, scale=1.0):
    """Map from final grid (optionally padded by `pad` px on every side, optionally at `scale` of full
    resolution) to original-frame coordinates (always full-resolution units)."""
    if scale != 1.0:
        h, w = int(round((R.HEIGHT + 2 * pad) * scale)), int(round((R.WIDTH + 2 * pad) * scale))
        yy, xx = (np.mgrid[0:h, 0:w].astype(np.float32) / scale) - pad
    else:
        yy, xx = np.mgrid[-pad:R.HEIGHT + pad, -pad:R.WIDTH + pad].astype(np.float32)
    Hinv = np.linalg.inv(Hi)
    den = Hinv[2, 0] * xx + Hinv[2, 1] * yy + Hinv[2, 2]
    ux = (Hinv[0, 0] * xx + Hinv[0, 1] * yy + Hinv[0, 2]) / den
    uy = (Hinv[1, 0] * xx + Hinv[1, 1] * yy + Hinv[1, 2]) / den
    dx, dy = ux - R.CX, uy - R.CY
    rd = np.sqrt(dx * dx + dy * dy) / R0
    rho = np.interp(rd, _rd, _rho).astype(np.float32)
    s = np.where(rd > 0, rho / np.maximum(rd, 1e-9), 1.0).astype(np.float32)
    ux, uy = (R.CX + dx * s).astype(np.float32), (R.CY + dy * s).astype(np.float32)
    ox = cv2.remap(Lx, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)
    oy = cv2.remap(Ly, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)
    return ox, oy


def filled(d, zero):
    """Blanked pixels take the local sky level so interpolation sees continuous data (no dark ring)."""
    from scipy import ndimage as ndi
    v = (1 - zero).astype(np.float32)[::8, ::8]
    out = np.empty_like(d)
    for c in range(3):
        num = ndi.gaussian_filter(d[c][::8, ::8] * v, 6); den = ndi.gaussian_filter(v, 6)
        fill = cv2.resize(num / np.maximum(den, 1e-6), (d.shape[2], d.shape[1]), interpolation=cv2.INTER_LINEAR)
        out[c] = np.where(zero > 0, fill, d[c])
    return out


def warp(d, ox, oy):
    """d: (3,H,W) float32 with 0 = no-data. Returns warped with a 1 px zero rim."""
    zero = (d == 0).any(axis=0).astype(np.uint8)
    d = filled(d, zero)
    out = np.stack([np.clip(cv2.remap(c, ox, oy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, None) for c in d])
    zw = cv2.remap(zero, ox, oy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    zw = cv2.dilate(zw, np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8))
    out[:, zw > 0] = 0
    return out


def one(i):
    if not have[i]:
        return i, False
    d = fits.getdata(R.P.light(i)).astype(np.float32) / 65535.0
    ox, oy = total_map(H_all[i])
    fits.PrimaryHDU(warp(d, ox, oy)).writeto(f"{W}/r2_sky_{i:05d}.fit", overwrite=True)
    return i, True


def main():
    ox, oy = total_map(np.eye(3))
    g = fits.getdata(f"{W}/ground.fit").astype(np.float32)
    fits.PrimaryHDU(np.stack([cv2.remap(c, ox, oy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in g])).writeto(f"{W}/ground_d.fit", overwrite=True)
    mk = np.load(f"{W}/mask_sky.npy").astype(np.uint8)
    np.save(f"{W}/mask_sky_d.npy", cv2.remap(mk, ox, oy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0)
    gm = fits.getdata(f"{W}/ground_mean.fit").astype(np.float32)
    fits.PrimaryHDU(np.stack([cv2.remap(c, ox, oy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in gm])).writeto(f"{W}/ground_mean_d.fit", overwrite=True)
    print("ground_d + mask_sky_d written", flush=True)
    with Pool(5) as p:
        for i, ok in p.imap_unordered(one, range(1, R.P.N + 1)):
            print(f"frame {i:3d} {'warped' if ok else 'skipped'}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
