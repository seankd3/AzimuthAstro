"""Two-minute synthetic smoke test of the chain. Builds a small project (1200 x 800 frames) with a
known star field rotating about a known pole, a jagged treeline, a lit lamp on the shore, a passing
cloud, hot pixels and noise; writes full_NNNNN.fit + project.json (lens "none": no distortion) and runs
the stages hot..tone through azastro, then checks the recovered pole and the composite.

  python smoke.py [workdir]      (default D:/AstroWork/_smoke)
"""
import os, sys, json, shutil, subprocess, numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optics

W = sys.argv[1] if len(sys.argv) > 1 else r"D:\AstroWork\_smoke"
HERE = os.path.dirname(os.path.abspath(__file__))
WID, HEI, N, EXPO, CAD = 1200, 800, 12, 8.0, 300.0      # 55 min of sky rotation: enough to pin the pole
POLE = (420.0, 200.0)                   # display px (x right, y down)
FOCAL, CROP = 16.0, 1.0                 # a 16 mm lens on a 36 mm sensor: the sky turns as a pinhole homography, not rigidly
F_PX = optics.focal_px(FOCAL, WID, CROP)
CX, CY = (WID - 1) / 2, (HEI - 1) / 2
OMEGA = 2 * np.pi / 86164.0905
rng = np.random.default_rng(1)


def scene():
    """static ground (display orientation, y down): treeline + lamp + lake; sky mask."""
    x = np.arange(WID)
    tree = 520 + 40 * np.sin(x / 37.0) + 25 * np.sin(x / 9.0) + rng.normal(0, 3, WID)
    yy, xx = np.mgrid[0:HEI, 0:WID]
    ground = yy > tree[None, :]
    img = np.zeros((3, HEI, WID), np.float32)
    img[:, ~ground] = 12.0                                             # sky level (ADU above black)
    img[:, ground] = 3.0
    lamp = np.exp(-((xx - 300) ** 2 + (yy - 560) ** 2) / (2 * 12 ** 2)) * 2000
    img += lamp[None] * np.array([1.0, 0.8, 0.5])[:, None, None]
    img[:, yy > 640] += 6.0                                             # lake reflection glow
    return img, ~ground


def stars(n=900):
    """reference-frame positions (spread beyond the frame so stars turn in) and fluxes"""
    pts = np.c_[rng.uniform(-400, WID + 400, n), rng.uniform(-400, HEI + 400, n)]
    flux = rng.pareto(1.5, n) * 40 + 20
    return pts, flux


def render_frame(i, base, sky, pts, F, t):
    img = base.copy()
    theta = -OMEGA * t                                                  # counter-clockwise on screen, like the northern sky
    q = optics.apply(optics.sky_homography(theta, F_PX, POLE, CX, CY), pts)
    x, y = q[:, 0], q[:, 1]
    yy, xx = np.mgrid[0:HEI, 0:WID]
    for xs, ys, f in zip(x, y, F):
        if -5 < xs < WID + 5 and -5 < ys < HEI + 5:
            x0, x1, y0, y1 = int(max(0, xs - 4)), int(min(WID, xs + 5)), int(max(0, ys - 4)), int(min(HEI, ys + 5))
            g = np.exp(-((xx[y0:y1, x0:x1] - xs) ** 2 + (yy[y0:y1, x0:x1] - ys) ** 2) / (2 * 1.2 ** 2)) * f
            img[:, y0:y1, x0:x1] += g[None] * np.array([0.9, 1.0, 1.1])[:, None, None]
    img[:, ~sky] = base[:, ~sky]                                        # stars do not show through trees
    if 5 <= i <= 7:                                                     # a passing cloud
        cx = 200 + 150 * (i - 5)
        cloud = np.exp(-((xx - cx) ** 2 / (2 * 120 ** 2) + (yy - 250) ** 2 / (2 * 60 ** 2))) * 6
        img[:, sky] += cloud[sky][None]
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    return img


def main():
    if os.path.exists(W):
        shutil.rmtree(W)
    os.makedirs(W)
    base, sky = scene()
    pts, F = stars()
    hot = rng.integers(0, HEI * WID, 300)
    times = [i * CAD for i in range(N)]
    ref = N // 2 + 1
    for i in range(1, N + 1):
        img = render_frame(i, base, sky, pts, F, times[i - 1] - times[ref - 1])
        flat = img.reshape(3, -1); flat[:, hot] += 400                     # hot pixels, same place every frame
        raw = np.clip(img + 2047, 0, 16383)                                # raw ADU with black level
        fits.PrimaryHDU(raw[:, ::-1, :].astype(np.uint16)).writeto(f"{W}/full_{i:05d}.fit", overwrite=True)   # FITS rows bottom-up
    cfg = {"name": "Smoke", "width": WID, "height": HEI, "frame_ids": list(range(1, N + 1)), "ref": ref, "exposure": EXPO,
           "times": [t - times[ref - 1] for t in times], "pole_display": [POLE[0] + 15, POLE[1] - 10], "sky_rows": 400,
           "orientation": "Horizontal (normal)", "wb": [1024, 1024, 1024], "date": "2026-01-01T00:00:00",
           "lens": "none", "focal": FOCAL, "crop": CROP,
           "source_folder": W, "sources": [f"full_{i:05d}.fit" for i in range(1, N + 1)]}
    json.dump(cfg, open(f"{W}/project.json", "w"), indent=1)
    env = dict(os.environ, ASTRO_WORK=W.replace("\\", "/"), SMOKE="1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "azastro.py"), "run", W, "--from", "hot", "--to", "tone"], env=env)
    m = np.load(f"{W}/model.npz")
    err = np.hypot(m["pole"][0] - POLE[0], (HEI - 1 - m["pole"][1]) - POLE[1])
    print(f"pole recovered within {err:.1f} px; run rc {r.returncode}")
    sys.exit(0 if r.returncode == 0 and err < 5 else 1)


if __name__ == "__main__":
    main()
