"""The numbers I otherwise measure by hand, in one screen: colour of the sky, the stars and the trails in
camera space and as delivered; per-channel noise; star coverage; frames per pixel; clouds; the model fit;
the footprint; the tone; and the sky colour of every delivered still and a frame of every video.
Read this before opening an image: it is 30 lines against 1,500 tokens a picture."""
import os, re, json, glob, subprocess, shutil
import numpy as np
from PIL import Image
import project as P

Image.MAX_IMAGE_PIXELS = None
W = P.W
DAYLIGHT = np.array(P.DAYLIGHT_WB, np.float32)
CAM = np.array(P.CAM_TO_SRGB, np.float32)


def have(name):
    return os.path.exists(f"{W}/{name}")


def ratios(v):
    """R/G and B/G of a channel triple, as a string."""
    return f"R/G {v[0] / max(v[1], 1e-9):.3f} B/G {v[2] / max(v[1], 1e-9):.3f}"


def delivered(v):
    """ratios of an 8-bit triple, or a note when it is too dark to carry a colour."""
    return ratios(v) if max(v) >= 6 else f"(too dark to judge: {np.round(v, 1).tolist()})"


def rendered(cam_rgb):
    """camera-space triple -> the sRGB triple it renders to under the daylight balance (linear, unscaled)."""
    return CAM @ (np.asarray(cam_rgb, np.float32) * DAYLIGHT)


def sky_of(path):
    """median RGB of the top 30% of rows of a still or a video frame, as delivered."""
    a = np.asarray(Image.open(path).convert("RGB"))[::8, ::8].astype(np.float32)
    return np.median(a[: a.shape[0] * 3 // 10].reshape(-1, 3), axis=0)


print(f"{P.NAME}: {P.N} frames {P.WIDTH}x{P.HEIGHT}, {P.EXPOSURE:.0f}s, pole {tuple(round(v) for v in P.POLE_DISPLAY)}, "
      f"daylight wb {[round(float(x), 3) for x in DAYLIGHT]}, neutral in camera space = {ratios(1 / DAYLIGHT)}")

if have("mask_sky_d.npy") and have("composite_lin.npy"):
    mask = np.load(f"{W}/mask_sky_d.npy")
    m7 = mask[::7, ::7]
    lin = np.load(f"{W}/composite_lin.npy", mmap_mode="r")[:, ::7, ::7]
    sky = np.median(np.asarray(lin)[:, m7], axis=1)
    print(f"sky background   camera {ratios(sky)}   -> rendered {ratios(rendered(sky))}")
    if have("stars_layer.npy"):
        st = np.load(f"{W}/stars_layer.npy", mmap_mode="r")[:, ::7, ::7]
        v = np.asarray(st)[:, m7]
        on = v[1] > 0
        tot = v[:, on].sum(axis=1) if on.any() else np.ones(3)
        print(f"star flux        camera {ratios(tot)}   -> rendered {ratios(rendered(tot))}   star pixels {100 * on.mean():.2f}% of sky")
    for name in ("trails_gapless", "trails_comet"):
        if have(f"{name}.npy"):
            t = np.asarray(np.load(f"{W}/{name}.npy", mmap_mode="r")[:, ::7, ::7])
            H = t.shape[1]; c = t[:, H // 2 - 100:H // 2 + 100, t.shape[2] // 2 - 100:t.shape[2] // 2 + 100].reshape(3, -1)
            print(f"{name:16s} centre {ratios(np.median(c, axis=1))}   sky-wide {ratios(np.median(t[:, m7], axis=1))}")

if have("padded_stack.npy"):
    import render as Rn
    sky_st = Rn.load_sky()[:, ::4, ::4]
    valid = sky_st[1] > 0
    d = sky_st - Rn.large_scale(sky_st, valid)
    n = np.array([1.4826 * np.median(np.abs(d[c][valid][::37])) for c in range(3)]) * 65535
    print(f"stack noise      {[round(float(x), 2) for x in n]} ADU per channel")

if have("count_map.npy") and have("mask_sky_d.npy"):
    c = np.load(f"{W}/count_map.npy")[mask]
    print(f"frames per pixel over sky: p5 {np.percentile(c, 5):.0f}  median {np.median(c):.0f}  max {c.max()} of {P.N}")

if have("clouds.npz"):
    z = np.load(f"{W}/clouds.npz"); ms = z["masks"]
    sk = np.load(f"{W}/mask_sky.npy")[::8, ::8][: ms.shape[1], : ms.shape[2]]
    fr = np.array([mm[sk].mean() for mm in ms])
    hi = [int(f) for f, x in zip(z["frames"], fr) if x > 0.05]
    print(f"clouds           mean {100 * fr.mean():.1f}% of sky; frames over 5%: {hi[:15]}{' ...' if len(hi) > 15 else ''}")

if have("log_fit.log"):
    txt = open(f"{W}/log_fit.log", errors="replace").read()
    tol = re.findall(r"^tol 3: .*$", txt, re.M)
    res = re.findall(r"r (\d+)-(\d+): n (\d+) median resid ([\d.]+) px", txt)
    print("model fit        " + (tol[-1][7:] if tol else "none") + "  |  " + "  ".join(f"r<{b}: {m}px" for a, b, n, m in res))

if have("tone.json"):
    t = json.load(open(f"{W}/tone.json"))
    fp = re.findall(r"footprint \((\d+), (\d+), (\d+), (\d+)\)", open(f"{W}/log_tone.log", errors="replace").read()) if have("log_tone.log") else []
    print(f"tone             black {[round(b * 65535, 1) for b in t['black']]} ADU  white {t['white'][0] * 65535:.0f} ADU  a {t['a']}"
          + (f"  footprint rows {fp[-1][0]}..{fp[-1][1]} cols {fp[-1][2]}..{fp[-1][3]}" if fp else ""))

stills = sorted(glob.glob(f"{W}/{P.NAME}_*.jpg"))
if stills:
    print("delivered stills, sky of the top third (neutral is 1.000 / 1.000):")
    for s in stills:
        print(f"  {os.path.basename(s):40s} {delivered(sky_of(s))}")
ff = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
videos = sorted(glob.glob(f"{W}/{P.NAME}_*_4K.mp4"))
if videos and os.path.exists(ff):
    print("delivered videos, one frame at 5 s:")
    for v in videos:
        frame = f"{W}/_probe_frame.png"
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "5", "-i", v, "-frames:v", "1", frame], capture_output=True)
        if os.path.exists(frame):
            print(f"  {os.path.basename(v):40s} {delivered(sky_of(frame))}  {os.path.getsize(v) / 1e6:.0f} MB")
            os.remove(frame)
