"""The numbers I otherwise measure by hand, in one screen: colour of the sky, the stars and the trails in
camera space and as rendered; per-channel noise; star coverage; frames per pixel; clouds; the model fit;
the footprint; the tone; and the sky colour of every delivered still and a frame of every video.
Read this before opening an image: it is 30 lines against 1,500 tokens a picture.
`--json` prints the same facts as one object: the interface a wrapper (an Azimuth Photo plugin) reads."""
import os, re, sys, json, glob, subprocess, shutil
import numpy as np
from PIL import Image
import project as P

Image.MAX_IMAGE_PIXELS = None
W = P.W
JSON = "--json" in sys.argv
DAYLIGHT = np.array(P.DAYLIGHT_WB, np.float32)
CAM = np.array(P.CAM_TO_SRGB, np.float32)
TONE_WB = np.array(json.load(open(f"{W}/tone.json"))["wb"], np.float32) if os.path.exists(f"{W}/tone.json") else DAYLIGHT
facts = {"name": P.NAME, "frames": P.N, "width": P.WIDTH, "height": P.HEIGHT, "exposure": P.EXPOSURE,
         "pole_display": [float(v) for v in P.POLE_DISPLAY], "daylight_wb": [round(float(x), 4) for x in DAYLIGHT],
         "tone_wb": [round(float(x), 4) for x in TONE_WB]}


def have(name):
    return os.path.exists(f"{W}/{name}")


def rg_bg(v):
    return [round(float(v[0] / max(v[1], 1e-9)), 4), round(float(v[2] / max(v[1], 1e-9)), 4)]


def ratios(v):
    r = rg_bg(v)
    return f"R/G {r[0]:.3f} B/G {r[1]:.3f}"


def delivered(v):
    """ratios of an 8-bit triple, or a note when it is too dark to carry a colour."""
    return ratios(v) if max(v) >= 6 else f"(too dark to judge: {np.round(v, 1).tolist()})"


def rendered(cam_rgb):
    """camera-space triple -> the sRGB triple it renders to under the tone's balance (linear, unscaled)."""
    return CAM @ (np.asarray(cam_rgb, np.float32) * TONE_WB)


def sky_of(path):
    """median RGB of the top 30% of rows of a still or a video frame, as delivered."""
    a = np.asarray(Image.open(path).convert("RGB"))[::8, ::8].astype(np.float32)
    return np.median(a[: a.shape[0] * 3 // 10].reshape(-1, 3), axis=0)


def say(text):
    if not JSON:
        print(text)


say(f"{P.NAME}: {P.N} frames {P.WIDTH}x{P.HEIGHT}, {P.EXPOSURE:.0f}s, pole {tuple(round(v) for v in P.POLE_DISPLAY)}, "
    f"tone wb {facts['tone_wb']} (daylight {facts['daylight_wb']}), neutral in camera space = {ratios(1 / TONE_WB)}")

if have("mask_sky_d.npy") and have("composite_lin.npy"):
    mask = np.load(f"{W}/mask_sky_d.npy")
    m7 = mask[::7, ::7]
    lin = np.load(f"{W}/composite_lin.npy", mmap_mode="r")[:, ::7, ::7]
    sky = np.median(np.asarray(lin)[:, m7], axis=1)
    facts["sky"] = {"camera": rg_bg(sky), "rendered": rg_bg(rendered(sky))}
    say(f"sky background   camera {ratios(sky)}   -> rendered {ratios(rendered(sky))}")
    if have("stars_layer.npy"):
        st = np.load(f"{W}/stars_layer.npy", mmap_mode="r")[:, ::7, ::7]
        v = np.asarray(st)[:, m7]
        on = v[1] > 0
        tot = v[:, on].sum(axis=1) if on.any() else np.ones(3)
        facts["stars"] = {"camera": rg_bg(tot), "rendered": rg_bg(rendered(tot)), "pixel_fraction": round(float(on.mean()), 5)}
        say(f"star flux        camera {ratios(tot)}   -> rendered {ratios(rendered(tot))}   star pixels {100 * on.mean():.2f}% of sky")
    for name in ("trails_gapless", "trails_comet"):
        if have(f"{name}.npy"):
            t = np.asarray(np.load(f"{W}/{name}.npy", mmap_mode="r")[:, ::7, ::7])
            H = t.shape[1]; c = t[:, H // 2 - 100:H // 2 + 100, t.shape[2] // 2 - 100:t.shape[2] // 2 + 100].reshape(3, -1)
            facts[name] = {"centre": rg_bg(np.median(c, axis=1)), "sky_wide": rg_bg(np.median(t[:, m7], axis=1))}
            say(f"{name:16s} centre {ratios(np.median(c, axis=1))}   sky-wide {ratios(np.median(t[:, m7], axis=1))}")

if have("padded_stack.npy"):
    import render as Rn
    sky_st = Rn.load_sky()[:, ::4, ::4]
    valid = sky_st[1] > 0
    d = sky_st - Rn.large_scale(sky_st, valid)
    n = np.array([1.4826 * np.median(np.abs(d[c][valid][::37])) for c in range(3)]) * 65535
    facts["stack_noise_adu"] = [round(float(x), 2) for x in n]
    say(f"stack noise      {facts['stack_noise_adu']} ADU per channel")

if have("count_map.npy") and have("mask_sky_d.npy"):
    c = np.load(f"{W}/count_map.npy")[mask]
    facts["frames_per_pixel"] = {"p5": int(np.percentile(c, 5)), "median": int(np.median(c)), "max": int(c.max())}
    say(f"frames per pixel over sky: p5 {np.percentile(c, 5):.0f}  median {np.median(c):.0f}  max {c.max()} of {P.N}")

if have("clouds.npz"):
    z = np.load(f"{W}/clouds.npz"); ms = z["masks"]
    sk = np.load(f"{W}/mask_sky.npy")[::8, ::8][: ms.shape[1], : ms.shape[2]]
    fr = np.array([mm[sk].mean() for mm in ms])
    hi = [int(f) for f, x in zip(z["frames"], fr) if x > 0.05]
    facts["clouds"] = {"mean_fraction": round(float(fr.mean()), 4), "frames_over_5pct": hi}
    say(f"clouds           mean {100 * fr.mean():.1f}% of sky; frames over 5%: {hi[:15]}{' ...' if len(hi) > 15 else ''}")

if have("log_fit.log"):
    txt = open(f"{W}/log_fit.log", errors="replace").read()
    tol = re.findall(r"^tol 3: .*$", txt, re.M)
    res = re.findall(r"r (\d+)-(\d+): n (\d+) median resid ([\d.]+) px", txt)
    facts["fit"] = {"summary": tol[-1][7:] if tol else None, "residual_px_by_radius": {b: float(m) for a, b, n, m in res}}
    say("model fit        " + (tol[-1][7:] if tol else "none") + "  |  " + "  ".join(f"r<{b}: {m}px" for a, b, n, m in res))

if have("tone.json"):
    t = json.load(open(f"{W}/tone.json"))
    fp = re.findall(r"footprint \((\d+), (\d+), (\d+), (\d+)\)", open(f"{W}/log_tone.log", errors="replace").read()) if have("log_tone.log") else []
    facts["tone"] = {"black_adu": [round(b * 65535, 1) for b in t["black"]], "white_adu": round(t["white"][0] * 65535), "a": t["a"],
                     "footprint": [int(v) for v in fp[-1]] if fp else None}
    say(f"tone             black {facts['tone']['black_adu']} ADU  white {facts['tone']['white_adu']} ADU  a {t['a']}"
        + (f"  footprint rows {fp[-1][0]}..{fp[-1][1]} cols {fp[-1][2]}..{fp[-1][3]}" if fp else ""))

stills = sorted(glob.glob(f"{W}/{P.NAME}_*.jpg"))
if stills:
    say("delivered stills, sky of the top third (neutral is 1.000 / 1.000):")
    facts["stills"] = {}
    for s in stills:
        v = sky_of(s); facts["stills"][os.path.basename(s)] = rg_bg(v) if max(v) >= 6 else None
        say(f"  {os.path.basename(s):40s} {delivered(v)}")
ff = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
videos = sorted(glob.glob(f"{W}/{P.NAME}_*_4K.mp4"))
if videos and os.path.exists(ff):
    say("delivered videos, one frame at 5 s:")
    facts["videos"] = {}
    for v in videos:
        frame = f"{W}/_probe_frame.png"
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "5", "-i", v, "-frames:v", "1", frame], capture_output=True)
        if os.path.exists(frame):
            c = sky_of(frame); facts["videos"][os.path.basename(v)] = {"sky": rg_bg(c) if max(c) >= 6 else None, "mb": round(os.path.getsize(v) / 1e6)}
            say(f"  {os.path.basename(v):40s} {delivered(c)}  {os.path.getsize(v) / 1e6:.0f} MB")
            os.remove(frame)
if JSON:
    print(json.dumps(facts, indent=1))
