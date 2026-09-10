"""azastro: one entry point for the nightscape engine.

  azastro inspect <cr3_folder>                 report the session: settings drift, cadence, fog/clouds, test frames
  azastro new <workdir> <name> <cr3_folder> [--first N --last M --skip a,b] [--pole x,y] [--skyrows r] [--auto]
  azastro run <workdir> [--from STAGE] [--to STAGE] [--mood-frame N] [--force] [--detach]
  azastro process <workdir> <name> <cr3_folder> [--mood-frame N]     inspect + new --auto + run --detach: a whole night
  azastro status <workdir>
  azastro deliver <workdir>

Stage scripts stay standalone (each reads ASTRO_WORK); this file only composes them.
"""
import os, sys, glob, json, time, argparse, subprocess, datetime
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["convert", "hot", "ground", "mask", "refine", "clouds", "reblank", "undist", "register", "fit", "warp", "stack",
          "pad", "count", "tone", "astap", "annotate", "traffic", "mood", "print", "trails", "export", "timelapse", "encode", "deliver"]
NONFATAL = {"annotate", "traffic", "mood", "print", "export", "timelapse", "encode", "deliver"}


# ----------------------------------------------------------------------------- inspect
EXIF_TAGS = ["FileName", "DateTimeOriginal", "ExposureTime", "FNumber", "ISO", "FocusMode", "FocusDistanceUpper",
             "ShutterMode", "Quality", "WhiteBalance", "ColorTemperature", "Orientation", "LongExposureNoiseReduction",
             "HighlightTonePriority", "ImageStabilization", "CameraTemperature", "LensModel", "Model", "ImageWidth", "ImageHeight", "FocalLength", "ScaleFactor35efl"]


def exif_table(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.CR3")))
    fmt = "\t".join(f"${t}" for t in EXIF_TAGS)
    out = subprocess.run(["exiftool", "-q", "-p", fmt, "-ext", "CR3", "-fileOrder", "FileName", folder], capture_output=True, text=True).stdout
    rows = [dict(zip(EXIF_TAGS, line.split("\t"))) for line in out.strip().split("\n") if line.strip()]
    return files, rows


def _preview_stats(args):
    """star count and sky level from one embedded JPEG preview (fast, no raw decode)."""
    path, orient = args
    from PIL import Image
    from scipy import ndimage as ndi
    im = Image.open(path).convert("L")
    if "90" in orient or "270" in orient:
        im = im.rotate(90 if "270" in orient else -90, expand=True)
    g = np.asarray(im, dtype=np.float32)
    hp = g - ndi.median_filter(g, 9)
    sig = 1.4826 * np.median(np.abs(hp))
    pk = (hp == ndi.maximum_filter(hp, 5)) & (hp > 5 * max(sig, 1.0))
    sky = g[: g.shape[0] // 2]
    return int(pk.sum()), float(np.median(sky))


def _peaks(path, orient, row_max=None, n=150):
    """the n brightest star-like peaks above display row `row_max` (sky only) of one preview, sub-pixel."""
    from PIL import Image
    from scipy import ndimage as ndi
    im = Image.open(path).convert("L")
    if "90" in orient or "270" in orient:
        im = im.rotate(90 if "270" in orient else -90, expand=True)
    g = np.asarray(im, dtype=np.float32)
    hp = g - ndi.median_filter(g, 9)
    sig = 1.4826 * np.median(np.abs(hp))
    sm = ndi.gaussian_filter(hp, 1.0)
    pk = (sm == ndi.maximum_filter(sm, 5)) & (sm > 6 * max(sig, 1.0))
    if row_max is not None:
        pk[row_max:] = False
    ys, xs = np.nonzero(pk)
    order = np.argsort(-sm[ys, xs])[:n]                                # the brightest are the reliable ones
    ys, xs = ys[order], xs[order]
    keep = (ys > 0) & (ys < g.shape[0] - 1) & (xs > 0) & (xs < g.shape[1] - 1)
    ys, xs = ys[keep], xs[keep]
    dy, dx = np.mgrid[-1:2, -1:2]
    w = np.clip(sm[ys[:, None, None] + dy, xs[:, None, None] + dx], 0, None)   # 3x3 weighted centroid
    tot = w.sum(axis=(1, 2))
    return np.c_[xs + (w * dx).sum(axis=(1, 2)) / tot, ys + (w * dy).sum(axis=(1, 2)) / tot].astype(np.float32), g.shape


def pole_from_previews(prev, times, orient, full, focal_px, pairs=12):
    """Where the sky turns. Between two previews the sky has turned by a known angle (the clock) about the
    celestial pole, and on the sensor that is the pinhole homography K R K^-1 (optics.sky_homography), so
    the only unknown is the pole's image point. Matched star peaks score candidate poles over the whole
    sphere; the best is refined by least squares, then re-matched over a longer baseline.
    `prev` holds only usable frames (in focus, not tests), `times` their timestamps, `full` the sensor (w, h),
    `focal_px` the focal length in full-resolution pixels. Also the display row above which trails exist in
    nearly every column (certainly sky). Returns pole (x, y) and sky row in full-resolution display pixels,
    plus the display size."""
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    from scipy import ndimage as ndi
    from PIL import Image
    import optics
    n = len(prev)
    step = float(np.median(np.diff(times))) if n > 1 else 30.0
    acc = None
    for path in prev[::4]:
        im = Image.open(path).convert("L")
        if "90" in orient or "270" in orient:
            im = im.rotate(90 if "270" in orient else -90, expand=True)
        a = np.asarray(im, dtype=np.float32)
        acc = a if acc is None else np.maximum(acc, a)
    H, W = acc.shape
    tr = np.clip(acc - ndi.median_filter(acc, 25), 0, None)
    cover = tr > (np.percentile(tr[tr > 0], 50) if (tr > 0).any() else 0)
    lowest = np.array([np.max(np.nonzero(cover[:, c])[0]) if cover[:, c].any() else 0 for c in range(W)])
    sky_row_prev = int(np.min(lowest[lowest > 0]) * 0.9) if (lowest > 0).any() else H // 2
    scale = (full[1] if ("90" in orient or "270" in orient) else full[0]) / W
    f = focal_px / scale
    cx, cy = (W - 1) / 2, (H - 1) / 2
    om = 2 * np.pi / 86164.0905

    def rays(pts):
        return np.c_[(pts[:, 0] - cx) / f, (pts[:, 1] - cy) / f, np.ones(len(pts))]

    def pairs_at(lag, pole=None, tol=30.0):
        """matched (q, q1, theta) over `pairs` preview pairs `lag` apart: mutual nearest neighbours, or
        nearest to the position a known pole predicts."""
        Q, Q1, TH = [], [], []
        for k in np.linspace(0, n - 1 - lag, pairs).astype(int):
            p0, _ = _peaks(prev[k], orient, sky_row_prev); p1, _ = _peaks(prev[k + lag], orient, sky_row_prev)   # sky only: lake reflections turn about a mirrored pole
            if len(p0) < 20 or len(p1) < 20:
                continue
            th = om * (times[k + lag] - times[k])                            # the rotation angle is known from the clock
            if pole is None:
                d, j = cKDTree(p1).query(p0, distance_upper_bound=tol)
                d2, j2 = cKDTree(p0).query(p1, distance_upper_bound=tol)
                ok = np.isfinite(d) & (d > 2.0) & (j2[np.minimum(j, len(p1) - 1)] == np.arange(len(p0)))
            else:
                pred = optics.apply(optics.sky_homography(pole[2] * th, f, pole[:2], cx, cy), p0)
                d, j = cKDTree(p1).query(pred, distance_upper_bound=tol)
                ok = np.isfinite(d)
            Q.append(p0[ok]); Q1.append(p1[j[ok]]); TH.append(np.full(int(ok.sum()), th))
        return np.concatenate(Q), np.concatenate(Q1), np.concatenate(TH)

    def predict(axes, Q, TH):
        """(M, N, 2) positions of rays Q turned by TH about each unit axis (Rodrigues, vectorised)."""
        v = rays(Q)
        c, s_ = np.cos(TH)[None, :, None], np.sin(TH)[None, :, None]
        nxv = np.cross(axes[:, None, :], v[None, :, :])
        ndv = (axes[:, None, :] * v[None, :, :]).sum(-1)[..., None]
        r = v[None] * c + nxv * s_ + axes[:, None, :] * ndv * (1 - c)
        return np.stack([r[..., 0] / r[..., 2] * f + cx, r[..., 1] / r[..., 2] * f + cy], -1)

    Q, Q1, TH = pairs_at(int(max(1, min(n // 4, round(150.0 / max(step, 1.0))))))   # ~2.5 min: ~10 px of motion, below the star spacing
    i = np.arange(4000) + 0.5                                                # every axis on the sphere (Fibonacci); theta > 0 covers both senses
    z = 1 - 2 * i / 4000; r = np.sqrt(1 - z * z); phi = np.pi * (1 + 5 ** 0.5) * i
    axes = np.c_[r * np.cos(phi), r * np.sin(phi), z]
    score = np.zeros(len(axes), int)
    for c0 in range(0, len(axes), 250):
        score[c0:c0 + 250] = (np.linalg.norm(predict(axes[c0:c0 + 250], Q, TH) - Q1[None], axis=2) < 3.0).sum(axis=1)
    ax = axes[score.argmax()]
    sign = 1.0 if ax[2] > 0 else -1.0                                        # the pole's image point is where the axis meets the sensor plane
    pole = np.array([ax[0] / ax[2] * f + cx, ax[1] / ax[2] * f + cy, sign])

    def residual(x, Q, Q1, TH):
        a = np.array([(x[0] - cx) / f, (x[1] - cy) / f, 1.0]); a /= np.linalg.norm(a)
        return (predict(a[None], Q, TH)[0] - Q1).ravel()

    for lag_s, tol in ((150.0, 3.0), (900.0, 4.0)):                          # refine, then again over ~15 min of motion matched by prediction
        lag = int(max(1, min(n - 2, round(lag_s / max(step, 1.0)))))
        Q, Q1, TH = pairs_at(lag, pole, tol)
        fit = least_squares(residual, pole[:2], args=(Q, Q1, TH * pole[2]), loss="soft_l1", f_scale=1.0)
        pole[:2] = fit.x
        res = np.linalg.norm(residual(pole, Q, Q1, TH * pole[2]).reshape(-1, 2), axis=1)
    print(f"  pole fit: {(res < 2).sum()} of {len(res)} star displacements agree within 2 px ({'counter-' if sign > 0 else ''}clockwise on screen)")
    return (pole[0] * scale, pole[1] * scale), int(sky_row_prev * scale), int(round(W * scale)), int(round(H * scale))


def inspect(folder, write_json=True):
    files, rows = exif_table(folder)
    n = len(rows)
    print(f"{n} frames in {folder}")
    if not n:
        return None
    ts = [datetime.datetime.strptime(r["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S").timestamp() for r in rows]
    gaps = np.diff(ts)
    cadence = float(np.median(gaps)) if len(gaps) else 0.0
    exp = [float(eval(r["ExposureTime"])) if "/" in r["ExposureTime"] else float(r["ExposureTime"]) for r in rows]
    warnings = []
    # settings drift and traps
    def distinct(key):
        return sorted(set(r[key] for r in rows))
    for key in ("ExposureTime", "FNumber", "ISO", "WhiteBalance", "ColorTemperature", "ShutterMode", "Quality"):
        d = distinct(key)
        if len(d) > 1:
            warnings.append(f"{key} varies: {d[:6]}")
    fm = distinct("FocusMode")
    if any("Servo" in f or "One-shot" in f or "AI" in f for f in fm):
        warnings.append(f"autofocus was on ({fm}); every frame may refocus. Use MF.")
    fd = distinct("FocusDistanceUpper")
    skip = []
    if len(fd) > 1:
        far = max(fd, key=lambda v: 1e9 if v == "inf" else float(v.split()[0]))     # the infinity setting
        skip = [i + 1 for i, r in enumerate(rows) if r["FocusDistanceUpper"] != far]
        warnings.append(f"focus distance changes between frames: {fd[:6]}; {len(skip)} frames are not at {far} and go in --skip")
    if "CRAW" in distinct("Quality"):
        warnings.append("CRAW: lossy raw, shadow artefacts when stretched. Use RAW.")
    if any(int(r["ISO"]) < 640 for r in rows):
        warnings.append("ISO below 640: the R5 bakes noise reduction into the raw there.")
    if "Mechanical" in distinct("ShutterMode"):
        warnings.append("mechanical shutter: prefer electronic first-curtain (14-bit, no shutter shock).")
    if any("Electronic" == s for s in distinct("ShutterMode")):
        warnings.append("full electronic shutter: raw drops to 12-bit on the R5.")
    if distinct("LongExposureNoiseReduction") != ["Off"]:
        warnings.append("long-exposure NR on: halves the frame rate, dashed trails.")
    if distinct("HighlightTonePriority") != ["Off"]:
        warnings.append("highlight tone priority on: one stop of raw exposure lost.")
    duty = float(np.median(exp)) / cadence if cadence else 0
    if duty < 0.8:
        warnings.append(f"duty cycle {duty:.0%} (exposure {np.median(exp):.0f}s every {cadence:.0f}s): trails will be dashed; set the interval to exposure + 1-2 s.")
    # timing: test frames are the ones before the cadence settles
    irregular = [i for i in range(1, n) if abs(gaps[i - 1] - cadence) > max(3, 0.15 * cadence)]
    tests = []
    k = 0
    while k < n - 1 and abs(gaps[k] - cadence) > max(3, 0.15 * cadence):
        tests.append(k + 1); k += 1
    # previews: star counts and sky level
    tmp = os.path.join(folder, "_previews")
    os.makedirs(tmp, exist_ok=True)
    if len(glob.glob(os.path.join(tmp, "*.jpg"))) < n:
        subprocess.run(["exiftool", "-q", "-b", "-PreviewImage", "-ext", "CR3", "-w", os.path.join(tmp, "%f.jpg"), folder], capture_output=True)
    prev = [os.path.join(tmp, os.path.splitext(os.path.basename(f))[0] + ".jpg") for f in files]
    with Pool(6) as p:
        stats = p.map(_preview_stats, [(pv, rows[0]["Orientation"]) for pv in prev])
    stars = np.array([s[0] for s in stats]); sky = np.array([s[1] for s in stats])
    good = np.ones(n, bool); good[[t - 1 for t in tests]] = False
    usable = good.copy(); usable[[i - 1 for i in skip]] = False              # misfocused previews have no matchable stars
    import optics
    full = (int(rows[0]["ImageWidth"]), int(rows[0]["ImageHeight"]))
    fpx = optics.focal_px(float(rows[0]["FocalLength"].split()[0]), max(full), float(rows[0]["ScaleFactor35efl"] or 1.0))
    pole, skyrows, full_w, full_h = pole_from_previews([p for p, u in zip(prev, usable) if u], np.array(ts)[usable], rows[0]["Orientation"], full, fpx)
    clear = np.median(stars[good][: max(10, good.sum() // 3)])
    thr = 0.6 * clear
    healthy = np.nonzero((stars > thr) & good)[0]
    last = int(healthy.max()) + 1 if len(healthy) else n
    first = (tests[-1] + 1) if tests else 1
    dips = [i + 1 for i in range(first - 1, last) if stars[i] < thr]
    print(f"cadence {cadence:.0f}s, exposure {np.median(exp):.0f}s, duty {duty:.0%}, span {(ts[-1]-ts[0])/60:.0f} min, orientation {rows[0]['Orientation']}")
    print(f"settings: {rows[0]['Model']}, {rows[0]['LensModel']}, f/{rows[0]['FNumber']}, ISO {rows[0]['ISO']}, {rows[0]['ShutterMode']}, {rows[0]['Quality']}, WB {rows[0]['WhiteBalance']} {rows[0]['ColorTemperature']}K, sensor {rows[0]['CameraTemperature']}")
    print(f"stars/preview: clear median {clear:.0f}; healthy frames {first}..{last} of {n}; test frames {tests}; cloud dips inside: {dips[:12]}{' ...' if len(dips) > 12 else ''}")
    if last < n:
        print(f"fog/haze from frame {last + 1} on ({n - last} frames dropped)")
    for w in warnings:
        print("WARNING:", w)
    print(f"pole (display px) {pole[0]:.0f},{pole[1]:.0f}; sky certain above display row {skyrows} of {full_h}")
    skip_note = f"  (--auto also skips the {len(skip)} misfocused frames listed in _inspect.json)" if skip else ""
    print("\nsuggested:  azastro new <workdir> <name> \"%s\" --auto   [= --first %d --last %d --pole %.0f,%.0f --skyrows %d]%s"
          % (folder, first, last, pole[0], pole[1], skyrows, skip_note))
    report = {"folder": folder, "n": n, "cadence": cadence, "exposure": float(np.median(exp)), "duty": duty, "tests": tests,
              "first": first, "last": last, "skip": skip, "pole": [float(pole[0]), float(pole[1])], "skyrows": int(skyrows),
              "dips": dips, "stars": stars.tolist(), "sky": sky.tolist(), "warnings": warnings, "orientation": rows[0]["Orientation"]}
    if write_json:
        json.dump(report, open(os.path.join(folder, "_inspect.json"), "w"), indent=1)
    return report


# ----------------------------------------------------------------------------- new / run / status / deliver
def new(args):
    rep = None
    if args.auto or args.first is None:
        rep = inspect(args.folder, write_json=True)
        first, last = rep["first"], rep["last"]
        if not args.pole:
            args.pole = "%.0f,%.0f" % tuple(rep["pole"])
        if not args.skyrows:
            args.skyrows = rep["skyrows"]
        if rep["skip"] and not args.skip:
            args.skip = ",".join(map(str, rep["skip"]))
    else:
        first, last = args.first, args.last
    cmd = [sys.executable, os.path.join(HERE, "newproject.py"), args.workdir, args.name, args.folder, str(first), str(last)]
    if args.skip:
        cmd += ["--skip", args.skip]
    if args.pole:
        cmd += ["--pole", args.pole]
    if args.skyrows:
        cmd += ["--skyrows", str(args.skyrows)]
    subprocess.run(cmd, check=True)
    print("next:  azastro run", args.workdir)


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong(); ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                        # STILL_ACTIVE
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def run(args):
    W = os.path.abspath(args.workdir)
    env = dict(os.environ, ASTRO_WORK=W.replace("\\", "/"))
    if not os.path.exists(os.path.join(W, "project.json")):
        raise SystemExit("no project.json; run `azastro new` first")
    cfg = json.load(open(os.path.join(W, "project.json")))
    gpu = has_cuda()
    log = open(os.path.join(W, "chain.log"), "a")
    i0, i1 = STAGES.index(args.frm), STAGES.index(args.to)

    def py(name, *a):
        return [sys.executable, os.path.join(HERE, name), *a]

    if args.detach:                                                      # survive the terminal: relaunch ourselves detached
        cmd = [sys.executable, os.path.abspath(__file__), "run", W, "--from", args.frm, "--to", args.to, "--mood-frame", str(args.mood_frame)] + (["--force"] if args.force else [])
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with open(os.path.join(W, "run_detached.log"), "a") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        print(f"detached pid {proc.pid}; follow with: azastro status {args.workdir}")
        return
    lock = os.path.join(W, "chain.pid")                                 # one chain per workdir: two would write the same files
    if os.path.exists(lock) and _alive(int(open(lock).read() or 0)):
        raise SystemExit(f"a chain is already running in {W} (pid {open(lock).read()}); see chain.log")
    open(lock, "w").write(str(os.getpid()))
    import atexit; atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    done_dir = os.path.join(W, ".done"); os.makedirs(done_dir, exist_ok=True)
    for st in STAGES[i0:i1 + 1]:
        marker = os.path.join(done_dir, st)
        if os.path.exists(marker) and not args.force:
            print(f"{st:10s} done earlier, skipped (use --force to redo)")
            continue
        for later in STAGES[STAGES.index(st):]:                            # everything downstream is stale now
            try: os.remove(os.path.join(done_dir, later))
            except OSError: pass
        cmd = {
            "convert": py("decode.py"), "hot": py("hot.py"), "mask": py("mask.py"), "refine": py("mask_refine.py"), "clouds": py("clouds.py"),
            "reblank": py("reblank.py"), "undist": py("undist_frames.py"), "register": py("register.py", "all"),
            "fit": py("fitmodel.py"), "warp": py("warp2.py"), "pad": py("padstack.py"), "count": py("countmap.py"),
            "tone": py("tone_fit.py", "60", "0.18"), "astap": py("astap_center.py"), "annotate": py("solve.py"),
            "mood": py("mood.py", str(args.mood_frame), "1.0"), "print": py("print.py"),
            "trails": py("trails_gpu.py" if gpu else "trails.py"), "export": py("export_trails.py"),
            "timelapse": py("timelapse_gpu.py" if gpu else "timelapse.py", "locked", "standard", "clouds"),
            "encode": ["bash", os.path.join(HERE, "encode.sh")], "deliver": py("deliver.py"),
            "ground": py("stack.py", "full", "ground", "median"), "stack": py("stack.py", "r2_sky", "sky", "sigma"),
            "traffic": None,
        }[st]
        log.write(f"{time.strftime('%H:%M')} {st}\n"); log.flush()
        t0 = time.time()
        with open(os.path.join(W, f"log_{st}.log"), "w") as lf:
            if st == "warp":
                for f in glob.glob(os.path.join(W, "r2_sky_*.fit")) + [os.path.join(W, "sky.fit")]:
                    try: os.remove(f)
                    except OSError: pass
            if st == "traffic":
                rc = subprocess.run(py("traffic.py"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                rc = rc or subprocess.run(py("traffic_still.py"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if st == "ground" and rc == 0:
                rc = subprocess.run(py("stack.py", "full", "ground_mean", "sigma"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        if rc == 0:
            gate = subprocess.run([sys.executable, os.path.join(HERE, "gates.py"), st], env=env, capture_output=True, text=True).stdout.strip()
            if gate:
                log.write(f"  gate {gate}\n"); log.flush(); print("  " + gate)
                if gate.startswith("FAIL"):
                    rc = 2
        if rc != 0:
            log.write(f"FAIL {st} after {mins:.1f} min{' (non-fatal)' if st in NONFATAL else ''}\n"); log.flush()
            print(f"FAIL {st} (see log_{st}.log)")
            if st not in NONFATAL:
                sys.exit(1)
        else:
            print(f"{st:10s} {mins:5.1f} min")
            open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M"))
    log.write("DONE\n"); log.close()
    print("done:", cfg["name"])


def status(args):
    W = os.path.abspath(args.workdir)
    lines = open(os.path.join(W, "chain.log")).read().strip().split("\n") if os.path.exists(os.path.join(W, "chain.log")) else []
    print("\n".join(lines[-6:]) or "not started")
    if lines and not lines[-1].startswith(("DONE", "FAIL")):
        st = lines[-1].split()[-1]
        lp = os.path.join(W, f"log_{st}.log")
        if os.path.exists(lp):
            tail = open(lp, errors="replace").read()[-300:].replace("\r", "\n").strip().split("\n")[-2:]
            print(f"  [{st}] " + " | ".join(tail))


def main():
    ap = argparse.ArgumentParser(prog="azastro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect"); p.add_argument("folder")
    p = sub.add_parser("new"); p.add_argument("workdir"); p.add_argument("name"); p.add_argument("folder")
    p.add_argument("--first", type=int); p.add_argument("--last", type=int); p.add_argument("--skip", default="")
    p.add_argument("--pole"); p.add_argument("--skyrows", type=int); p.add_argument("--auto", action="store_true")
    p = sub.add_parser("run"); p.add_argument("workdir"); p.add_argument("--from", dest="frm", default="convert", choices=STAGES)
    p.add_argument("--to", default="deliver", choices=STAGES); p.add_argument("--mood-frame", type=int, default=1)
    p.add_argument("--force", action="store_true", help="redo stages that already have a done marker")
    p.add_argument("--detach", action="store_true", help="run in a detached process and return")
    p = sub.add_parser("status"); p.add_argument("workdir")
    p = sub.add_parser("deliver"); p.add_argument("workdir")
    p = sub.add_parser("process", help="the whole night in one go: inspect, new --auto, run --detach")
    p.add_argument("workdir"); p.add_argument("name"); p.add_argument("folder"); p.add_argument("--mood-frame", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a.folder)
    elif a.cmd == "new":
        new(a)
    elif a.cmd == "process":
        new(argparse.Namespace(workdir=a.workdir, name=a.name, folder=a.folder, first=None, last=None, skip="", pole=None, skyrows=None, auto=True))
        run(argparse.Namespace(workdir=a.workdir, frm="convert", to="deliver", mood_frame=a.mood_frame, force=False, detach=True))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "status":
        status(a)
    elif a.cmd == "deliver":
        env = dict(os.environ, ASTRO_WORK=os.path.abspath(a.workdir).replace("\\", "/"))
        subprocess.run([sys.executable, os.path.join(HERE, "deliver.py")], env=env)


if __name__ == "__main__":
    main()
