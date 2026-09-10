"""azastro: one entry point for the nightscape engine.

  azastro inspect <cr3_folder>                 report the session: settings drift, cadence, fog/clouds, test frames
  azastro new <workdir> <name> <cr3_folder> [--first N --last M --skip a,b] [--pole x,y] [--skyrows r] [--auto]
  azastro run <workdir> [--from STAGE] [--to STAGE]
  azastro status <workdir>
  azastro deliver <workdir>

Stage scripts stay standalone (each reads ASTRO_WORK); this file only composes them.
"""
import os, sys, glob, json, time, argparse, subprocess, datetime
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIRIL = r"C:\Program Files\Siril\bin\siril-cli.exe"
STAGES = ["convert", "ground", "mask", "refine", "clouds", "reblank", "undist", "register", "fit", "warp", "stack",
          "pad", "count", "tone", "astap", "annotate", "traffic", "mood", "print", "trails", "export", "timelapse", "encode", "deliver"]
NONFATAL = {"annotate", "traffic", "mood", "print", "export", "timelapse", "encode", "deliver"}


# ----------------------------------------------------------------------------- inspect
EXIF_TAGS = ["FileName", "DateTimeOriginal", "ExposureTime", "FNumber", "ISO", "FocusMode", "FocusDistanceUpper",
             "ShutterMode", "Quality", "WhiteBalance", "ColorTemperature", "Orientation", "LongExposureNoiseReduction",
             "HighlightTonePriority", "ImageStabilization", "CameraTemperature", "LensModel", "Model"]


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


def _peaks(path, orient):
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
    ys, xs = np.nonzero(pk)
    order = np.argsort(-sm[ys, xs])[:300]                              # the brightest are the reliable ones
    return np.c_[xs[order], ys[order]].astype(np.float32), g.shape


def pole_from_previews(prev, orient, cadence=30.0, pairs=12):
    """Stars move on circles around the celestial pole: a star's displacement between two previews
    `lag` frames apart is perpendicular to the line from the pole to the star, so every matched pair
    gives one linear equation for the pole. Ground lights do not move and drop out.
    Also the display row above which trails exist in nearly every column (certainly sky).
    Returns pole (x, y) and the sky row in full-resolution display pixels, plus the full size."""
    from scipy.spatial import cKDTree
    from scipy import ndimage as ndi
    from PIL import Image
    n = len(prev)
    lag = int(max(2, min(n // 4, round(300.0 / max(cadence, 1.0)))))    # ~5 min apart: 20-40 px of motion at preview scale
    idx = np.linspace(0, n - 1 - lag, pairs).astype(int)
    A, b = [], []
    shape = None
    for k in idx:
        p0, shape = _peaks(prev[k], orient); p1, _ = _peaks(prev[k + lag], orient)
        if len(p0) < 20 or len(p1) < 20:
            continue
        d, j = cKDTree(p1).query(p0, distance_upper_bound=60)
        ok = np.isfinite(d) & (d > 3.0)
        q, q1 = p0[ok], p1[j[ok]]
        disp = q1 - q                                                        # (q - pole) . disp = 0
        A.append(disp); b.append((disp * q).sum(axis=1))
    A = np.concatenate(A); b = np.concatenate(b)
    norms = np.maximum(np.linalg.norm(A, axis=1), 1e-6)
    best, best_n = None, -1                                                  # RANSAC: two lines fix a pole; count lines within 25 px
    rs = np.random.default_rng(0)
    for _ in range(400):
        i, j = rs.choice(len(A), 2, replace=False)
        M = A[[i, j]]
        if abs(np.linalg.det(M)) < 1e-3:
            continue
        cand = np.linalg.solve(M, b[[i, j]])
        n_in = int((np.abs(A @ cand - b) / norms < 25).sum())
        if n_in > best_n:
            best, best_n = cand, n_in
    inl = np.abs(A @ best - b) / norms < 25
    p = np.linalg.lstsq(A[inl], b[inl], rcond=None)[0]                       # refit on the consensus set
    print(f"  pole fit: {inl.sum()} of {len(A)} star displacements agree")
    H, W = shape
    scale = (5463 if ("90" in orient or "270" in orient) else 8191) / W
    acc = None
    for path in prev[::4]:
        im = Image.open(path).convert("L")
        if "90" in orient or "270" in orient:
            im = im.rotate(90 if "270" in orient else -90, expand=True)
        a = np.asarray(im, dtype=np.float32)
        acc = a if acc is None else np.maximum(acc, a)
    tr = np.clip(acc - ndi.median_filter(acc, 25), 0, None)
    cover = tr > (np.percentile(tr[tr > 0], 50) if (tr > 0).any() else 0)
    lowest = np.array([np.max(np.nonzero(cover[:, c])[0]) if cover[:, c].any() else 0 for c in range(W)])
    skyrows = int(np.min(lowest[lowest > 0]) * scale * 0.9) if (lowest > 0).any() else H // 2   # above the highest tree
    return (p[0] * scale, p[1] * scale), skyrows, int(round(W * scale)), int(round(H * scale))


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
    pole, skyrows, full_w, full_h = pole_from_previews(prev, rows[0]["Orientation"], cadence)
    good = np.ones(n, bool); good[[t - 1 for t in tests]] = False
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

    def siril(script, text):
        path = os.path.join(W, script)
        open(path, "w").write(text)
        return [SIRIL, "-s", path.replace("\\", "/")]

    PW = W.replace("\\", "/")
    if args.detach:                                                      # survive the terminal: relaunch ourselves detached
        cmd = [sys.executable, os.path.abspath(__file__), "run", W, "--from", args.frm, "--to", args.to, "--mood-frame", str(args.mood_frame)] + (["--force"] if args.force else [])
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with open(os.path.join(W, "run_detached.log"), "a") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        print(f"detached pid {proc.pid}; follow with: azastro status {args.workdir}")
        return
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
            "convert": py("convert.py"), "mask": py("mask.py"), "refine": py("mask_refine.py"), "clouds": py("clouds.py"),
            "reblank": py("reblank.py"), "undist": py("undist_frames.py"), "register": py("register.py", "all"),
            "fit": py("fitmodel.py"), "warp": py("warp2.py"), "pad": py("padstack.py"), "count": py("countmap.py"),
            "tone": py("tone_fit.py", "60", "0.18"), "astap": py("astap_center.py"), "annotate": py("solve.py"),
            "mood": py("mood.py", str(args.mood_frame), "1.0"), "print": py("print.py"),
            "trails": py("trails_gpu.py" if gpu else "trails.py"), "export": py("export_trails.py"),
            "timelapse": py("timelapse_gpu.py" if gpu else "timelapse.py", "locked", "standard", "clouds"),
            "encode": ["bash", os.path.join(HERE, "encode.sh")], "deliver": py("deliver.py"),
            "ground": siril("ground.ssf", f"requires 1.2.0\ncd {PW}\nsetext fit\nset32bits\nsetcpu 14\nstack full median -nonorm -out=ground\nstack full rej w 3 3 -nonorm -out=ground_mean\n"),
            "stack": siril("stack.ssf", f"requires 1.2.0\ncd {PW}\nsetext fit\nset32bits\nsetcpu 14\nstack r2_sky rej w 3 3 -nonorm -rejmap -out=sky\n"),
            "traffic": None,
        }[st]
        log.write(f"{time.strftime('%H:%M')} {st}\n"); log.flush()
        t0 = time.time()
        with open(os.path.join(W, f"log_{st}.log"), "w") as lf:
            if st == "ground":
                for f in ("full_.seq",):
                    try: os.remove(os.path.join(W, f))
                    except OSError: pass
            if st == "warp":
                for f in glob.glob(os.path.join(W, "r2_sky_*.fit")) + [os.path.join(W, "r2_sky_.seq"), os.path.join(W, "sky.fit")]:
                    try: os.remove(f)
                    except OSError: pass
            if st == "traffic":
                rc = subprocess.run(py("traffic.py"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                rc = rc or subprocess.run(py("traffic_still.py"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if st in ("ground", "stack"):
                ok = "Script execution finished successfully" in open(os.path.join(W, f"log_{st}.log")).read()
                rc = 0 if ok else 1
            if st == "ground" and rc == 0:
                import shutil; shutil.copyfile(os.path.join(W, "ground.fit"), os.path.join(W, "ground_fixed.fit"))
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
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a.folder)
    elif a.cmd == "new":
        new(a)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "status":
        status(a)
    elif a.cmd == "deliver":
        env = dict(os.environ, ASTRO_WORK=os.path.abspath(a.workdir).replace("\\", "/"))
        subprocess.run([sys.executable, os.path.join(HERE, "deliver.py")], env=env)


if __name__ == "__main__":
    main()
