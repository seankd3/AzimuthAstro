"""azastro: one entry point for the nightscape engine.

  azastro inspect <cr3_folder>                 report the session: settings drift, cadence, fog/clouds, test frames
  azastro new <workdir> <name> <cr3_folder> [--first N --last M --skip a,b] [--pole x,y] [--skyrows r] [--auto]
  azastro run <workdir> [--from STAGE] [--to STAGE] [--mood-frame N] [--force] [--detach]
  azastro process <workdir> <name> <cr3_folder> [--mood-frame N]     inspect + new --auto + run --detach: a whole night
  azastro status <workdir>
  azastro deliver <workdir>

Stage scripts stay standalone (each reads ASTRO_WORK); this file only composes them. session.py reads a night's folder.
"""
import os, sys, glob, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                                # the stages and session.py live beside this file
from session import inspect
STAGES = ["convert", "hot", "ground", "mask", "refine", "clouds", "reblank", "undist", "register", "fit", "warp", "stack",
          "pad", "count", "tone", "astap", "annotate", "traffic", "mood", "print", "trails", "export", "timelapse", "encode", "deliver"]
NONFATAL = {"annotate", "traffic", "mood", "print", "export", "timelapse", "encode", "deliver"}


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
            "encode": py("encode.py"), "deliver": py("deliver.py"),
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
