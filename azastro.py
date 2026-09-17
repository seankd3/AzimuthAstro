"""azastro: one entry point for the nightscape engine.

  azastro inspect <cr3_folder>                 report the session: settings drift, cadence, fog/clouds, test frames
  azastro new <workdir> <name> <cr3_folder> [--first N --last M --skip a,b] [--pole x,y] [--skyrows r] [--auto]
  azastro run <workdir> [--from STAGE] [--to STAGE] [--mood-frame N] [--force] [--detach]
  azastro run <workdir> --stills | --videos     re-render only the stills (after a colour/tone change) or only the videos
  azastro process <workdir> <name> <cr3_folder> [--mood-frame N]     inspect + new --auto + run --detach: a whole night
  azastro status <workdir>                     current run: stages, gates, last progress line
  azastro wait <workdir> [--stage S] [--timeout SEC]   block until stage S (or the run) ends; prints only what is new
  azastro stop <workdir>                       kill the running chain
  azastro report <workdir>                     numbers that decide a night (gates, fit, clouds, footprint, tone) + report.jpg
  azastro probe <workdir>                      every number otherwise measured by hand: colour of sky/stars/trails, noise, coverage, each delivered file
  azastro deliver <workdir>

`run` resumes at the first stage without a done marker; `run --from S` redoes S and everything after.

Stage scripts stay standalone (each reads ASTRO_WORK); this file only composes them. session.py reads a night's folder.
"""
import os, re, sys, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                                # the stages and session.py live beside this file
from session import inspect
STAGES = ["convert", "hot", "ground", "mask", "refine", "clouds", "register", "fit", "warp", "stack",
          "tone", "astap", "annotate", "traffic", "mood", "print", "trails", "export", "timelapse", "encode", "deliver"]
NONFATAL = {"annotate", "traffic", "mood", "print", "export", "timelapse", "encode", "deliver"}
STILLS = ["tone", "astap", "annotate", "traffic", "mood", "print", "export", "deliver"]      # everything a colour or tone change touches
VIDEOS = ["trails", "timelapse", "encode", "deliver"]
TIMINGS = os.path.join(os.path.expanduser("~"), ".azastro", "timings.json")               # minutes per stage, every run ever: the ETA


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
    redo = args.frm is not None or args.force or args.stills or args.videos   # --from S: S and everything after is redone
    i0, i1 = STAGES.index(args.frm or "convert"), STAGES.index(args.to)
    stages = STILLS if args.stills else VIDEOS if args.videos else STAGES[i0:i1 + 1]

    def py(name, *a):
        return [sys.executable, os.path.join(HERE, name), *a]

    if args.detach:                                                      # survive the terminal: relaunch ourselves detached
        cmd = [sys.executable, os.path.abspath(__file__), "run", W, "--to", args.to, "--mood-frame", str(args.mood_frame)] + (["--from", args.frm] if args.frm else []) + (["--force"] if args.force else []) + (["--stills"] if args.stills else []) + (["--videos"] if args.videos else [])
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with open(os.path.join(W, "run_detached.log"), "a") as lf:
            proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        print(f"detached pid {proc.pid}; follow with: azastro status {args.workdir}")
        return
    lock = os.path.join(W, "chain.pid")                                 # one chain per workdir: two would write the same files
    if os.path.exists(lock) and _alive(int(open(lock).read() or 0)):
        raise SystemExit(f"a chain is already running in {W} (pid {open(lock).read()}); see chain.log")
    open(lock, "w").write(str(os.getpid()))
    import atexit; atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    done_dir = os.path.join(W, ".done"); os.makedirs(done_dir, exist_ok=True)
    log.write(f"== run {time.strftime('%Y-%m-%d %H:%M')} {'stills' if args.stills else 'videos' if args.videos else 'from ' + STAGES[i0] + ' to ' + args.to}; ETA {_eta(stages)}\n"); log.flush()
    for st in stages:
        marker = os.path.join(done_dir, st)
        if os.path.exists(marker) and not redo:
            print(f"{st:10s} done earlier, skipped")
            continue
        redo = True                                                      # once one stage runs, everything after it is stale
        for later in STAGES[STAGES.index(st):]:                            # everything downstream is stale now
            try: os.remove(os.path.join(done_dir, later))
            except OSError: pass
        cmd = {
            "convert": py("decode.py"), "hot": py("hot.py"), "mask": py("mask.py"), "refine": py("mask_refine.py"), "clouds": py("clouds.py"),
            "register": py("register.py", "all"), "fit": py("fitmodel.py"), "warp": py("warp.py"),
            "tone": py("tone_fit.py", "60", "0.18"), "astap": py("astap_center.py"), "annotate": py("solve.py"),
            "mood": py("mood.py", str(args.mood_frame), "1.0"), "print": py("print.py"),
            "trails": py("trails_gpu.py" if gpu else "trails.py"), "export": py("export_trails.py"),
            "timelapse": py("timelapse_gpu.py" if gpu else "timelapse.py", "locked", "standard", "clouds"),
            "encode": py("encode.py"), "deliver": py("deliver.py"),
            "ground": py("stack.py", "full_[0-9]*.fit", "ground.fit", "median"), "stack": py("stack.py", "warped/*.npy", "padded_stack.npy", "sigma"),
            "traffic": None,
        }[st]
        log.write(f"{time.strftime('%H:%M')} {st}\n"); log.flush()
        t0 = time.time()
        with open(os.path.join(W, f"log_{st}.log"), "w") as lf:
            if st == "traffic":
                rc = subprocess.run(py("traffic.py"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                rc = rc or subprocess.run(py("traffic_still.py"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if st == "ground" and rc == 0:
                rc = subprocess.run(py("stack.py", "full_[0-9]*.fit", "ground_mean.fit", "sigma"), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
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
            _record(st, mins, cfg.get("frame_ids", []))
    log.write("DONE\n"); log.close()
    subprocess.run(py("report.py"), env=env)
    print("done:", cfg["name"])


def _record(stage, mins, frame_ids):
    """minutes per stage per 100 frames, kept across every project: what the next ETA is made of."""
    if os.environ.get("SMOKE"):                                              # a 12-frame synthetic night is not a night
        return
    os.makedirs(os.path.dirname(TIMINGS), exist_ok=True)
    t = json.load(open(TIMINGS)) if os.path.exists(TIMINGS) else {}
    t.setdefault(stage, []).append(round(mins * 100.0 / max(len(frame_ids), 1), 3))
    t[stage] = t[stage][-20:]
    json.dump(t, open(TIMINGS, "w"), indent=1)


def _eta(stages, frames=None):
    """'~HH:MM (N min)' for the stages still to run, from recorded stage times; '?' with no history."""
    t = json.load(open(TIMINGS)) if os.path.exists(TIMINGS) else {}
    if frames is None:
        try:
            frames = len(json.load(open(os.path.join(os.getcwd(), "project.json")))["frame_ids"])
        except Exception:
            frames = 100
    known = [sorted(t[s])[len(t[s]) // 2] * frames / 100.0 for s in stages if s in t]
    if not known:
        return "?"
    mins = sum(known)
    return f"~{time.strftime('%H:%M', time.localtime(time.time() + mins * 60))} ({mins:.0f} min{'' if len(known) == len(stages) else ', some stages unmeasured'})"


def _remaining(W, lines):
    """stages the current run has not finished yet, from its header line and the stage lines so far."""
    head = next((l for l in lines if l.startswith("== run")), "")
    if " stills;" in head:
        plan = STILLS
    elif " videos;" in head:
        plan = VIDEOS
    else:
        m = re.search(r"from (\w+) to (\w+)", head)
        plan = STAGES[STAGES.index(m.group(1)):STAGES.index(m.group(2)) + 1] if m else STAGES
    done = {l.split()[-1] for l in lines if re.match(r"^\d\d:\d\d [a-z]+$", l)}
    started = [l.split()[-1] for l in lines if re.match(r"^\d\d:\d\d [a-z]+$", l)]
    current = started[-1] if started else None
    return [s for s in plan if s not in done or s == current]


def _current(W):
    """(lines of the current run, running?) from chain.log: everything after the last run header."""
    path = os.path.join(W, "chain.log")
    lines = open(path, errors="replace").read().strip().split("\n") if os.path.exists(path) else []
    starts = [i for i, l in enumerate(lines) if l.startswith("== run")]
    lines = lines[starts[-1]:] if starts else lines
    lock = os.path.join(W, "chain.pid")
    running = os.path.exists(lock) and _alive(int(open(lock).read() or 0))
    return lines, running


def _progress(W, lines):
    """the last progress line of the stage now running, if any."""
    stage = [l for l in lines if re.match(r"^\d\d:\d\d [a-z]+$", l)]
    if not stage:
        return ""
    st = stage[-1].split()[-1]
    lp = os.path.join(W, f"log_{st}.log")
    if not os.path.exists(lp):
        return ""
    tail = open(lp, errors="replace").read()[-300:].replace("\r", "\n").strip().split("\n")
    return f"  [{st}] {tail[-1][-120:]}" if tail and tail[-1] else ""


def status(args):
    W = os.path.abspath(args.workdir)
    lines, running = _current(W)
    frames = len(json.load(open(os.path.join(W, "project.json")))["frame_ids"]) if os.path.exists(os.path.join(W, "project.json")) else 100
    remaining = _remaining(W, lines) if running else []
    if getattr(args, "json", False):                                        # the shape a wrapper reads
        gates = {l.split()[2].rstrip(":"): l.split(" ", 3)[3] if len(l.split(" ", 3)) > 3 else "" for l in lines if l.startswith("  gate ")}
        state = "running" if running else "done" if any(l.startswith("DONE") for l in lines) else "failed" if any(l.startswith("FAIL") for l in lines) else "stopped"
        print(json.dumps({"state": state, "stage": remaining[0] if remaining else None, "remaining": remaining,
                          "eta": _eta(remaining, frames) if remaining else None, "gates": gates, "progress": _progress(W, lines).strip()}, indent=1))
        return
    print("\n".join(lines) or "not started")
    if running:
        print(_progress(W, lines)); print(f"  remaining: {' '.join(remaining)}; ETA {_eta(remaining, frames)}")
    else:
        print("(not running)")


def wait(args):
    """Block until the run passes `stage` (its gate line or the next stage starts), ends, or fails; print only
    the lines that appeared meanwhile. Exit 0 done, 1 failed, 3 still running at the timeout."""
    W = os.path.abspath(args.workdir)
    seen = len(_current(W)[0])
    t0 = time.time()
    after = STAGES[STAGES.index(args.stage) + 1] if args.stage and STAGES.index(args.stage) + 1 < len(STAGES) else None
    while True:
        lines, running = _current(W)
        new = lines[seen:]
        done = any(l.startswith("DONE") for l in new)
        failed = any(re.match(r"^FAIL [a-z]+ after [\d.]+ min$", l) for l in new)
        passed = args.stage and any(l.startswith("  gate ") and f" {args.stage}:" in l or (after and re.match(rf"^\d\d:\d\d {after}$", l)) for l in new)
        if done or failed or passed or (not running and time.time() - t0 > 30) or time.time() - t0 > args.timeout:   # a detached run takes a few seconds to write chain.pid
            print("\n".join(new) or "(nothing new)")
            if running and not (done or failed or passed):
                frames = len(json.load(open(os.path.join(W, "project.json")))["frame_ids"])
                print(_progress(W, lines)); print(f"  ETA {_eta(_remaining(W, lines), frames)}"); sys.exit(3)
            sys.exit(1 if failed or (not running and not done and not passed) else 0)
        time.sleep(15)


def stop(args):
    W = os.path.abspath(args.workdir)
    lock = os.path.join(W, "chain.pid")
    pid = int(open(lock).read() or 0) if os.path.exists(lock) else 0
    if not pid or not _alive(pid):
        print("no chain running"); return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, 15)
    try: os.remove(lock)
    except OSError: pass
    with open(os.path.join(W, "chain.log"), "a") as f:
        f.write(f"STOPPED {time.strftime('%H:%M')}\n")
    print(f"stopped pid {pid}")


def main():
    ap = argparse.ArgumentParser(prog="azastro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect"); p.add_argument("folder")
    p = sub.add_parser("new"); p.add_argument("workdir"); p.add_argument("name"); p.add_argument("folder")
    p.add_argument("--first", type=int); p.add_argument("--last", type=int); p.add_argument("--skip", default="")
    p.add_argument("--pole"); p.add_argument("--skyrows", type=int); p.add_argument("--auto", action="store_true")
    p = sub.add_parser("run"); p.add_argument("workdir"); p.add_argument("--from", dest="frm", default=None, choices=STAGES, help="redo from this stage on")
    p.add_argument("--to", default="deliver", choices=STAGES); p.add_argument("--mood-frame", type=int, default=1)
    p.add_argument("--force", action="store_true", help="redo stages that already have a done marker")
    p.add_argument("--detach", action="store_true", help="run in a detached process and return")
    p.add_argument("--stills", action="store_true", help="only the still outputs (tone .. print, export, deliver)")
    p.add_argument("--videos", action="store_true", help="only the videos (trails, timelapse, encode, deliver)")
    p = sub.add_parser("status"); p.add_argument("workdir"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("wait"); p.add_argument("workdir"); p.add_argument("--stage", choices=STAGES); p.add_argument("--timeout", type=float, default=540)
    p = sub.add_parser("stop"); p.add_argument("workdir")
    p = sub.add_parser("report"); p.add_argument("workdir")
    p = sub.add_parser("probe"); p.add_argument("workdir"); p.add_argument("--json", action="store_true")
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
        run(argparse.Namespace(workdir=a.workdir, frm=None, to="deliver", mood_frame=a.mood_frame, force=False, detach=True, stills=False, videos=False))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "status":
        status(a)
    elif a.cmd == "wait":
        wait(a)
    elif a.cmd == "stop":
        stop(a)
    elif a.cmd in ("deliver", "report", "probe"):
        env = dict(os.environ, ASTRO_WORK=os.path.abspath(a.workdir).replace("\\", "/"))
        subprocess.run([sys.executable, os.path.join(HERE, f"{a.cmd}.py")] + (["--json"] if getattr(a, "json", False) else []), env=env)


if __name__ == "__main__":
    main()
