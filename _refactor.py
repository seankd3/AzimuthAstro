import pathlib
E = pathlib.Path(r"D:\AstroWork\engine")
NH = r'W = r"D:\AstroWork\nhlake"'


def edit(name, pairs):
    p = E / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, f"{name}: missing {a!r}"
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


edit("register.py", [
    (NH + "\nWIDTH, HEIGHT = 8191, 5463", "import project as P\nW = P.W\nWIDTH, HEIGHT = P.WIDTH, P.HEIGHT"),
    ("REF = 45", "REF = P.REF"),
    ('''def frame_times():
    """seconds since first frame, indexed by u_sky index (1..90) from exif.tsv (odd CR3s)."""
    import datetime
    t = {}
    rows = [l.split("\\t") for l in open(r"C:\\Users\\smast\\AppData\\Local\\Temp\\exif.tsv")]
    for k, r in enumerate(rows):
        n = k + 1
        if n % 2 == 1:
            d = datetime.datetime.strptime(r[1], "%Y:%m:%d %H:%M:%S")
            t[(n + 1) // 2] = d.timestamp()
    t0 = t[REF]
    return {i: v - t0 for i, v in t.items()}''',
     '''def frame_times():
    """seconds of each frame relative to the reference frame, indexed 1..N."""
    return dict(P.TIMES)'''),
    ("def diag(i, j, pole_display=(1700, 717)):", "def diag(i, j, pole_display=P.POLE_DISPLAY):"),
    ("def run_all(pole_display=(1700, 717)):", "def run_all(pole_display=P.POLE_DISPLAY):"),
    ("for i in range(1, 91)]", "for i in range(1, P.N + 1)]"),
    ('    print("registered", len(Hs), "of 90")', '    print("registered", len(Hs), "of", P.N)'),
])
edit("fitmodel.py", [("N = 90", "N = R.P.N")])
edit("warp2.py", [
    ('    n = 2 * i - 1\n    d = fits.getdata(f"{W}/light_{n:05d}.fit").astype(np.float32) / 65535.0',
     '    d = fits.getdata(R.P.light(i)).astype(np.float32) / 65535.0'),
    ("p.imap_unordered(one, range(1, 91))", "p.imap_unordered(one, range(1, R.P.N + 1))"),
])
edit("render.py", [
    ("EXPOSURE = 8.0", "EXPOSURE = R.P.EXPOSURE\nP = R.P"),
    ("WB = np.array([1659, 1024, 2378], np.float32) / 1024.0", "WB = np.array(R.P.WB, np.float32) / 1024.0"),
    ('_idx = [i for i in range(1, 91) if _m["have"][i]]', '_idx = [i for i in range(1, R.P.N + 1) if _m["have"][i]]'),
    ('    return fits.getdata(f"{W}/full_{i:05d}.fit").astype(np.float32) / 65535.0', '    return fits.getdata(R.P.full(i)).astype(np.float32) / 65535.0'),
])
edit("padstack.py", [("N = 90", "N = R.P.N"),
                     ('        d = fits.getdata(f"{W}/light_{2 * i - 1:05d}.fit").astype(np.float32) / 65535.0',
                      '        d = fits.getdata(R.P.light(i)).astype(np.float32) / 65535.0')])
edit("countmap.py", [("for i in range(1, 91):", "for i in range(1, Rn.P.N + 1):"),
                     ('    d = fits.getdata(f"{W}/light_{2*i-1:05d}.fit")[1]', '    d = fits.getdata(Rn.P.light(i))[1]')])
edit("clouds.py", [(NH, "import project as P\nW = P.W"), ("FRAMES = list(range(1, 181, 2))", "FRAMES = list(P.IDS)"),
                   ("sky_rows = slice(2400 // B, None)", "sky_rows = slice((P.HEIGHT - P.SKY_ROWS) // B, None)")])
edit("reblank.py", [(NH, "import project as P\nW = P.W"),
                    ('    shutil.copyfile(f"{W}/full_{i:05d}.fit", f"{W}/light_{n:05d}.fit")', '    shutil.copyfile(P.full(i), P.light(i))'),
                    ('    with fits.open(f"{W}/light_{n:05d}.fit", mode="update") as hd:', '    with fits.open(P.light(i), mode="update") as hd:')])
edit("timelapse.py", [("N = 90", "N = Rn.P.N")])
edit("traffic.py", [("for i in range(1, 91):", "for i in range(1, Rn.P.N + 1):")])
edit("mask.py", [(NH, "import project as P\nW = P.W"), ("    hy = (H - 2900) // S ", "    hy = (H - P.SKY_ROWS) // S ")])
edit("solve.py", [('DATE = "2026-09-07T03:40:00"', "DATE = Rn.P.DATE"),
                  ("ASTAP_CENTER = (105.58397642017538, 46.416752071632374)", "ASTAP_CENTER = Rn.P.ASTAP_CENTER")])
for name in ("export_trails.py", "mood.py", "print.py", "solve.py", "traffic_still.py"):
    p = E / name
    s = p.read_text(encoding="utf-8").replace("NH_Lake_16mm_", "{Rn.NAME}_")
    p.write_text(s, encoding="utf-8")
edit("render.py", [("W = R.W\n", "W = R.W\nNAME = R.P.NAME\n")])
p = E / "encode.sh"
p.write_text(p.read_text().replace("NH_Lake_16mm_", "${NAME}_").replace("cd /d/AstroWork/nhlake", 'cd "$(cygpath "$ASTRO_WORK")"\nNAME=$(python -c "import json,os;print(json.load(open(os.path.join(os.environ[\'ASTRO_WORK\'],\'project.json\')))[\'name\'])")'))
print("edited")
