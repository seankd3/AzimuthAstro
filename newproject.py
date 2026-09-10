"""Create project.json from a CR3 folder and a selection.
usage: python newproject.py <workdir> <name> <cr3_folder> <first> <last> [--skip n,n,...] [--pole x,y] [--skyrows r]
  first/last: 1-based positions in the sorted CR3 list (inclusive). frame_ids become 1..N in that order;
  the caller converts exactly those files in that order (see convert.py).
"""
import sys, os, json, glob, subprocess, datetime

work, name, folder, first, last = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
opts = dict(zip(sys.argv[6::2], sys.argv[7::2]))
skip = set(int(x) for x in opts.get("--skip", "").split(",") if x)
files = sorted(glob.glob(os.path.join(folder, "*.CR3")))
sel = [f for k, f in enumerate(files, start=1) if first <= k <= last and k not in skip]
tags = subprocess.run(["exiftool", "-q", "-p", "$FileName\t$DateTimeOriginal\t$ExposureTime\t$Orientation\t$WB_RGGBLevelsAsShot\t$ImageWidth\t$ImageHeight"] + sel,
                      capture_output=True, text=True).stdout.strip().split("\n")
rows = [t.split("\t") for t in tags]
ts = [datetime.datetime.strptime(r[1], "%Y:%m:%d %H:%M:%S").timestamp() for r in rows]
ref = len(sel) // 2 + 1
times = [t - ts[ref - 1] for t in ts]
exposure = float(rows[0][2])
orient = rows[0][3]
w, h = 8191, 5463                                  # Siril drops the last row/column of the R5 frame
if "90" in orient or "270" in orient:
    w, h = h, w
wb = [int(x) for x in rows[0][4].split()]
cfg = {"name": name, "width": w, "height": h, "frame_ids": list(range(1, len(sel) + 1)), "ref": ref,
       "exposure": exposure, "times": times, "pole_display": [float(x) for x in opts.get("--pole", "0,0").split(",")],
       "sky_rows": int(opts.get("--skyrows", h // 2)), "orientation": orient, "wb": [wb[0], wb[1], wb[3]],
       "date": datetime.datetime.utcfromtimestamp(ts[ref - 1]).strftime("%Y-%m-%dT%H:%M:%S"),
       "source_folder": folder, "sources": [os.path.basename(f) for f in sel]}
os.makedirs(work, exist_ok=True)
json.dump(cfg, open(os.path.join(work, "project.json"), "w"), indent=1)
print(f"{name}: {len(sel)} frames, {w}x{h}, exposure {exposure}s, orientation {orient}, ref {ref}, span {times[-1]-times[0]:.0f}s")
