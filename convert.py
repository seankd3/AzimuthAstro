"""CR3 -> full_NNNNN.fit (16-bit, debayered, rotated to display orientation).
Runs Siril on a link farm of the project's source files, then rotates in place.
"""
import os, json, subprocess, numpy as np
from astropy.io import fits
import project as P

cfg = json.load(open(f"{P.W}/project.json"))
folder = cfg["source_folder"]
farm = f"{P.W}/in"
os.makedirs(farm, exist_ok=True)
for k, src in enumerate(cfg["sources"], start=1):
    dst = f"{farm}/{k:05d}.CR3"
    if not os.path.exists(dst):
        os.link(os.path.join(folder, src), dst)
ssf = f"{P.W}/convert.ssf"
open(ssf, "w").write(f"requires 1.2.0\ncd {P.W.replace(chr(92), '/')}/in\nsetext fit\nset16bits\nsetcpu 14\nconvert full -debayer -out=.. -start=1\n")
subprocess.run([r"C:\Program Files\Siril\bin\siril-cli.exe", "-s", ssf], stdout=open(f"{P.W}/convert.log", "w"), stderr=subprocess.STDOUT)
assert "Conversion succeeded" in open(f"{P.W}/convert.log").read(), "siril conversion failed"

orient = cfg.get("orientation", "Horizontal (normal)")
k = {"Rotate 90 CW": 1, "Rotate 270 CW": 3, "Rotate 180": 2}.get(orient, 0)   # FITS rows run bottom-up, which reverses np.rot90's CCW sense
if k:
    for i in range(1, P.N + 1):
        hdr = fits.getheader(P.full(i))
        if (hdr["NAXIS2"], hdr["NAXIS1"]) == (P.HEIGHT, P.WIDTH):        # already in display orientation (resumed run)
            continue
        with fits.open(P.full(i), mode="update") as h:
            h[0].data = np.ascontiguousarray(np.rot90(h[0].data, k=k, axes=(1, 2)))
            h.flush()
    print("rotated", P.N, "frames by", orient, flush=True)
