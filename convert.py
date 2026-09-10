"""CR3 -> full_NNNNN.fit (16-bit, debayered, rotated to display orientation, hot pixels median-patched).
Runs Siril on a link farm of the project's source files, then post-processes in place.
"""
import os, json, subprocess, numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
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
        with fits.open(P.full(i), mode="update") as h:
            h[0].data = np.ascontiguousarray(np.rot90(h[0].data, k=k, axes=(1, 2)))
            h.flush()
    print("rotated", P.N, "frames by", orient, flush=True)

# hot pixels: bright in the minimum over six frames spread through the night; median-patch them
picks = [max(1, round(x)) for x in np.linspace(1, P.N, 6)]
mn = None
for i in picks:
    d = fits.getdata(P.full(i)).astype(np.float32)
    mn = d if mn is None else np.minimum(mn, d)
hot = np.zeros(mn.shape[1:], bool)
for c in range(3):
    diff = mn[c] - ndi.median_filter(mn[c], 5)
    sig = 1.4826 * np.median(np.abs(diff[::7, ::7]))          # noise of the minimum image
    hot |= diff > max(8 * sig, 25)
lab, n = ndi.label(hot)
sizes = ndi.sum(hot, lab, index=np.arange(1, n + 1))
hot = ndi.binary_dilation(np.isin(lab, np.arange(1, n + 1)[sizes <= 25]), iterations=1)
np.save(f"{P.W}/hot_mask.npy", hot)
print("hot pixels", int(hot.sum()), flush=True)
for i in range(1, P.N + 1):
    with fits.open(P.full(i), mode="update") as h:
        d = h[0].data
        for c in range(3):
            med = ndi.median_filter(d[c], 5)
            d[c][hot] = med[hot]
        h.flush()
    if i % 50 == 0:
        print("patched", i, flush=True)
print("done")
