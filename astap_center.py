"""Solve a central crop of the stack with ASTAP and write astap_center (RA, Dec of the frame centre)
into project.json. The full-frame solve (solve.py) starts from it."""
import json, subprocess, os, numpy as np
from astropy.io import fits
import project as P

import render as Rn
s = Rn.load_sky()[1]
H, W = s.shape
cy, cx = H // 2, W // 2
hh, hw = 800, 1000
crop = np.clip(s[cy - hh:cy + hh, cx - hw:cx + hw] * 65535 - 2047, 0, None).astype(np.float32)
path = f"{P.W}/astap_crop.fits"
fits.PrimaryHDU(crop).writeto(path, overwrite=True)
fov = 2 * hh * np.degrees(1.0 / Rn.FPX)                            # the stack's own scale: the fitted pinhole focal length
r = subprocess.run([r"C:\Program Files\astap\astap_cli.exe", "-f", path.replace("\\", "/"), "-r", "180", "-fov", f"{fov:.1f}", "-z", "0", "-update"], capture_output=True, text=True)
print(r.stdout[-400:])
ini = f"{P.W}/astap_crop.ini"
assert os.path.exists(ini) and "PLTSOLVD=T" in open(ini).read(), "ASTAP did not solve"
kv = dict(l.strip().split("=", 1) for l in open(ini) if "=" in l)
ra, dec = float(kv["CRVAL1"]), float(kv["CRVAL2"])
cfg = json.load(open(f"{P.W}/project.json")); cfg["astap_center"] = [ra, dec]; cfg["astap_scale"] = abs(float(kv["CDELT1"]))   # deg/px at the centre
json.dump(cfg, open(f"{P.W}/project.json", "w"), indent=1)
print(f"centre RA {ra:.4f} Dec {dec:.4f}, scale {abs(float(kv['CDELT1']))*3600:.1f} arcsec/px")
