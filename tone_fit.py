"""Build the composite (stationary background + rotating stars + static ground), fit the shared
tone curve once, write composite_lin.npy, stars_layer.npy, bg_layer.npy and a preview.
usage: python tone_fit.py [a] [sky_target]
"""
import sys, numpy as np, render as Rn
from PIL import Image
sky = Rn.load_fits("sky"); ground = Rn.load_fits("ground_d"); ground_mean = Rn.load_fits("ground_mean_d")
mask = np.load(f"{Rn.W}/mask_sky_d.npy")
count = np.load(f"{Rn.W}/count_map.npy")
stars, bg, resid = Rn.backgrounds(sky, ground, mask, count)
np.save(f"{Rn.W}/resid_layer.npy", resid.astype(np.float32))
valid = mask.copy()                       # the whole sky has a residual now (static grain where the aligned stack is thin)
np.save(f"{Rn.W}/valid_layer.npy", valid)
np.save(f"{Rn.W}/stars_layer.npy", stars.astype(np.float32))
np.save(f"{Rn.W}/bg_layer.npy", bg.astype(np.float32))
lin = Rn.compose(stars, bg, ground_mean, mask, resid, valid)
np.save(f"{Rn.W}/composite_lin.npy", lin.astype(np.float32))
a = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.18
core = mask & (sky[1] > 0)
t = Rn.Tone.fit(lin, a=a, sky_mask=core[::7, ::7], sky_target=target)
print("tone black", t.black * 65535, "white", t.white[0] * 65535, "a", a)
img = Rn.to_display(t.apply(lin))
Image.fromarray(img[::4, ::4]).save(f"{Rn.W}/preview_tone.jpg", quality=92)
Image.fromarray(img[3300:3700, 3300:4100]).save(f"{Rn.W}/preview_seam.jpg", quality=92)
Image.fromarray(img[3100:3500, 5800:6600]).save(f"{Rn.W}/preview_seam2.jpg", quality=92)
print("ok")
