"""Finished print version. Sentence: take the composite, flatten its sky background (large-scale
masked smoothing of the star-free background, removed at 80% so some natural horizon glow stays),
and stretch with a print-strength curve of its own.
Writes {Rn.NAME}_Print.jpg and .tif (16-bit sRGB).
"""
import numpy as np, cv2
from scipy import ndimage as ndi
import render as Rn

W = Rn.W
box = Rn.footprint()                                                       # everything below works inside the data footprint
lin = Rn.crop(np.load(f"{W}/composite_lin.npy"), box)
bg = Rn.crop(np.load(f"{W}/bg_layer.npy"), box) - Rn.BLACK                # the star-free sky background, already stationary
mask = Rn.crop(np.load(f"{W}/mask_sky_d.npy"), box)

SIG = 250 / 8
msub = mask[::8, ::8].astype(np.float32)
den = ndi.gaussian_filter(msub, SIG)
soft = ndi.gaussian_filter(ndi.binary_dilation(msub > 0, iterations=4).astype(np.float32), 2)
soft_full = cv2.resize(soft, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
for c in range(3):
    z = bg[c][::8, ::8] * msub
    smooth = ndi.gaussian_filter(z, SIG) / np.maximum(den, 1e-3)
    level = float(np.median(smooth[msub > 0]))
    corr = cv2.resize((smooth - level).astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
    lin[c] -= 0.8 * corr * soft_full
    print(f"ch{c}: background range {(smooth[msub>0].max()-smooth[msub>0].min())*65535:.1f} ADU")

tone = Rn.Tone.fit(lin, a=110.0, sky_target=0.20, sky_mask=mask[::7, ::7], path=f"{W}/tone_print.json")
Rn.save_still("Print", tone, lin, box=(0, lin.shape[1], 0, lin.shape[2]))
print("done")
