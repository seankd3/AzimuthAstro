"""Mood still: the clean stacked sky with one chosen frame's cirrus blended back in.
Cloud layer = that frame (undistorted) minus the clear-sky stack, stars removed, inside the sky mask.
"""
import sys, numpy as np, tifffile
from scipy import ndimage as ndi
from PIL import Image
import render as Rn

W = Rn.W
i = int(sys.argv[1]) if len(sys.argv) > 1 else 68          # sharp-frame index (CR3 #135)
gain = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
frame = Rn.to_d(Rn.load_full(i))
ground = Rn.load_fits("ground_d")
mask = np.load(f"{W}/mask_sky_d.npy")
layer = frame - ground
layer[:, ~mask] = 0
layer = np.stack([ndi.gaussian_filter(ndi.median_filter(c[::2, ::2], 5), 2) for c in layer])
layer = np.stack([np.kron(c, np.ones((2, 2), np.float32))[:mask.shape[0], :mask.shape[1]] for c in layer])
layer = np.clip(layer, 0, None)
lin = np.load(f"{W}/composite_lin.npy") + gain * layer
img8 = Rn.Tone().apply(lin)
Image.fromarray(Rn.to_display(img8)).save(f"{W}/{Rn.NAME}_Mood_f{i}.jpg", quality=94)
lin16 = np.clip(lin * 65535.0 * 4.0 * Rn.WB[:, None, None], 0, 65535).astype(np.uint16)
tifffile.imwrite(f"{W}/{Rn.NAME}_Mood_f{i}.tif", np.moveaxis(lin16, 0, -1)[::-1], photometric="rgb", compression="zlib")
print("wrote mood", i)
