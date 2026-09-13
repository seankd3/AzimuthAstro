"""Mood still: the clean stacked sky with one chosen frame's cirrus blended back in.
Cloud layer = that frame (undistorted) minus the clear-sky stack, stars removed, inside the sky mask.
"""
import sys, numpy as np
from scipy import ndimage as ndi
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
Rn.save_still(f"Mood_f{i}", Rn.Tone(), lin)
print("wrote mood", i)
