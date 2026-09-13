"""Traffic map still: the finished composite with every streak of the night lightened onto the sky."""
import numpy as np
import render as Rn
W = Rn.W
acc = np.load(f"{W}/traffic_max.npy")
lin = np.load(f"{W}/composite_lin.npy")
mask = np.load(f"{W}/mask_sky_d.npy")
acc[:, ~mask] = 0
acc = np.clip(acc - np.median(acc[:, mask][:, ::53], axis=1)[:, None, None], 0, None)   # only what stands above the noise floor
out = np.maximum(lin, lin + acc)                  # additive where something moved: streaks keep their colour
Rn.save_still("Traffic", Rn.Tone(), out)
print("ok")
