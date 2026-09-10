"""How many frames contributed to each pixel of the sky stack: the blanked masks of every frame,
warped like the frames, summed. Written at 1/4 scale, upsampled on load. -> count_map.npy (H,W) uint8
"""
import numpy as np, cv2
from astropy.io import fits
import render as Rn
W = Rn.W
S = 0.25
acc = None
for i in range(1, Rn.P.N + 1):
    if not Rn._m["have"][i]:
        continue
    d = fits.getdata(Rn.P.light(i))[1]
    valid = (d > 0).astype(np.float32)[None]
    v = Rn.to_d(valid, Rn.H_FRAME[i], S, interp=cv2.INTER_NEAREST)[0]
    acc = v.copy() if acc is None else acc + v
    if i % 15 == 0: print("frame", i, flush=True)
full = cv2.resize(acc, (Rn.WIDTH, Rn.HEIGHT), interpolation=cv2.INTER_LINEAR)
np.save(f"{W}/count_map.npy", np.clip(np.round(full), 0, 255).astype(np.uint8))
print("done; count percentiles in sky:", np.percentile(full[full > 0], [1, 5, 25, 50]))
