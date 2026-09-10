"""Stills from the trail engine: gapless and comet trails over the stationary background and ground."""
import numpy as np, tifffile
from PIL import Image
import render as Rn

W = Rn.W
ground = Rn.load_fits("ground_d"); bg = np.load(f"{W}/bg_layer.npy"); resid = np.load(f"{W}/resid_layer.npy"); valid = np.load(f"{W}/valid_layer.npy")
mask = np.load(f"{W}/mask_sky_d.npy")
tone = Rn.Tone()
for name in ("gapless", "comet"):
    tr = np.load(f"{W}/trails_{name}.npy")
    lin = Rn.compose(tr, bg, ground, mask, resid, valid)
    Image.fromarray(Rn.to_display(tone.apply(lin))).save(f"{W}/{Rn.NAME}_Trails_{name}.jpg", quality=94)
    lin16 = np.clip(lin * 65535.0 * 4.0 * Rn.WB[:, None, None], 0, 65535).astype(np.uint16)
    tifffile.imwrite(f"{W}/{Rn.NAME}_Trails_{name}.tif", np.moveaxis(lin16, 0, -1)[::-1], photometric="rgb", compression="zlib")
    print("wrote", name)
