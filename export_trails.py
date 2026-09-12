"""Stills from the trail engine: gapless and comet trails over the stationary background and ground."""
import numpy as np
import render as Rn

W = Rn.W
ground = Rn.load_fits("ground_d"); bg = np.load(f"{W}/bg_layer.npy"); resid = np.load(f"{W}/resid_layer.npy"); valid = np.load(f"{W}/valid_layer.npy")
mask = np.load(f"{W}/mask_sky_d.npy")
tone = Rn.Tone()
box = Rn.footprint()
for name in ("gapless", "comet"):
    tr = np.load(f"{W}/trails_{name}.npy")
    lin = Rn.compose(tr, bg, ground, mask, resid, valid)
    Rn.save_still(f"Trails_{name}", tone.apply(lin), lin, box)
    print("wrote", name)
