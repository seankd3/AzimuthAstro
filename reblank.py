"""light_<odd> = full_<i> (unblanked, hot-patched) with ground (new mask) and clouds blanked to 0."""
import numpy as np, shutil
from astropy.io import fits
import project as P
W = P.W
sky = np.load(f"{W}/mask_sky.npy"); H, Wd = sky.shape
cl = np.load(f"{W}/clouds.npz"); masks, frames = cl["masks"], cl["frames"]; B = 8
def up(m):
    full = np.repeat(np.repeat(m, B, axis=0), B, axis=1); out = np.zeros((H, Wd), bool)
    h, w = min(H, full.shape[0]), min(Wd, full.shape[1]); out[:h, :w] = full[:h, :w]; return out
for i, (n, cm) in enumerate(zip(frames, masks), start=1):
    shutil.copyfile(P.full(i), P.light(i))
    blank = ~sky | up(cm)
    with fits.open(P.light(i), mode="update") as hd:
        hd[0].data[:, blank] = 0; hd.flush()
    if n % 20 == 1: print("blanked", n, f"{blank.mean()*100:.1f}%", flush=True)
print("done")
