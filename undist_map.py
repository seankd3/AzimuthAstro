"""lensfun undistortion map for the project's frame size -> undist_coords.npy (H, W, 2)."""
import numpy as np, optics
import project as P
coords = optics.undist_map(P.WIDTH, P.HEIGHT, P.LENS, P.FOCAL)
np.save(f"{P.W}/undist_coords.npy", coords)
print("map", coords.shape, "corner src", coords[0, 0], "centre", coords[P.HEIGHT // 2, P.WIDTH // 2])
