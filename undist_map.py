"""lensfun undistortion map for the project's frame size -> undist_coords.npy (H, W, 2): for each
output pixel, the source pixel (x, y). Radial model, so orientation does not matter."""
import numpy as np, lensfunpy, os
import project as P
db = lensfunpy.Database(paths=[os.path.join(os.path.dirname(__file__), "lensfun-mil-canon.xml")], load_common=False)
lens = [l for l in db.lenses if "16mm" in l.model and "RF" in l.model][0]
mod = lensfunpy.Modifier(lens, 1.0, P.WIDTH, P.HEIGHT)
mod.initialize(16.0, 4.0, 1000.0, pixel_format=np.float32, flags=lensfunpy.ModifyFlags.DISTORTION)
coords = mod.apply_geometry_distortion()
np.save(f"{P.W}/undist_coords.npy", coords.astype(np.float32))
print("map", coords.shape, "corner src", coords[0, 0], "centre", coords[P.HEIGHT // 2, P.WIDTH // 2])
