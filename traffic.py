"""Traffic map: everything the rejection stacking threw away. Sentence: the positive residual of
each registered frame against the stack is what moved; the max over frames is every streak of the
night on one clean sky. Also scores each frame's longest streak so meteor candidates surface.

Writes traffic_max.npy (3,H,W float, linear signal), traffic_streaks.json, preview_traffic.jpg,
and a contact sheet of the top streak frames.
"""
import os, json, numpy as np, cv2
from scipy import ndimage as ndi
from PIL import Image
import render as Rn

W = Rn.W
sky = Rn.load_sky()
valid = sky[1] > 0
mask = np.load(f"{W}/mask_sky_d.npy") & valid
sm = ndi.binary_erosion(mask, iterations=12)
acc = np.zeros_like(sky)
streaks = []
noise = None
for i in range(1, Rn.P.N + 1):
    if not os.path.exists(f"{W}/warped/{i:05d}.npy"):                       # frames without a registration are not warped
        continue
    f = Rn.load_warped(i)
    v = (f[1] > 0) & sm
    res = np.where(v[None], f - sky, 0)
    # the residual noise floor from this frame, once
    if noise is None:
        noise = 1.4826 * np.median(np.abs(res[1][v][::53]))
    np.maximum(acc, res, out=acc)
    # streak score: threshold the green residual, keep long thin components
    g = ndi.gaussian_filter(res[1], 1.5)
    b = g > 5 * noise
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    best = None
    if n:
        objs = ndi.find_objects(lab)
        for k, sl in enumerate(objs, start=1):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            L = np.hypot(h, w)
            if L < 60:
                continue
            area = int((lab[sl] == k).sum())
            thin = area / max(L, 1)
            if thin < 12 and (best is None or L > best["length"]):
                best = {"frame": i, "length": float(L), "area": area, "row": int(sl[0].start), "col": int(sl[1].start), "h": int(h), "w": int(w)}
    if best:
        streaks.append(best)
    print(f"frame {i:2d} noise {noise*65535:.2f} ADU streak {best['length'] if best else 0:.0f}", flush=True)

acc -= np.median(acc[:, sm][:, ::53], axis=1)[:, None, None]        # a max over N frames always floats on a noise floor;
np.clip(acc, 0, None, out=acc)                                      # what stands above it is the traffic
np.save(f"{W}/traffic_max.npy", acc)
streaks.sort(key=lambda s: -s["length"])
json.dump(streaks, open(f"{W}/traffic_streaks.json", "w"), indent=1)
print("frames with streaks:", len(streaks))
# preview: the traffic layer on a dark version of the stack
t = Rn.Tone()
base = t.apply(np.maximum(sky - Rn.BLACK, 0))
layer = np.clip(acc / (noise * 25), 0, 1)
out = np.clip(base.astype(np.float32) * 0.35 + 255 * layer, 0, 255).astype(np.uint8)
Image.fromarray(Rn.to_display(out)[::3, ::3]).save(f"{W}/preview_traffic.jpg", quality=90)
# contact sheet of top 8 streak frames
tiles = []
for s in streaks[:8]:
    f = Rn.load_warped(s["frame"])[1]
    r0, c0 = max(0, s["row"] - 100), max(0, s["col"] - 100)
    crop = (f - sky[1])[r0:r0 + s["h"] + 200, c0:c0 + s["w"] + 200]
    crop = cv2.resize(np.clip(crop / (noise * 20), 0, 1), (400, 300))[::-1]
    tiles.append((crop * 255).astype(np.uint8))
if tiles:
    while len(tiles) % 4:
        tiles.append(np.zeros((300, 400), np.uint8))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    Image.fromarray(np.vstack(rows)).save(f"{W}/preview_streaks.jpg", quality=90)
print("done")
