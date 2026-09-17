"""CR3 -> full_NNNNN.fit: LibRaw debayer (AHD) to raw sensor ADU with the camera's black level kept (the
stages fit the black), no white balance, no colour matrix, rotated to display orientation, hot pixels
patched (found in the minimum of six frames decoded first). Written 16-bit with FITS rows bottom-up,
the layout every later stage reads."""
import os, json, numpy as np, rawpy, hot
from astropy.io import fits
from multiprocessing import Pool
import project as P

cfg = json.load(open(f"{P.W}/project.json"))
K = {"Rotate 90 CW": 3, "Rotate 270 CW": 1, "Rotate 180": 2}.get(cfg.get("orientation", ""), 0)   # np.rot90 turns counter-clockwise in a y-down image
IDX = None


def decode(i):
    """(3, H, W) float32 sensor ADU of frame i, display orientation, y down."""
    src = os.path.join(cfg["source_folder"], cfg["sources"][i - 1])
    with rawpy.imread(src) as r:
        black, white = float(np.mean(r.black_level_per_channel)), float(r.white_level)
        rgb = r.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=False, use_auto_wb=False,
                            user_wb=[1, 1, 1, 1], demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.raw, user_flip=0)
    img = np.moveaxis(rgb.astype(np.float32) * ((white - black) / 65535.0) + black, -1, 0)   # LibRaw scales (black..white) to 0..65535; undo it
    return np.rot90(img, K, axes=(1, 2)) if K else img


def _init(idx):
    global IDX
    IDX = idx


def write(i):
    img = decode(i)
    if IDX is not None:
        hot.patch(img, IDX)
    fits.PrimaryHDU(np.round(img[:, ::-1, :]).astype(np.uint16)).writeto(P.full(i), overwrite=True)   # FITS rows run bottom-up
    return i


if __name__ == "__main__":
    workers = max(2, (os.cpu_count() or 4) // 2)                            # LibRaw is single-threaded; ~1.2 GB per worker
    picks = [max(1, round(x)) for x in np.linspace(1, P.N, 6)]
    with Pool(workers) as p:
        mn = None
        for d in p.imap_unordered(decode, picks):
            mn = d if mn is None else np.minimum(mn, d)
    mask = hot.find(mn)
    np.save(f"{P.W}/hot_mask.npy", mask[::-1])                                # FITS orientation, like the hot stage writes it
    with rawpy.imread(os.path.join(cfg["source_folder"], cfg["sources"][0])) as r:
        cfg["clip"] = float(r.white_level) / 65535.0                          # the ceiling every later stage must treat as no measurement
    json.dump(cfg, open(f"{P.W}/project.json", "w"), indent=1)
    print("hot pixels", int(mask.sum()), flush=True)
    with Pool(workers, initializer=_init, initargs=(hot.neighbours(mask),)) as p:
        for i in p.imap_unordered(write, range(1, P.N + 1)):
            print(i, end=" ", flush=True)
    print("\ndecoded", P.N, "frames", f"({P.WIDTH}x{P.HEIGHT}, {cfg.get('orientation', 'Horizontal (normal)')})", flush=True)
