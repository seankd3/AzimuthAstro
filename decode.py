"""CR3 -> full_NNNNN.fit: LibRaw debayer (AHD) to raw sensor ADU with the camera's black level kept (the
stages fit the black), no white balance, no colour matrix, rotated to display orientation. Written
16-bit with FITS rows bottom-up, the layout every later stage reads."""
import os, json, numpy as np, rawpy
from astropy.io import fits
from multiprocessing import Pool
import project as P

cfg = json.load(open(f"{P.W}/project.json"))
K = {"Rotate 90 CW": 3, "Rotate 270 CW": 1, "Rotate 180": 2}.get(cfg.get("orientation", ""), 0)   # np.rot90 turns counter-clockwise in a y-down image


def one(i):
    src = os.path.join(cfg["source_folder"], cfg["sources"][i - 1])
    with rawpy.imread(src) as r:
        black, white = float(np.mean(r.black_level_per_channel)), float(r.white_level)
        rgb = r.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=False, use_auto_wb=False,
                            user_wb=[1, 1, 1, 1], demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.raw, user_flip=0)
    adu = rgb.astype(np.float32) * ((white - black) / 65535.0) + black      # LibRaw scales (black..white) to 0..65535; undo it
    img = np.moveaxis(adu, -1, 0)                                            # (3, H, W), y down
    if K:
        img = np.rot90(img, K, axes=(1, 2))
    fits.PrimaryHDU(np.round(img[:, ::-1, :]).astype(np.uint16)).writeto(P.full(i), overwrite=True)
    return i


if __name__ == "__main__":
    with Pool(3) as p:
        for i in p.imap_unordered(one, range(1, P.N + 1)):
            print(i, end=" ", flush=True)
    print("\ndecoded", P.N, "frames", f"({P.WIDTH}x{P.HEIGHT}, {cfg.get('orientation', 'Horizontal (normal)')})", flush=True)
