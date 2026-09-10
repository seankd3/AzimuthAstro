"""u_sky_NNNNN.fit = blanked frame i through the lensfun map only (no fitted residual yet).
These exist only to detect stars for the registration fit; 16-bit, zero rims kept."""
import numpy as np, cv2
from astropy.io import fits
from multiprocessing import Pool
import project as P

L = np.load(f"{P.W}/undist_coords.npy")
mx, my = np.ascontiguousarray(L[..., 0]), np.ascontiguousarray(L[..., 1])
RIM = 5


def one(i):
    d = fits.getdata(P.light(i)).astype(np.float32)
    zero = (d == 0).any(axis=0).astype(np.uint8)
    out = np.stack([np.clip(cv2.remap(c, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, None) for c in d])
    zw = cv2.dilate(cv2.remap(zero, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1), np.ones((2 * RIM + 1,) * 2, np.uint8))
    out[:, zw > 0] = 0
    fits.PrimaryHDU(np.round(out).astype(np.uint16)).writeto(f"{P.W}/u_sky_{i:05d}.fit", overwrite=True)
    return i


if __name__ == "__main__":
    with Pool(5) as p:
        for i in p.imap_unordered(one, range(1, P.N + 1)):
            if i % 20 == 0:
                print("undistorted", i, flush=True)
    print("done")
