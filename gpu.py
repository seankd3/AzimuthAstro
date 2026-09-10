"""GPU resampling for the engine (torch, CUDA). Same signatures as the cv2 calls they replace;
falls back to cv2 when CUDA is missing. Bicubic on the GPU versus lanczos4 on the CPU: the
difference on stars is below the stack noise, the speed difference is ~20x on a 3050 Ti.

warp_perspective(img3, H, (w, h))  -> (3, h, w)
remap(img3, mapx, mapy)            -> (3, H, W) sampling img3 at (mapx, mapy)
"""
import numpy as np, cv2

try:
    import torch, torch.nn.functional as F
    CUDA = torch.cuda.is_available()
except Exception:                      # pragma: no cover
    CUDA = False


def _sample(img3, gx, gy, mode):
    """img3 (3,Hs,Ws) float32 numpy; gx, gy (Hd,Wd) float32 source pixel coords. One channel at a time
    keeps peak memory under 2 GB for a 45 MP frame."""
    Hs, Ws = img3.shape[-2:]
    grid = torch.empty((1, gx.shape[0], gx.shape[1], 2), dtype=torch.float32, device="cuda")
    grid[0, ..., 0] = torch.from_numpy(gx).cuda() * (2.0 / (Ws - 1)) - 1.0
    grid[0, ..., 1] = torch.from_numpy(gy).cuda() * (2.0 / (Hs - 1)) - 1.0
    out = np.empty((3,) + gx.shape, np.float32)
    for c in range(3):
        src = torch.from_numpy(np.ascontiguousarray(img3[c])).cuda()[None, None]
        res = F.grid_sample(src, grid, mode=mode, padding_mode="zeros", align_corners=True)
        out[c] = res[0, 0].clamp_(min=0).cpu().numpy()
        del src, res
    del grid
    torch.cuda.empty_cache()
    return out


def warp_perspective(img3, H, size, nearest=False):
    w, h = size
    if not CUDA:
        flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
        return np.stack([cv2.warpPerspective(c, H, (w, h), flags=flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in img3])
    Hinv = np.linalg.inv(H).astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    den = Hinv[2, 0] * xx + Hinv[2, 1] * yy + Hinv[2, 2]
    gx = ((Hinv[0, 0] * xx + Hinv[0, 1] * yy + Hinv[0, 2]) / den).astype(np.float32)
    gy = ((Hinv[1, 0] * xx + Hinv[1, 1] * yy + Hinv[1, 2]) / den).astype(np.float32)
    return _sample(img3, gx, gy, "nearest" if nearest else "bicubic")


def remap(img3, mapx, mapy, nearest=False):
    if not CUDA:
        flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
        return np.stack([cv2.remap(c, mapx, mapy, flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in img3])
    return _sample(img3, np.ascontiguousarray(mapx, np.float32), np.ascontiguousarray(mapy, np.float32), "nearest" if nearest else "bicubic")
