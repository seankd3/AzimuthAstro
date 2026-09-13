"""One project = one directory with a project.json. Every engine script reads its facts from here.
Select the project directory with the ASTRO_WORK environment variable (required).

project.json:
  name            output file prefix
  width, height   frame size in display orientation
  frame_ids       underlying frame numbers, in time order; frame i (1-based) is frame_ids[i-1]
  ref             1-based index of the reference frame (mid-sequence)
  exposure        seconds
  times           seconds of each frame relative to the reference frame (list aligned with frame_ids)
  pole_display    (x, y) of the celestial pole in display pixels, from a trail stack
  sky_rows        display rows above this are certainly sky (for the mask)
  astap_center    (RA, Dec) degrees of the frame centre from an ASTAP crop solve (for solve.py)
  date            ISO UTC of mid-session
  wb              as-shot RGGB multipliers R, G, B (G = 1024)
  cam_to_srgb     3x3 taking white-balanced camera RGB to linear sRGB (rows sum to 1)
  lens, focal     EXIF LensModel and focal length in mm (lensfun distortion profile)
  crop            35 mm crop factor of the sensor (focal length in pixels)
"""
import os, json

W = os.environ.get("ASTRO_WORK")
if not W:
    raise SystemExit("set ASTRO_WORK to the project directory (holds project.json)")
_cfg = json.load(open(os.path.join(W, "project.json")))
NAME = _cfg["name"]
WIDTH, HEIGHT = int(_cfg["width"]), int(_cfg["height"])
IDS = list(_cfg["frame_ids"])
N = len(IDS)
REF = int(_cfg["ref"])
EXPOSURE = float(_cfg["exposure"])
TIMES = {i + 1: float(t) for i, t in enumerate(_cfg["times"])}
POLE_DISPLAY = tuple(_cfg["pole_display"])
SKY_ROWS = int(_cfg["sky_rows"])
ASTAP_CENTER = tuple(_cfg.get("astap_center", (0.0, 0.0)))
ASTAP_SCALE = float(_cfg.get("astap_scale", 0.0))           # deg/px of the stack at its centre, from the ASTAP crop solve
DATE = _cfg.get("date", "2000-01-01T00:00:00")
WB = tuple(_cfg.get("wb", (1659, 1024, 2378)))
CAM_TO_SRGB = _cfg.get("cam_to_srgb") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
LENS = _cfg.get("lens", "RF16mm F2.8 STM")
FOCAL = float(_cfg.get("focal", 16.0))
CROP = float(_cfg.get("crop", 1.0))          # 35 mm crop factor


def full(i):
    """Unblanked, hot-patched frame i, original geometry."""
    return f"{W}/full_{IDS[i - 1]:05d}.fit"
