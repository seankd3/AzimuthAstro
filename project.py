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
"""
import os, json

W = os.environ.get("ASTRO_WORK", r"D:\AstroWork\nhlake")
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
DATE = _cfg.get("date", "2000-01-01T00:00:00")
WB = tuple(_cfg.get("wb", (1659, 1024, 2378)))


def light(i):
    """Blanked (ground/cloud = 0) frame i, original geometry."""
    return f"{W}/light_{IDS[i - 1]:05d}.fit"


def full(i):
    """Unblanked, hot-patched frame i, original geometry."""
    return f"{W}/full_{IDS[i - 1]:05d}.fit"
