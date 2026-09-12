"""The numbers that decide a night, on one screen: gates of the current run, model fit, clouds,
footprint, tone, star counts; and report.jpg, one small contact sheet (mask overlay, refined treeline,
composite) so a night is checked with one look."""
import os, re, json, glob

from PIL import Image
import project as P

W = P.W


def lines_of(path, pat, n=6):
    if not os.path.exists(path):
        return []
    return [l.rstrip() for l in open(path, errors="replace").read().replace("\r", "\n").split("\n") if re.search(pat, l)][-n:]


chain = open(f"{W}/chain.log", errors="replace").read().strip().split("\n") if os.path.exists(f"{W}/chain.log") else []
starts = [i for i, l in enumerate(chain) if l.startswith("== run")]
chain = chain[starts[-1]:] if starts else chain
print(f"{P.NAME}: {P.N} frames, {P.WIDTH}x{P.HEIGHT}, exposure {P.EXPOSURE:.0f}s, pole prior {tuple(round(v) for v in P.POLE_DISPLAY)}")
print("run:", chain[0] if chain else "none")
for l in chain:
    if l.startswith("  gate") or l.startswith("FAIL") or l.startswith("DONE") or l.startswith("STOPPED"):
        print(" ", l.strip())
for l in lines_of(f"{W}/log_fit.log", r"^tol 3|median resid"):
    print("  fit:", l.strip())
if os.path.exists(f"{W}/tone.json"):
    t = json.load(open(f"{W}/tone.json"))
    print(f"  tone: black {[round(b * 65535, 1) for b in t['black']]} ADU, white {t['white'][0] * 65535:.0f} ADU, a {t['a']}")
for l in lines_of(f"{W}/log_tone.log", r"footprint"):
    print("  " + l.strip())
outs = sorted(glob.glob(f"{W}/{P.NAME}_*.*"))
print("  outputs:", ", ".join(os.path.basename(o) for o in outs) or "none yet")

panels = []
for name, width in (("preview_mask_overlay.jpg", 500), ("preview_mask_refined.jpg", 1100), ("preview_tone.jpg", 500)):
    p = f"{W}/{name}"
    if os.path.exists(p):
        im = Image.open(p); im.thumbnail((width, 700), Image.LANCZOS); panels.append(im)
if panels:
    h = max(p.height for p in panels); w = sum(p.width for p in panels) + 10 * (len(panels) - 1)
    sheet = Image.new("RGB", (w, h), (20, 20, 20)); x = 0
    for p in panels:
        sheet.paste(p, (x, 0)); x += p.width + 10
    sheet.save(f"{W}/report.jpg", quality=85)
    print(f"  report.jpg {sheet.size[0]}x{sheet.size[1]}: mask overlay | refined treeline | composite")
