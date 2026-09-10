"""Copy a project's finished outputs next to its source folder (../<name>_processed/), with camera
EXIF from the first source frame (no lens tag, so Lightroom does not re-apply the distortion profile)."""
import os, glob, shutil, subprocess, json
import project as P

cfg = json.load(open(f"{P.W}/project.json"))
src = os.path.join(cfg["source_folder"], cfg["sources"][0])
out = os.path.join(os.path.dirname(cfg["source_folder"].rstrip("\\/")), f"{P.NAME}_processed")
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(f"{P.W}/{P.NAME}_*.jpg") + glob.glob(f"{P.W}/{P.NAME}_*.tif") + glob.glob(f"{P.W}/{P.NAME}_*.mp4"))
for f in files:
    dst = os.path.join(out, os.path.basename(f))
    shutil.copyfile(f, dst)
    if dst.lower().endswith((".jpg", ".tif")):
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-tagsFromFile", src, "-Make", "-Model", "-DateTimeOriginal", "-ISO",
                        "-FNumber", "-ExposureTime", "-FocalLength", "-WhiteBalance", "-ColorTemperature", dst])
    print("delivered", os.path.basename(f), f"{os.path.getsize(dst) / 1e6:.0f} MB")
print("->", out)
