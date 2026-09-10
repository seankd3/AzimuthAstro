"""Frame folders -> 4K H.264 videos with ffmpeg: growing trails, sky-locked, standard and clouds-only timelapses."""
import os, shutil, subprocess
import project as P

FF = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
VIDEOS = [("trails_frames", 24, "TrailsGrowing"), ("timelapse_locked", 24, "SkyLocked"),
          ("timelapse_standard", 24, "Timelapse"), ("timelapse_clouds", 12, "CloudsOnly")]
for folder, fps, tag in VIDEOS:
    src = os.path.join(P.W, folder)
    if not os.path.isdir(src):
        continue
    out = os.path.join(P.W, f"{P.NAME}_{tag}_4K.mp4")
    with open(os.path.join(P.W, f"encode_{folder}.log"), "w") as lf:
        rc = subprocess.run([FF, "-y", "-framerate", str(fps), "-i", os.path.join(src, "%04d.jpg"),
                             "-vf", "scale=3840:-2:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                             "-movflags", "+faststart", out], stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"{tag}: {len(os.listdir(src))} frames -> {os.path.getsize(out) / 1e6:.0f} MB" if rc == 0 else f"{tag}: ffmpeg failed (rc {rc})", flush=True)
    if rc:
        raise SystemExit(1)
