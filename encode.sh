cd "$(cygpath "$ASTRO_WORK")"
NAME=$(python -c "import json,os;print(json.load(open(os.path.join(os.environ['ASTRO_WORK'],'project.json')))['name'])")
FF=/c/ffmpeg/bin/ffmpeg
enc() {  # dir fps out
  "$FF" -y -framerate "$2" -i "$1/%04d.jpg" -vf "scale=3840:-2:flags=lanczos,format=yuv420p" -c:v libx264 -preset slow -crf 16 -movflags +faststart "$3" > "encode_$(basename $1).log" 2>&1 && echo "encoded $3"
}
[ -d trails_frames ] && enc trails_frames 24 ${NAME}_TrailsGrowing_4K.mp4
[ -d timelapse_locked ] && enc timelapse_locked 24 ${NAME}_SkyLocked_4K.mp4
[ -d timelapse_standard ] && enc timelapse_standard 24 ${NAME}_Timelapse_4K.mp4
[ -d timelapse_clouds ] && enc timelapse_clouds 12 ${NAME}_CloudsOnly_4K.mp4
echo DONEENC
