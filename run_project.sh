#!/bin/bash
# Full chain for a project directory: ASTRO_WORK=D:/AstroWork/<proj> bash run_project.sh [stage]
# Stages run in order from the one given (default: all). Each stage logs to <proj>/log_<stage>.log.
set -u
export ASTRO_WORK
E=/d/AstroWork/engine
P=$(cygpath "$ASTRO_WORK")
PW=$(echo "$ASTRO_WORK" | sed 's#\\#/#g')
SIRIL="/c/Program Files/Siril/bin/siril-cli.exe"
cd "$P"
MOOD_FRAME=${MOOD_FRAME:-1}
start=${1:-convert}
stages=(convert ground mask refine clouds reblank undist register fit warp stack pad count tone astap annotate traffic mood print trails export timelapse encode deliver)
run=0
fail() { echo "FAIL $1" >> chain.log; exit 1; }
for st in "${stages[@]}"; do
  [ "$st" = "$start" ] && run=1
  [ $run = 1 ] || continue
  echo "$(date +%H:%M) $st" >> chain.log
  case $st in
    convert)  [ -f full_00001.fit ] || python $E/convert.py > log_convert.log 2>&1 || fail convert ;;
    ground)   rm -f full_.seq; printf 'requires 1.2.0\ncd %s\nsetext fit\nset32bits\nsetcpu 14\nstack full median -nonorm -out=ground\nstack full rej w 3 3 -nonorm -out=ground_mean\n' "$PW" > ground.ssf
              "$SIRIL" -s "$PW/ground.ssf" > log_ground.log 2>&1; grep -q "Script execution finished successfully" log_ground.log || fail ground
              cp ground.fit ground_fixed.fit ;;
    mask)     python $E/mask.py > log_mask.log 2>&1 || fail mask ;;
    refine)   python $E/mask_refine.py > log_refine.log 2>&1 || fail refine ;;
    clouds)   python $E/clouds.py > log_clouds.log 2>&1 || fail clouds ;;
    reblank)  python $E/reblank.py > log_reblank.log 2>&1 || fail reblank ;;
    undist)   python $E/undist_frames.py > log_undist.log 2>&1 || fail undist ;;
    register) python $E/register.py all > log_register.log 2>&1 || fail register ;;
    fit)      python $E/fitmodel.py > log_fit.log 2>&1 || fail fit ;;
    warp)     rm -f r2_sky_*.fit r2_sky_.seq sky.fit; python $E/warp2.py > log_warp.log 2>&1 || fail warp ;;
    stack)    printf 'requires 1.2.0\ncd %s\nsetext fit\nset32bits\nsetcpu 14\nstack r2_sky rej w 3 3 -nonorm -rejmap -out=sky\n' "$PW" > stack.ssf
              "$SIRIL" -s "$PW/stack.ssf" > log_stack.log 2>&1; grep -q "Script execution finished successfully" log_stack.log || fail stack ;;
    pad)      python $E/padstack.py > log_pad.log 2>&1 || fail pad ;;
    count)    python $E/countmap.py > log_count.log 2>&1 || fail count ;;
    tone)     python $E/tone_fit.py 60 0.18 > log_tone.log 2>&1 || fail tone ;;
    astap)    python $E/astap_center.py > log_astap.log 2>&1 || fail astap ;;
    annotate) python $E/solve.py > log_solve.log 2>&1 || echo "FAIL annotate (non-fatal)" >> chain.log ;;
    traffic)  python $E/traffic.py > log_traffic.log 2>&1 && python $E/traffic_still.py > log_traffic2.log 2>&1 || echo "FAIL traffic (non-fatal)" >> chain.log ;;
    mood)     python $E/mood.py $MOOD_FRAME 1.0 > log_mood.log 2>&1 || echo "FAIL mood (non-fatal)" >> chain.log ;;
    print)    python $E/print.py > log_print.log 2>&1 || echo "FAIL print (non-fatal)" >> chain.log ;;
    trails)   rm -rf trails_frames; if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then python $E/trails_gpu.py > log_trails.log 2>&1 || fail trails; else python $E/trails.py > log_trails.log 2>&1 || fail trails; fi ;;
    export)   python $E/export_trails.py > log_export.log 2>&1 || echo "FAIL export (non-fatal)" >> chain.log ;;
    timelapse) rm -rf timelapse_locked timelapse_standard timelapse_clouds; python $E/timelapse.py locked standard clouds > log_timelapse.log 2>&1 || echo "FAIL timelapse (non-fatal)" >> chain.log ;;
    encode)   bash $E/encode.sh > log_encode.log 2>&1 ;;
    deliver)  python $E/deliver.py > log_deliver.log 2>&1 || echo "FAIL deliver (non-fatal)" >> chain.log ;;
  esac
done
echo "DONE" >> chain.log
