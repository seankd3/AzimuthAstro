cd /d/AstroWork/lake2
until grep -qE "^done|Traceback|Error" convert_py.log 2>/dev/null; do sleep 30; done
grep -q "^done" convert_py.log || { echo "FAIL convert" >> chain.log; exit 1; }
export ASTRO_WORK=D:/AstroWork/lake2 MOOD_FRAME=92
bash /d/AstroWork/engine/run_project.sh ground
