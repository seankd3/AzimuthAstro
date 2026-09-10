cd /d/AstroWork/nhlake
until grep -qE "DONE31|FAIL" 31.log 2>/dev/null; do sleep 60; done
# engine numbering: full_<odd id>. Rename descending to avoid collisions (full_00003 -> full_00005 first, etc.)
for i in $(seq 90 -1 1); do n=$((2*i-1)); [ -f "full_$(printf %05d $i).fit" ] && [ $n -ne $i ] && mv "full_$(printf %05d $i).fit" "full_$(printf %05d $n).fit"; done
rm -f full_.seq
export ASTRO_WORK=D:/AstroWork/nhlake MOOD_FRAME=68
bash /d/AstroWork/engine/run_project.sh mask
