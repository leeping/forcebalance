set terminal pdf size 2,2
set output "energies.pdf"

set title "Dimethyl Sulfide Interaction Energy"
set ylabel "Interaction Energy (kcal/mol)"
set xlabel "Center of Mass Distance (Angstrom)"
set yrange[-6:10]

p "<awk '/LABEL/ {printf \"%.1f \", $2} /INTERACTION/ {print $2}' ../targets/S2BPose/qdata.txt" u 1:($2*627.51) w lp pt 6 lc -1 lw 2 t 'Pose "S2B"', \
"<awk '/LABEL/ {printf \"%.1f \", $2} /INTERACTION/ {print $2}' ../targets/S2CPose/qdata.txt" u 1:($2*627.51) w lp pt 12 lc 2 lw 2 t 'Pose "S2C"', \
"<awk '/LABEL/ {printf \"%.1f \", $2} /INTERACTION/ {print $2}' ../targets/S2EPose/qdata.txt" u 1:($2*627.51) w lp pt 21 lc 3 lw 2 t 'Pose "S2E"'

set xrange[3:7]
set yrange[-5:3]
set title "ForceBalance Fitting to S2E Pose"

p 'optimize-qm.dat' u 1:3 w l lc rgb "red" lw 2 t 'Initial (GAFF) Parameters', \
'optimize-qm.dat' u 1:4 w l lc rgb "orange" lw 2 t '1st Iteration', \
'optimize-qm.dat' u 1:5 w l lc rgb "green" lw 2 t '2nd Iteration', \
'optimize-qm.dat' u 1:6 w l lc rgb "blue" lw 2 t '3rd Iteration (Done)', \
'optimize-qm.dat' u 1:2 w l lc -1 lw 2 t 'Ab initio Reference'
