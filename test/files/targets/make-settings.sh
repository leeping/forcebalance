#!/bin/bash

for i in `seq 2 14`; do
    j=`printf "%02i" $i`
    cd cluster-$j
    mkdir settings
    cp ../shot.mdp .
    cat <<EOF > topol.top
#include "water.itp"

[ system ]
Clusters of $i water molecules extracted from liquid, solid, and gas phase

[ molecules ]
SOL $i
EOF
    ../modify-gro.py all.gro
    mv new.gro all.gro
    cd ..
done
