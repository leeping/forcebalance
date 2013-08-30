#!/bin/bash

. /etc/profile
. $HOME/.bashrc

module load psi/tip.opt

echo "#=======================#"
echo "# ENVIRONMENT VARIABLES #"
echo "#=======================#"
echo

set

echo
echo "#=======================#"
echo "# STARTING CALCULATION! #"
echo "#=======================#"
echo
echo $@

# Sara-specific hack.
if [ $USER == "skokkila" ] ; then
    export PSISCRATCH=/u/skokkila
fi

if [ ! -f $PSISCRATCH/$1.dat ] ; then
    echo "Running psi4 build.dat since $PSISCRATCH/$1.dat does not exist"
    psi4 build.dat
fi

cat objective.dat

psi4 objective.dat
