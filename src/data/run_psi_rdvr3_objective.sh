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

if [ ! -f $PSISCRATCH/$1.dat ] ; then
    print "Running psi4 build.dat since $PSISCRATCH/$1.dat does not exist"
    psi4 build.dat
fi
psi4 objective.dat
