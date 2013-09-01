#!/bin/bash

. /etc/profile
. $HOME/.bashrc

export PATH=/opt/python/2.7.2/bin:$PATH
export LD_LIBRARY_PATH=/opt/python/2.7.2/lib:$LD_LIBRARY_PATH
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

#if [ ! -f $PSISCRATCH/$1.dat ] ; then
#    echo "Running psi4 build.dat since $PSISCRATCH/$1.dat does not exist"
#    psi4 build.dat
#fi

echo "--> objective.dat <--"
cat objective.dat

psi4 objective.dat
