#!/bin/bash

while [ $# -gt 0 ]
do
    case $1 in
        -c | --cache) shift; cache=$1 ;;
        *) break ;;
    esac
    shift
done

. /etc/profile
. $HOME/.bashrc

export PATH=/opt/python/2.7.2/bin:$PATH
export LD_LIBRARY_PATH=/opt/python/2.7.2/lib:$LD_LIBRARY_PATH
module load psi/tip.opt

echo "#=======================#"
echo "# ENVIRONMENT VARIABLES #"
echo "#=======================#"
echo

env

echo
echo "#=======================#"
echo "# STARTING CALCULATION! #"
echo "#=======================#"
echo
echo $@

export PSISCRATCH=$PWD

for mol in $(awk '/molecule/ {print $2}' objective.dat) ; do
    if [ ! -f $PSISCRATCH/$mol.dat ] ; then
        if [ -f $cache/$mol.dat ] ; then
            cp $cache/$mol.dat .
        else
            echo "Running psi4 build.dat since $PSISCRATCH/$mol.dat does not exist"
            psi4 build.dat
        fi 
    fi
done 

#if [ ! -f $PSISCRATCH/$1.dat ] ; then
#    if [ -f $cache/$1.dat ] ; then
#        cp $cache/$1.dat .
#    else 
#        echo "Running psi4 build.dat since $PSISCRATCH/$1.dat does not exist"
#        psi4 build.dat
#    fi
#fi

echo "--> objective.dat <--"
cat objective.dat

psi4 objective.dat
