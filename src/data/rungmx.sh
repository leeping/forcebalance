#!/bin/bash

# This wrapper script is for running GROMACS jobs on clusters.

# Load my environment variables. :)
. ~/.bashrc

# Backup folder
export BAK=$HOME/temp/rungmx-backups

# Disable GROMACS backup files
export GMX_MAXBACKUP=-1

if [[ $HOSTNAME =~ "leeping" ]] ; then
    . /opt/intel/Compiler/11.1/072/bin/iccvars.sh intel64
    export PATH=/home/leeping/opt/gromacs-4.5.5/bin:$PATH
fi

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

rm -f npt_result.p npt_result.p.bz2
export PYTHONUNBUFFERED="y"
time $@
# Delete backup files that are older than one week.
find $BAK/$PWD -type f -mtime +7 -exec rm {} \;
mkdir -p $BAK/$PWD
cp * $BAK/$PWD
# For some reason I was still getting error messages about the bzip already existing..
rm -f npt_result.p.bz2
bzip2 npt_result.p

