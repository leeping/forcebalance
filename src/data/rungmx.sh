#!/bin/bash

# This wrapper script is for running GROMACS jobs on clusters.

COMMAND=$@

# Load my environment variables. :)
. ~/.bashrc

# Backup folder
export BAK=$HOME/temp/rungmx-backups

# Disable GROMACS backup files
export GMX_MAXBACKUP=-1

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
echo $COMMAND

rm -f npt_result.p npt_result.p.bz2
export PYTHONUNBUFFERED="y"
# Unset OMP_NUM_THREADS otherwise gromacs will complain.
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
time $COMMAND
# Delete backup files that are older than one week.
find $BAK/$PWD -type f -mtime +7 -exec rm {} \;
mkdir -p $BAK/$PWD
cp * $BAK/$PWD
# For some reason I was still getting error messages about the bzip already existing..
rm -f npt_result.p.bz2
bzip2 npt_result.p

