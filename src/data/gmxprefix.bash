#!/bin/bash

# This wrapper script is for running GROMACS jobs on clusters.

COMMAND=$@

# Backup folder
export BAK=$HOME/temp/rungmx-backups

# Disable GROMACS backup files
export GMX_MAXBACKUP=-1

echo
echo "#=======================#"
echo "# STARTING CALCULATION! #"
echo "#=======================#"
echo
echo $@

rm -f md_result.p md_result.p.bz2
export PYTHONPATH=~/research/forcebalance/lib/python2.7/site-packages/
time $COMMAND
exitstat=$?
# Delete backup files that are older than one week.
find $BAK/$PWD -type f -mtime +7 -exec rm {} \;
mkdir -p $BAK/$PWD
cp * $BAK/$PWD
# For some reason I was still getting error messages about the bzip already existing.
rm -f md_result.p.bz2
#bzip2 md_result.p

exit $exitstat
