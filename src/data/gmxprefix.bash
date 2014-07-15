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

# Remove the result file (so the result from a previous calculation
# doesn't get accidentally sent back)
rm -f md_result.p
export PYTHONPATH=~/research/forcebalance/lib/python2.7/site-packages/
time $COMMAND
exitstat=$?
# Delete backup files that are older than one week.
find $BAK/$PWD -type f -mtime +7 -exec rm {} \;
mkdir -p $BAK/$PWD
cp * $BAK/$PWD

exit $exitstat
