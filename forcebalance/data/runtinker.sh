#!/bin/bash

# This wrapper script is for running TINKER jobs on clusters.

# Command line switch indicates whether to do backups
do_bak=0
while [ $# -gt 0 ]
do
    case $1 in
        -b) do_bak=1 ;;
        *) break ;;
    esac
    shift
done

# This is the command that we want to run.
COMMAND=$@

# Load my environment variables. :)
if [ -f /etc/profile ] ; then . /etc/profile ; fi
if [ -f /etc/bashrc ] ; then . /etc/bashrc ; fi
if [ -f ~/.bash_profile ] ; then . ~/.bash_profile ; fi
if [ -f ~/.bashrc ] ; then . ~/.bashrc ; fi

# Load Gromacs environment variables if needed (e.g. Intel compiler variables)
if [[ $HOSTNAME =~ "sh" ]] ; then
    . /share/sw/licensed/intel-cluster-studio-2013.1.046/composer_xe_2013_sp1.2.144/bin/compilervars.sh intel64
elif [[ $HOSTNAME =~ "biox" || $HOSTNAME =~ "vsp-compute" ]] ; then
    . /home/leeping/opt/intel/bin/compilervars.sh intel64
fi

# Backup folder
export BAK=$HOME/temp/runtinker-backups

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

rm -f npt_result.p
export PYTHONUNBUFFERED="y"

# Actually run the command.
time $COMMAND
exitstat=$?

# Delete backup files that are older than one week.
find $BAK -type f -mtime +7 -exec rm {} \;

# Copy backup files.
if [ $do_bak -gt 0 ] ; then
    mkdir -p $BAK/$PWD
    cp * $BAK/$PWD
fi

exit $exitstat
