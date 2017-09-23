#!/bin/bash

# This wrapper script is for running CUDA jobs (hint hint, OpenMM) on clusters.

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

# Make sure the Cuda environment is turned on
# module load cuda
# module load cudatoolkit
# export OPENMM_CUDA_COMPILER=`which nvcc`
export BAK=$HOME/temp/runcuda-backups

if [[ $HOSTNAME =~ "leeping" ]] ; then
    export CUDA_HOME=/opt/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
elif [[ $HOSTNAME =~ "fire" ]] ; then
    module load cuda/6.5
    export OPENMM_CUDA_COMPILER=/opt/CUDA/cuda-6.5/bin/nvcc
    #export CUDA_HOME=/opt/CUDA/4.0
    #export PATH=$CUDA_HOME/bin:$PATH
    #export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    #export INCLUDE=$CUDA_HOME/include:$INCLUDE
elif [[ $HOSTNAME =~ "certainty" || $HOSTNAME =~ "compute-" || $HOSTNAME =~ "largemem-" ]] ; then
    # Keeneland, currently don't have access
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
    export BAK=/scratch/leeping/runcuda-backups
elif [[ $HOSTNAME =~ "kid" ]] ; then
    export CUDA_HOME=/sw/keeneland/cuda/4.1/linux_binary
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
    export BAK=/lustre/medusa/leeping/runcuda-backups
elif [[ $HOSTNAME =~ "icme-gpu" || $HOSTNAME =~ "node0" ]] ; then
    module load cuda55/toolkit
    export OPENMM_CUDA_COMPILER=/cm/shared/apps/cuda55/toolkit/5.5.22/bin/nvcc
    #export OPENMM_CUDA_COMPILER=/cm/shared/apps/cuda50/toolkit/5.0.35/bin/nvcc
elif [[ $HOSTNAME =~ "longhorn" ]] ; then
    module unload intel
    module load gcc
    module load cuda/4.1
    export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/share/apps/cuda/4.1/cuda/bin/nvcc
    export BAK=$SCRATCH/runcuda-backups
elif [[ $HOSTNAME =~ "not0rious" ]] ; then
    module load cuda/5.5
elif [[ $HOSTNAME =~ "ls4" ]] ; then
    module unload intel
    module load gcc
    module load cuda/5.0
    export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/opt/apps/cuda/5.0/bin/nvcc
    export BAK=$SCRATCH/runcuda-backups
elif [[ $HOSTNAME =~ "biox3" ]] ; then
    # biox3 cluster
    export PATH=/usr/local/cuda-5.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-5.0/lib64:/usr/local/cuda-5.0/lib:$LD_LIBRARY_PATH
    export OPENMM_CUDA_COMPILER=/usr/local/cuda-5.0/bin/nvcc
elif [[ $HOSTNAME =~ "cn" ]] ; then
    # HS GPU Cluster
    # 6-29-2013 Seems environment modules don't work for noninteractive shells, strange.
    export PATH=/opt/cuda5.0/bin:$PATH
    export LD_LIBRARY_PATH=/opt/cuda5.0/lib64:$LD_LIBRARY_PATH
    export CUDA_CACHE_PATH=/tmp/leeping/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/opt/cuda5.0/bin/nvcc
    export BAK=/hsgs/nobackup/leeping/scratch/runcuda-backups
elif [[ $HOSTNAME =~ "gpu-" ]] ; then
    # GPU nodes on Sherlock cluster
    export PATH=/usr/local/cuda-5.5/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:$LD_LIBRARY_PATH
    export CUDA_CACHE_PATH=/scratch/users/leeping/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/usr/local/cuda-5.5/bin/nvcc
    export BAK=/scratch/users/leeping/runcuda-backups
elif [[ $HOSTNAME =~ "nid" ]] ; then
    # Blue Waters XK Compute Node
    . /etc/bash.bashrc.local
    module add gcc/4.7.2
    module add cudatoolkit
    export CUDA_HOME=$CRAY_CUDATOOLKIT_DIR
    export OPENMM_PLUGIN_DIR=$HOME/opt/openmm/lib/plugins
    export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
    export LD_LIBRARY_PATH=$HOME/opt/openmm/lib:$OPENMM_PLUGIN_DIR:$LD_LIBRARY_PATH
    export CRAY_CUDA_PROXY=1 
    export BAK=/scratch/sciteam/leeping/runcuda-backups
    numactl --hardware
elif [[ `env | grep -i tacc | wc -l` -gt 0 ]] ; then
    # Currently this is the only way I can be sure I'm on Stampede...
    module load cuda/5.0
    export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=`which nvcc`
    export BAK=$SCRATCH/runcuda-backups
fi

if [[ x$CUDA_DEVICE != x ]] ; then
    sleep $(( CUDA_DEVICE * 10 ))
elif [[ x$PBS_JOBID != x ]] ; then
    sleep $(( PBS_JOBID * 10 ))
fi

echo "#=======================#"
echo "# ENVIRONMENT VARIABLES #"
echo "#=======================#"
echo

env

echo
echo "#=======================#"
echo "#   GPU CONFIGURATION   #"
echo "#=======================#"
echo

echo "I'm using GPU number $CUDA_DEVICE"
echo "nvidia-smi output:"
nvidia-smi
echo "lspci output (to see buses):"
/sbin/lspci -t

echo
echo "#=======================#"
echo "# STARTING CALCULATION! #"
echo "#=======================#"
echo
echo $COMMAND

# rm -f npt_result.p
export PYTHONUNBUFFERED="y"

# Run the actual command.
time $COMMAND
exitstat=$?

# Delete backup files that are older than one week.
find $BAK -type f -mtime +7 -exec rm {} \;

# Copy backup files.
if [ $do_bak -gt 0 ] ; then
    mkdir -p $BAK/$PWD
    cp * $BAK/$PWD
fi

# Avoid the stupid segfault-on-quit that happens on fire
# Ahh, i don't know how to do this..
if [ $? -gt 30000 ] ; then
    exit 0
fi

exit $exitstat
