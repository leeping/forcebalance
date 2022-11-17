#!/bin/bash

# This wrapper script is for running CUDA jobs (hint hint, OpenMM) on clusters.

# Load my environment variables. :)
. ~/.bashrc
# Make sure the Cuda environment is turned on

# module load cuda
# module load cudatoolkit
# export OPENMM_CUDA_COMPILER=`which nvcc`
# export BAK=$HOME/temp/runcuda-backups

if [[ $HOSTNAME =~ "leeping" ]] ; then
    export CUDA_HOME=/opt/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
elif [[ $HOSTNAME =~ "fire" ]] ; then
    module load cuda/4.1-experimental
    export OPENMM_CUDA_COMPILER=/opt/CUDA/cuda4.1/bin/nvcc
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
    # export BAK=/scratch/leeping/runcuda-backups
elif [[ $HOSTNAME =~ "kid" ]] ; then
    export CUDA_HOME=/sw/keeneland/cuda/4.1/linux_binary
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
    # export BAK=/lustre/medusa/leeping/runcuda-backups
elif [[ $HOSTNAME =~ "icme-gpu" || $HOSTNAME =~ "node0" ]] ; then
    module load cuda50/toolkit/5.0.35
    export OPENMM_CUDA_COMPILER=/cm/shared/apps/cuda50/toolkit/5.0.35/bin/nvcc
elif [[ $HOSTNAME =~ "longhorn" ]] ; then
    module unload intel
    module load gcc
    module load cuda/4.1
    export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/share/apps/cuda/4.1/cuda/bin/nvcc
    # export BAK=$SCRATCH/runcuda-backups
elif [[ $HOSTNAME =~ "not0rious" ]] ; then
    module load openmm
elif [[ $HOSTNAME =~ "ls4" ]] ; then
    module unload intel
    module load gcc
    module load cuda/5.0
    export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/opt/apps/cuda/5.0/bin/nvcc
    # export BAK=$SCRATCH/runcuda-backups
elif [[ $HOSTNAME =~ "cn" ]] ; then
    # HS GPU Cluster
    module load cuda
    export CUDA_CACHE_PATH=/hsgs/projects/pande/leeping/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=/opt/cuda5.0/bin/nvcc
    # export BAK=/hsgs/projects/pande/leeping/scratch/runcuda-backups
elif [[ $ARCHIVER == "ranch.tacc.utexas.edu" ]] ; then
    # Currently this is the only way I can be sure I'm on Stampede...
    # This is a wild shot. Kamelasa! It's difficult to tell the cluster name 
    # from the environment variables on a Stampede compute node.
    module load cuda/5.0
    export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
    export OPENMM_CUDA_COMPILER=`which nvcc`
    # export BAK=$SCRATCH/runcuda-backups
fi

echo "#=======================#"
echo "# ENVIRONMENT VARIABLES #"
echo "#=======================#"
echo

set

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
echo $@

rm -f npt_result.p npt_result.p.bz2
export PYTHONUNBUFFERED="y"
time $@
# Delete backup files that are older than one week.
# find $BAK/$PWD -type f -mtime +7 -exec rm {} \;
# mkdir -p $BAK/$PWD
# cp * $BAK/$PWD
# For some reason I was still getting error messages about the bzip already existing..
rm -f npt_result.p.bz2
bzip2 npt_result.p

# Avoid the stupid segfault-on-quit that happens on fire
# Ahh, i don't know how to do this..

if [ $? -gt 30000 ] ; then
    exit 0
fi
