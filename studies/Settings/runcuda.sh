#!/bin/bash

# This wrapper script is for running CUDA jobs (hint hint, OpenMM) on clusters.

# Load my environment variables. :)
. ~/.bashrc
# Make sure the Cuda environment is turned on
export BAK=$HOME/temp/runcuda-backups
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
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
elif [[ $HOSTNAME =~ "kid" ]] ; then
    export CUDA_HOME=/sw/keeneland/cuda/4.1/linux_binary
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
    export BAK=/lustre/medusa/leeping/runcuda-backups
elif [[ $HOSTNAME =~ "icme-gpu" || $HOSTNAME =~ "node0" ]] ; then
    module load gcc/4.4.6
    module load cuda41/toolkit/4.1.28
elif [[ $HOSTNAME =~ "longhorn" ]] ; then
    module unload intel
    module load gcc
    module load cuda/4.1
    export OPENMM_CUDA_COMPILER=/opt/apps/cuda/4.1/cuda/bin/nvcc
elif [[ $HOSTNAME =~ "not0rious" ]] ; then
    module load openmm
elif [[ $HOSTNAME =~ "ls4" ]] ; then
    module unload intel
    module load gcc
    module load cuda/5.0
    export OPENMM_CUDA_COMPILER=/opt/apps/cuda/5.0/bin/nvcc
    export BAK=$WORK/runcuda-backups
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

rm -rf npt_result.p
time $@
mkdir -p $BAK/$PWD
cp * $BAK/$PWD

# Avoid the stupid segfault-on-quit that happens on fire
# Ahh, i don't know how to do this..

if [ $? -gt 30000 ] ; then
    exit 0
fi
