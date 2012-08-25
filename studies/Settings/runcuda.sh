#!/bin/bash

# This wrapper script is for running CUDA jobs (hint hint, OpenMM) on clusters.

# Load my environment variables. :)
. ~/.bashrc
# Make sure the Cuda environment is turned on
if [[ $HOSTNAME =~ "leeping" ]] ; then
    export CUDA_HOME=/opt/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
    export INCLUDE=$CUDA_HOME/include:$INCLUDE
elif [[ $HOSTNAME =~ "fire" ]] ; then
    module load cuda/4.1-experimental
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
elif [[ $HOSTNAME =~ "icme-gpu" || $HOSTNAME =~ "node0" ]] ; then
    module load gcc/4.4.6
    module load cuda41/toolkit/4.1.28
elif [[ $HOSTNAME =~ "ls4" || $HOSTNAME =~ "longhorn" ]] ; then
    module unload intel
    module load gcc
    module load cuda
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

time $@

# Avoid the stupid segfault-on-quit that happens on fire
# Ahh, i don't know how to do this..

if [ $? -gt 30000 ] ; then
    exit 0
fi
