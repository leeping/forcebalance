#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J forcebal
#BSUB -W 12:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:

. ~/.bashrc

# Use the right conda environment
conda activate forcebalance

# Start the estimation server.
./run_server.py &> server_console_output.log &
echo $! > save_pid.txt

sleep 30

# Run ForceBalance
ForceBalance.py optimize.in &> force_balance.log

# Kill the server
kill -9 `cat save_pid.txt`
rm save_pid.txt