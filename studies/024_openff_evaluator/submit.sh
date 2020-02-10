#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J forcebal
#BSUB -W 03:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set cpu options.
#BSUB -n 1 -R "rusage[mem=4]"

. ~/.bashrc

# Use the right conda environment
conda activate forcebalance

# Start the estimation server.
./run_server.py &> server_console_output.log &
echo $! > save_pid.txt

sleep 10

# Run ForceBalance
ForceBalance.py optimize.in &> force_balance.log

# Kill the server
kill -9 `cat save_pid.txt`
rm save_pid.txt