#!/bin/bash

echo "Distance  Energy-S2B  Energy-S2C  Energy-S2E"
paste -d " " <(awk '/LABEL/ {printf "%8.1f ", $2} /INTERACTION/ {printf "% 11.4e\n", $2*627.51}' S2BPose/qdata.txt) <(awk '/INTERACTION/ {printf "% 11.4e\n", $2*627.51}' S2BPose/qdata.txt) <(awk '/INTERACTION/ {printf "% 11.4e\n", $2*627.51}' S2BPose/qdata.txt)
