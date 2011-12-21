#!/bin/bash

ln -fs ../../bin/* .
ln -fs ../../forcebalance/* .
./CallGraph.py | dot -Tpng > ../CallGraph.png
