#!/usr/bin/env python

from sys import argv
from numpy import *

file1 = open(argv[1]).readlines()
file2 = open(argv[2]).readlines()
file3 = open(argv[3],'w')

if len(argv) != 4:
    print "Usage: cubediff.py CUBE1 CUBE2 CUBEOUT"

switch = 0
for linenum in range(len(file1)):
    sline1 = file1[linenum].split()
    sline2 = file2[linenum].split()
    try:
        numlist = [float(i) for i in sline1]
        if len(sline1) == 6:
            switch = 1
    except: pass
    if switch == 0:
        print >> file3, file1[linenum],
    elif switch == 1:
        x1 = array([float(i) for i in sline1])
        x2 = array([float(i) for i in sline2])
        dx = x1 - x2
        for i in dx:
            print >> file3, "% .6E" % i,
        print >> file3
    if len(sline1) == 2:
        if sline1[0] == "1" and sline1[1] == "0":
            switch = 1
file3.close()
        
## x1 = 
## x2 = array([float(i) for i in file2[-1].split()])

## dx = x2 - x1

## for i in file1[:-1]:
##     print >> file3,i,

## for i in dx:
##     print >> file3,'%.6e' % i,

