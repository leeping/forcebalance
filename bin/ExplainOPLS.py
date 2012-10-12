#!/usr/bin/env python

import os
from sys import argv
import forcebalance
from forcebalance.PT import PeriodicTable

awkx = "awk \'(NF==0){p++}(p==2&&NF==4){print}\' "
awkc = "awk \'(NF==0){p++}(p==3&&NF>0){print}\' "
OPLSExplanations = {}

# Locations of files
datadir = os.path.split(forcebalance.__file__)[0]
oplsatoms = os.path.join(datadir,'data','oplsaa.ff','atomtypes.atp')
oplsnb = os.path.join(datadir,'data','oplsaa.ff','ffnonbonded.itp')
oplsbon = os.path.join(datadir,'data','oplsaa.ff','ffbonded.itp')

Element = "C"

if len(argv) > 1:
    Element = argv[1]

for line in os.popen("awk '$1 !~ /;/' %s" % oplsatoms).readlines():
    doc = line.split(";")[1].strip()
    OPLSExplanations[line.split()[0]] = doc

OPLSChoices = list(os.popen("awk '$2 ~ /%s/' %s | awk '(($4 - %.3f) < 0.1 && ($4 - %.3f) > -0.1)'" % (Element,oplsnb,PeriodicTable[Element],PeriodicTable[Element])).readlines())
for line in OPLSChoices:
    sline = line.split()
    atomname = sline[0]
    print "%10s%5s%10.5f%10.3f%15.5e%15.5e  " % (atomname,sline[1],float(sline[3]),float(sline[4]),float(sline[6]),float(sline[7])), OPLSExplanations[atomname]
