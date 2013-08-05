#!/usr/bin/env python

import os, sys, re
import numpy as np
from collections import defaultdict, OrderedDict
from forcebalance.chemistry import *

#======================================================================#
#|                                                                    |#
#|        Interaction augmenting script for .itp and .top files       |#
#|                                                                    |#
#|                Lee-Ping Wang (leeping@stanford.edu)                |#
#|                    Last updated June 14, 2012                      |#
#|                                                                    |#
#|      Purpose: To convert harmonic bonds/angles to Morse+UB         |#
#|                                                                    |#
#======================================================================#

## Read in the force field file and expand all tabs.
ffdata = [line.expandtabs() for line in open(sys.argv[1]).readlines()]
## A dictionary of which atoms are bonded to which.
bonds = defaultdict(list)
## A list of the atom types for each atom.
atomtypes = []
## Dictionaries of (atomtypes_involved,interactiontype : parameters)
bondtypes = OrderedDict()
angletypes = OrderedDict()
dihedraltypes = OrderedDict()
master = {'bonds':(bondtypes, 2), 'angles':(angletypes, 3), 'dihedrals':(dihedraltypes, 4)}
## The current section in the ITP file
sec = None

atelem = {}
atbond = defaultdict(dict)
atang = defaultdict(lambda:defaultdict(dict))
## Pass 1 - obtain atom number -> element dictionary, angle values, bond lengths.
for line in ffdata:
    line = line.split(';')[0]
    s = line.split()
    if re.match('^\[.*\]',line):
        # This regular expression determines which section we are in.
        sec = re.sub('[\[\] \n]','',line)
    if sec == 'atoms' and re.match('^ *[0-9]',line):
        atelem[s[1]] = LookupByMass(float(s[7]))
    elif sec == 'bondtypes' and len(s) >= 5:
        a1 = s[0]
        a2 = s[1]
        b = float(s[3])
        k = float(s[4])
        atbond[a1][a2] = (b,k)
        atbond[a2][a1] = (b,k) 
    elif sec == 'angletypes' and len(s) >= 6:
        a1 = s[0]
        a2 = s[1]
        a3 = s[2]
        b = float(s[4])
        k = float(s[5])
        atang[a1][a2][a3] = (b,k)
        atang[a3][a2][a1] = (b,k)

## Pass 2.  This time, we print stuff out.
sec = None
for line in ffdata:
    # Get rid of comments.
    if len(line.split()) == 0:
        print
    line = line.split(';')[0].replace('\n','')
    # Split line by words and keep whitespace for nice formatting.
    s = line.split()
    w = re.findall('[ ]+',line)
    if re.match('^\[.*\]',line):
        sec = re.sub('[\[\] \n]','',line)
        print line
    elif len(s) == 0:
        pass
    elif sec == 'bondtypes' and len(s) >= 5:
        a1 = s[0]
        a2 = s[1]
        b = float(s[3])
        k = float(s[4])
        BS = BondStrengthByLength(atelem[a1],atelem[a2],b)[0]
        Alpha = (k / (2*BS))**0.5
        print "%5s%5s%5i%15.5e%15.5e%15.5e" % (a1, a2, 3, b, BS, Alpha)
    elif sec == 'angletypes' and len(s) >= 6:
        a1 = s[0]
        a2 = s[1]
        a3 = s[2]
        t = float(s[4])
        k = float(s[5])
        a = atbond[a1][a2][0]
        b = atbond[a2][a3][0]
        C = t * np.pi / 180
        ubb = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(C))
        ubk = 0.0
        print "%5s%5s%5s%5i%15.5e%15.5e%15.5e%15.5e" % (a1, a2, a3, 5, t, k, ubb, ubk) 
    elif sec == 'bonds' and len(s) >= 3:
        idict, nat = master[sec]
        itype = '3'
        print ''.join([w[j]+s[j] for j in range(nat)])+w[nat]+itype
    elif sec == 'angles' and len(s) >= 4:
        idict, nat = master[sec]
        itype = '5'
        print ''.join([w[j]+s[j] for j in range(nat)])+w[nat]+itype
    else:
        print line
