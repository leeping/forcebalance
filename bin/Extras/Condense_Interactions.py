#!/usr/bin/env python

import os, sys, re
import numpy as np
from collections import defaultdict, OrderedDict

#======================================================================#
#|                                                                    |#
#|        Interaction condensing script for .itp and .top files       |#
#|                                                                    |#
#|                Lee-Ping Wang (leeping@stanford.edu)                |#
#|                    Last updated June 14, 2012                      |#
#|                                                                    |#
#|      This script is intended to be run in conjunction with         |#
#|      amb2gmx_LP.pl, with the .top file as the only argument        |#
#|                                                                    |#
#|      Purpose: To convert a .top file with parameters specified     |#
#|      by _interaction_ to one where parameters are specified by     |#
#|      _interaction type_.  This makes the force field more          |#
#|      readable and eliminates redundancies.                         |#
#|                                                                    |#
#|      I have only tested this on self-contained .top files.         |#
#|      (with defaults, atomtypes, moleculetype, interactions)        |#
#|      Energies from the converted .top file agree to within         |#
#|      1e-5 kJ/mol of the original .top file converted using         |#
#|      amb2gmx.pl                                                    |#
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
## When there are two possible angle minima, angles cannot be defined uniquely using angletypes.
skips = []
## The current section in the ITP file
sec = None
for line in ffdata:
    strip = line.strip()
    line = line.split(';')[0]
    s = line.split()
    if re.match('^ *\[.*\]',line):
        # This regular expression determines which section we are in.
        sec = re.sub('[\[\] \n]','',line)
    if sec == 'atoms' and re.match('^ *[0-9]',line):
        atomtypes.append(s[1])
    elif sec in ['bonds', 'angles', 'dihedrals'] and re.match('^ *[0-9]',line):
        # Determine which atoms are bonded.
        if sec == 'bonds':
            a = int(s[0]) if int(s[0]) < int(s[1]) else int(s[1])
            b = int(s[1]) if int(s[0]) < int(s[1]) else int(s[0])
            bonds[a].append(b)
            bonds[b].append(a)
        # Determine the appropriate dictionary to store parameters in.
        idict, nat = master[sec]
        # Look up atom types for the interaction.
        ats = [atomtypes[int(i)-1] for i in s[:nat]]
        ans = [int(i) for i in s[:nat]]
        # Reverse the ordering of atom types so they're in forward alphabetical order
        if (ats[0] > ats[-1]): 
            ats = ats[::-1]
        elif (ats[0] == ats[-1]) and len(ats) == 4 and (ats[1] > ats[2]):
            ats = ats[::-1]
        # The interaction type; this is always a number
        itype = s[nat]
        # The 'suffix' is only for redundant interaction type 9
        suf = ''
        if sec == 'dihedrals' and itype == '9':
            if all([ans[i+1] in bonds[ans[i]] for i in range(nat-1)]): 
                # Proper dihedrals are defined by whether bonds exist for atoms 1-2-3-4.
                suf = '_'+s[7]
            else:
                # Improper dihedrals are given type 4
                itype = '4'
        key = ','.join(ats+[itype])+suf
        val = ','.join(s[nat+1:])
        # If two interactions have the same atom types but the parameters are different, this script won't work.
        # The exception is the AMBER-style proper dihedral, which can be redundant up to the multiplicity.
        if key in idict and idict[key] != val:
            skips.append(strip)
            sys.stderr.write("Key %s already exists in idict with a different value (%s -> %s)\n" % (key, idict[key], val))
        else:
            idict[key] = val

# print atomtypes
# print bondtypes
# print angletypes
# print dihedraltypes
# print skips
# sys.exit()

## Pass 2.  This time, we loop through the force field file, 
Insert = True
dihe_nodup = []
sec = None
for line in ffdata:
    # Split line by words and keep whitespace for nice formatting.
    strip = line.strip()
    s = line.split()
    w = re.findall('[ ]+',line)
    if re.match('^ *\[.*\]',line.split(';')[0]):
        sec = re.sub('[\[\] \n]','',line.split(';')[0])
    if sec == 'moleculetype' and Insert:
        # Here is where we insert the 'interaction type' sections.
        Insert = False
        print "[ bondtypes ]"
        for key, val in bondtypes.items():
            print ''.join(['%5s' % i for i in key.split(',')])+''.join(['%12s' % i for i in val.split(',')])
        print
        print "[ angletypes ]"
        for key, val in angletypes.items():
            print ''.join(['%5s' % i for i in key.split(',')])+''.join(['%12s' % i for i in val.split(',')])
        print
        print "[ dihedraltypes ]"
        for key, val in dihedraltypes.items():
            print ''.join(['%5s' % i for i in key.split('_')[0].split(',')])+''.join(['%12s' % i for i in val.split(',')])
        print
    if sec in ['bonds', 'angles', 'dihedrals'] and re.match('^ *[0-9]',line):
        idict, nat = master[sec]
        itype = s[nat]
        if sec == 'dihedrals' and itype == '9':
            ans = [int(i) for i in s[:nat]]
            if all([ans[i+1] in bonds[ans[i]] for i in range(nat-1)]): pass
            else: itype = '4'
        if itype == '9':
            # On the second pass through the force field file,
            # we need to skip over duplicate 'interaction type 9' lines.
            # Otherwise the interaction gets double-counted.
            if ans in dihe_nodup: continue
            dihe_nodup.append(ans)
        if strip in skips:
            print line,
        else:
            print ''.join([w[j]+s[j] for j in range(nat)])+w[nat]+itype
    else:
        print line,
