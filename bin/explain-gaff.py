#!/usr/bin/env python

import numpy as np
import sys
import os
from re import match

#==========================================================#
# GAFF Explanation Script, Lee-Ping Wang           05/2012 #
#                                                          #
# The purpose of this script is to explain the atom types  #
# that come from an Antechamber run.  Sometimes the atom   #
# types are 'wrong' and we have to correct them.           #
#                                                          #
# The script works by reading in the antechamber output    #
# mol2 file (it doesn't work on other file types yet.)     #
# It parses the mol2 file for the coordinates / atomtypes  #
# and cross-references them with the gaff.dat data file.   #
#                                                          #
# Then it abbreviates the descriptions and prints them     #
# to a VMD script file along with the atomic coordinates.  #
# When this file is executed in VMD using 'vmd -e .runvmd' #
# it will load the molecule and label all of the atomtypes #
# using the abbrevaited description.                       #
#==========================================================#

try:
    DataFile=os.path.join(os.environ['AMBERHOME'],"dat/leap/parm/gaff.dat")
except:
    raise Exception("To make sure we find the gaff.dat file, please make sure the $AMBERHOME environment variable is set.")
if not os.path.exists(DataFile):
    raise Exception("gaff.dat not found!")
# Lee-Ping's gaff.dat
# DataFile="/home/leeping/opt/amber/dat/leap/parm/gaff.dat"
Atypes = {}

def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, integer, or what have you"""
    return match('^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$',word)

for line in open(DataFile).readlines():
    sline = line.split()
    if len(sline) == 0:
        break
    if isfloat(sline[1]) and isfloat(sline[2]):
        # Following is a bunch of things to reduce the text description size.
        dline = ' '.join(sline[3:])
        dline = dline.replace("identical to","=")
        dline = dline.replace('in pure aromatic systems', 'pure arom')
        dline = dline.replace('in non-pure aromatic systems', 'nonpure arom')
        dline = dline.replace('in conjugated systems', 'conj')
        dline = dline.replace('in triangle systems', 'triangle')
        dline = dline.replace('in square systems', 'square')
        dline = dline.replace('Head Sp2 C that connect two rings in biphenyl sys.', 'bridgehead C')
        dline = dline.replace('H bonded to aliphatic carbon without electrwd. group', 'H-C (aliph) 0 EWG')
        dline = dline.replace('H bonded to aliphatic carbon with 1 electrwd. group', 'H-C (aliph) 1 EWG')
        dline = dline.replace('H bonded to aliphatic carbon with 2 electrwd. group', 'H-C (aliph) 2 EWG')
        dline = dline.replace('H bonded to aliphatic carbon with 3 electrwd. group', 'H-C (aliph) 3 EWG')
        dline = dline.replace('H bonded to C next to positively charged group', 'H-C (positive), 3 EWG')
        dline = dline.replace('H bonded to non-sp3 carbon with', 'H-C (non-sp3),')
        dline = dline.replace('H bonded to aromatic carbon', 'H-C (arom),')
        dline = dline.replace('H bonded to nitrogen atoms', 'H-N')
        dline = dline.replace('with two connected atoms', '2 bonds')
        dline = dline.replace('with three connected atoms', '3 bonds')
        dline = dline.replace('with four connected atoms', '4 bonds')
        dline = dline.replace('carbons', 'C')
        dline = dline.replace('carbon ', 'C ')
        Atypes[sline[0]] = '\ '.join(dline.split())

Descriptions = []
XYZs = []

if len(sys.argv) != 2:
    print "Usage: %s molecule.mol2" % __file__
    print "Exiting..."
    sys.exit()

for line in os.popen("awk '/ATOM/,/BOND/' %s" % sys.argv[1]):
    sline = line.split()
    if len(sline) == 9:
        try:
            Descriptions.append("\ \ \ \[%s\]\ " % sline[5] + Atypes[sline[5]])
        except:
            raise KeyError("The atom type %s is not in gaff.dat" % sline[5])
        XYZs.append([float(i) for i in sline[2:5]])

#graphics top text {23.4840 -7.0510 10.1000} 'Sniffy'

vmdfnm = '.runvmd'
OutVMD = open(vmdfnm,'w')

header = """
#==================================================#
#           Global settings for display            #
#==================================================#
axes location Off
display rendermode GLSL
color Display Background white
display projection Orthographic
display depthcue off
display nearclip set 0.010000
material change opacity Ghost 0.000000

# Load the new molecule
mol new {xyzname} type mol2 waitfor all

mol modstyle 0 0 CPK 1.000000 0.300000 25.000000 25.000000
"""

print >> OutVMD, header.format(xyzname=sys.argv[1])

for xyz, desc in zip(XYZs, Descriptions):
    print >> OutVMD, "graphics top text {% .3f % .3f % .3f} %s size 0.4" % (xyz[0],xyz[1],xyz[2],desc)

OutVMD.close()

RunVMD = False
# Now Actually run VMD.
if RunVMD:
    os.system('vmd -e %s' % vmdfnm)
else:
    print "Finished analyzing mol2 file.  Now run vmd with the following command: vmd -e %s" % vmdfnm
