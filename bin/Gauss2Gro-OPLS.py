#!/usr/bin/env python

from numpy import *
import os
from sys import argv,exit
import forcebalance
from forcebalance.PT import PeriodicTable
from forcebalance.nifty import isint, printcool
import getopt

opts,args = getopt.getopt(argv[1:],'h')

AutoH = False
for o,a in opts:
    argv.remove(o)
    if o == "-h":
        AutoH = True

Alphabet = list(map(chr, range(65, 91)))

printcool(" Welcome to Lee-Ping's Force Field Generator --- ")
print " This script attempts to generate an OPLS force field "
print " from any Gaussian input file.  Here's how it works.  "
print " 1.  Define your atom types.                          "
print " 2.  Define corresponding OPLS atom types.            "
print "     If there is no corresponding OPLS type, then you "
print "     must define your own."
print " 3.  Select or input bonding parameters.              "
print "     If you selected OPLS atom types, then the program"
print "     will attempt to recommend parameters for you.    "
print " 4.  Finished: GRO and ITP files will be generated.   "

"""
gauss2gro-OPLS.py is an automatic force field maker, which builds a
force field from a molecular topology (in Gaussian .com format) and
fills in parameters automatically from OPLS-AA.  This is probably the
most complicated script I've written next to ForceBalance.  To avoid
confusion, this script DOES NOT do parameter optimization, it just
generates a force field from the OPLS parameter libraries for pretty
much any molecule you want.  You still have to fill in parameters by
hand for any gaps in OPLS (which are plenty).
 
Here's how it works.

As an input, we must have a Gaussian .com file with connectivity.
Usually, it's easiest to open a .com file with GaussView and resave
it, making sure that geom=connectivity is in the first line.

To enable automated atom-type definition (very helpful if you want to
follow OPLS 'standards',) you need to do either of the following:

OPTION 1 (recommended)
In the Gaussian .com file, after every atom name, type in '!opls_xxx' 
where opls_xxx is the standard opls atomtype 
(look it up using 'explain-opls.py' C for carbon).

OPTION 2:
In the Gaussian .com file, after every atom name, type in '!atom_type' 
where atom_type is an atom type of your choosing.
Somewhere else in the Gaussian file (preferably after the first line
and before the xyz coordinates), type in '! atom_type opls_xxx' to
identify that atom type with the OPLS parameters.

When the program is run, the first thing it does is try to get the
OPLS atom types.  You can assign atom types using one of the two
options above, in which case it will get the whole list automatically.
Otherwise, you can define atom types by element, or more specifically
by element + nearest neighbors.  In the latter case, you will be asked
to pick OPLS atom types manually, or enter your own nonbonded
parameters.  Picking OPLS atom types has the advantage that the bonded
parameters will be automatically chosen for you.

Once the atom types are chosen, the program will sum up all of the
atomic charges.  Because a simulation generally likes to have an
integer (usually zero) total charge, you will be given the option to
add a constant charge to every atom in order to make the overall
molecule neutral.  Finally, there is a fine-tuning step which allows
you to add small amounts to the charge so that it's _exactly_ zero,
despite the finite floating-point precision of the force field file.

The next thing the program does is to automatically parameterize all
of the detected bonds, angles, and dihedral types.  To do this, there
is a rather complicated set of lists.  Basically, in GROMACS there
each atom has several types associated with it: the 'element' (e.g. C
= 6), the 'base type', the 'OPLS type', the 'bonded type', and the
'OPLS nickname' defined in ffoplsaanb.itp.  In orthodox OPLS, the base
type starts with 'opls_', and is the same as the OPLS type; the bonded
atom type (e.g. 'CA', 'C_2' etc). and the OPLS nickname are the same.
In custom force fields, the base type, and the bonded atom type are
the same, but the OPLS type and the OPLS nickname are different.  The
OPLS type and nickname are essential for pulling the bonded parameters
out of the ffoplsaabon.itp file.  Thus, all four lists are important.

awkx gives a list of atoms and Cartesian coordinates, much like the
data in the xyz file.  awkc gives a list of the bonds at the end of
the Gaussian .com file.  awkpd gives a list of OPLS atom types which
correspond to the custom atom types, useful only in OPTION 2 above.
awkat gives a N-length list of cutom atom types or OPLS atom types,
depending on what the user has defined.

Below are explanations of names of variables.

oplsatoms, oplsnb, oplsbon are OPLS force field files, which are
essential to the working of this script.  They MUST point to the
locations of these files for automatic parameterization.
OPLSExplanations is a dictionary of strings from ffoplsaa.atp of 'OPLS
type':
E is a N-length list of elements.
R is a 3N-length array of Cartesian coordinates.
T is a N-length list of base types.
sbonds is a N-length list of lists, where the i-th entry contains all
 of the atoms j bonded to atom i.  It's the same as abonds (from the
 Gaussian file), except that bonds are bidirectional (i.e. the j-h
 entry still lists i as a bonded atom, even if j>i).
M is a dictionary of 'base type' : [atomic number, atomic mass, sigma,
 epsilon].  As you can see, each base type has its own nonbonded
 parameters.
BAT is a dictionary of 'base type': 'bonded type'
TEDict is a dictionary of 'base type U bonded type': 'element'
OPLSNicks is a dictionary of 'bonded type': 'OPLS nickname'
BList, AList, and DList are lists of interactions, which are built by
 searching the topology.  They are printed out in the later part of the
 force field file which lists all of the specific bonded interactions.
BTypeList, ATypeList, and DTypeList are lists of interaction types,
 which are built by searching the topology and enumerating the unique
 combinations of bonds, angles, and dihedrals (in terms of the OPLS
 nicknames).  They enter into the DefineParameters function, which
 returns dictionaries of interaction types ready for printing.
BondTypes, AngleTypes, and DihedralTypes are dictionaries of
 'bonded_type_1.bonded_type_2...' : [InteractionClass,[Bonded
 Parameters]].  These dictionaries are all created in the
 'DefineParameters' function, all enter into the 'PrintTypeSection'
 function as the 'SectionDict' argument.  OPLS nicknames are separated
 with periods, because OPLS nicknames themselves do not contain
 periods.  For example, a key in DihedralTypes would look like
 'CT.CA_2.CB.C!' and the value would look like [[3],[0.0 1.0 2.0 3.0
 4.0 5.0]].
AtomTypeCount is a dictionary of 'base type' : 'number of
 occurrences', which is used for zeroing out the net charge.
QDict is a dictionary of 'base type': 'atomic charge'.  Note that the
 atomic charge is printed out in the atoms section only; currently I
 choose to not print out any charges in the atomtypes section, as they
 are defined in the atoms section anyway..
predict and preatom are used for generating pre-defined atom types.
TDict is only used for generating neighbor-based atom types.

As of October 2012, this script is a part of ForceBalance.

All code in this repository is released under the GNU General Public License.

#===========#
#| License |#
#===========#

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness for a
particular purpose.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Gets the molecule name
molname = argv[1][:3].upper()
molnametemp = raw_input("Enter molecule name (three letters, default %s) >> " % molname)
if len(molnametemp) == 3:
    molname = molnametemp
else:
    print "Going with the default name (%s)" % molname

# ENHANCED, SUPER-SPECIFIC FILTERS!!!
awkx = "awk -F '!' '{print $1}' %s | awk \'(NF==4 && $1*1!=$1 && $2*1==$2 && $3*1==$3 && $4*1==$4){print}\' " % argv[1]
awkc = "awk -F '!' '{print $1}' %s | awk \'($1==sprintf(\"%%i\",$1) && NF%%2==1){p=1;for(i=1;i<=NF;i++){if(i%%2==0){if($i!=sprintf(\"%%i\",$i)){p=0}}else{if($i!=$i*1){p=0}}}if(p==1){print}}\'" % argv[1]
awkpd = "grep '!' %s | awk '/ATYPE/ {print $(NF-1),$NF}'" % argv[1]
awkat = "awk \'(NF>=4 && $1*1!=$1 && $2*1==$2 && $3*1==$3 && $4*1==$4){print}\' %s | awk -F '!' '{if (NF>1) {print $NF}}' | awk '{print $1}'" % argv[1]

# Locations of files
datadir = os.path.split(forcebalance.__file__)[0]
oplsatoms = os.path.join(datadir,'data','oplsaa.ff','atomtypes.atp')
oplsnb = os.path.join(datadir,'data','oplsaa.ff','ffnonbonded.itp')
oplsbon = os.path.join(datadir,'data','oplsaa.ff','ffbonded.itp')

OPLSExplanations = {}
for line in os.popen("awk '$1 !~ /;/' %s" % oplsatoms).readlines():
    doc = line.split(";")[1].strip()
    OPLSExplanations[line.split()[0]] = doc

# Gets elements and cartesian coordinates
E = [l.split()[0] for l in os.popen(awkx)] # Element list
R = array([[float(i) for i in l.split()[1:]] for l in os.popen(awkx)])/10 # Cartesian coordinate list

# Gets list of bonds
abonds = [sorted([int(i)-1 for i in l.split()[1::2]]) for l in os.popen(awkc)]

# Gets lists of pre-defined atom types
predict = {}
for l in os.popen(awkpd):
    predict[l.split()[0]] = l.split()[1]
preatom = [l.strip() for l in os.popen(awkat)]
for l in preatom:
    if "opls_" in l:
        predict[l] = l

# Initialize variables
sbonds = []
T = []
BAT = {}
TDict = {} # BONDS: TYPE
TEDict = {} # TYPE: ELEMENT
na = len(abonds)
M = {}
OPLSNicks = {}
AtomTypeCount = {}
BList = [] # List of Bonds
AList = [] # List of Angles
DList = [] # List of Dihedrals
BTypeList = []
ATypeList = []
DTypeList = []
QDict = {} # List of Charges

# Alternate angle parameters that kick in if the angle exceeds ThreshAngle.  This part is ugly and might need rewriting.
altangles = {"O.Co.O":[[1],[1.4661e+02,5.4081e+01]],
             "O.Co.OA":[[1],[1.4661e+02,5.4081e+01]],
             "O.Co.OB":[[1],[1.3392e+02,5.1575e+01]],
             "O.Co.OC":[[1],[1.4661e+02,5.4081e+01]],
             "OA.Co.OA":[[1],[1.4661e+02,5.4081e+01]],
             "OA.Co.OB":[[1],[1.4661e+02,5.4081e+01]],
             "OA.Co.OC":[[1],[1.4661e+02,5.4081e+01]],
             "OB.Co.OB":[[1],[1.4661e+02,5.4081e+01]],
             "OB.Co.OC":[[1],[1.4661e+02,5.4081e+01]],
             "OC.Co.OC":[[1],[1.4661e+02,5.4081e+01]],
             "Co.OB.Co":[[1],[1.4661e+02,1.0312e+03]],
             "Co.OC.Co":[[1],[1.4661e+02,1.0312e+03]]}
ThreshAngle = 150

### FUNCTION DEFINITIONS ###

# In: An atom number and all of the atoms bonded to it.
# Out: All of the angle interactions with this atom at the center.
def acomb(a,set):
    c = []
    for i in set:
        for j in set:
            if j>i:
                c.append([i,a,j])
    return c

# In: Atom number A, atom number B, the atoms bonded to atom A, the atoms bonded to atom B.
# Out: All of the dihedral interactions for atom A and atom B.
def dcomb(a,b,seta,setb):
    d = []
    for i in seta:
        for j in setb:
            if i != b and j != a:
                if j>i:
                    d.append([i,a,b,j])
                elif j<i:
                    d.append([j,b,a,i])
    return d

# In: Atom number A, atom number B
# Out: The Cartesian distance between the two atoms in nanometers.
def dist(a,b):
    return dot(R[b]-R[a],R[b]-R[a])**0.5

# In: Three atom numbers
# Out: The angle ABC in degrees.
def angle(a,b,c):
    ab = R[a]-R[b];
    cb = R[c]-R[b];
    theta=arccos(dot(ab,cb)/(dot(ab,ab)**0.5*dot(cb,cb)**0.5))*180/pi
    return theta

# Recursive ring finder
# This is a cool function but we currently don't use it for anything
def ring(size,STEP,NOW,PREV):
    FWD = [i for i in sbonds[NOW]]
    for i in PREV[1:]:
        if i in FWD:
            FWD.remove(i)
    if STEP <= size:
        for i in FWD:
            ring(size,STEP+1,i,PREV+[NOW])
    if STEP == size+1 and PREV[-1] == PREV[0]:
        if sorted(PREV[:-1]) not in ring.rsort:
            ring.ratoms.append(PREV[:-1])
            ring.rsort.append(sorted(PREV[:-1]))

# Recursive base type definition
# The initial call is with atomtype(elem,elem)
# Afterward, it goes through the list of base types (CA, CB, CC, CD...)
# until it finds an unused name and returns it
def atomtype(E, A):
    if A in T:
        if A == E:
            A += "A"
            return atomtype(E, A)
        else:
            ANew = A[:-1] + Alphabet[Alphabet.index(A[-1])+1]
            return atomtype(E, ANew)
    else:
        return A

# Used in neighbor-based base type definition
# Returns a unique "key" corresponding to all of the bonded atoms of atom A.
def BuildAKey(i):
    AllNeighbors = [E[j] for j in sbonds[i]]
    AllNeighbors.sort()
    AKey = E[i] + "." + ".".join(AllNeighbors)
    return AllNeighbors, AKey

# This simply prints out the atom number, element, neighbors, and base type of a given atom.
def PrintInfo(i):
    AllNeighbors,AKey = BuildAKey(i)
    print "%9i%10s%16s%15s" % (i+1,E[i]," ".join(AllNeighbors),T[i])

# NOTE: This function is only called by DefineParameters
# Input: Section name ('bondtypes','angletypes','dihedraltypes'), interaction type ('CA.CB.CT')
# Output: Filtered OPLS-AA parameters.
# This goes through the ffoplsaabon.itp file and filters out relevant lines for parameter selection.
# A special feature is "recommendations": If the OPLS nicknames for the input interaction type matches the nicknames on a given line,
# that line is given "points" by the number of matches, denoted by a red carat ^.  
# Note: If we're working on a dihedral interaction, then the two middle atoms get three points while the edge atoms get one point.
# If all of the nicknames match, then that line gets ten points (otherwise impossible).
# Also: "X" is a wild card for dihedral interactions; it counts as a match but doesn't qualify for ten points.
def OPLS_Filter(SectionName,Type):
    ElementList = [TEDict[i] for i in Type.split(".")]
    NickList = [OPLSNicks[i] for i in Type.split(".")]
    rawfile = os.popen("awk '/%s/,(NF==0)' %s" % (SectionName,oplsbon)).readlines()
    filtered = []
    original = []
    already = []
    elemrev = ElementList[::-1]
    typerev = NickList[::-1]
    for line in rawfile:
        sline = line.split()
        try:
            if len(NickList) == 4:
                score = max(sum(array([NickList[i] == sline[i] for i in range(len(NickList))])*array([1,3,3,1])),sum(array([typerev[i] == sline[i] for i in range(len(NickList))])*array([1,3,3,1]))) + line.count(" X ")
            else:
                score = max(sum(array([NickList[i] == sline[i] for i in range(len(NickList))])),sum(array([typerev[i] == sline[i] for i in range(len(NickList))]))) + line.count(" X ")
            if line in already: continue
            already.append(line)
            if array([NickList[i] == sline[i] for i in range(len(NickList))]).all():
                filtered.append(line.replace("\n","")+"\x1b[91m^^^^^^^^^^\x1b[0m\n")
                original.append(line)
            elif array([typerev[i] == sline[i] for i in range(len(NickList))]).all():
                filtered.append(line.replace("\n","")+"\x1b[91m^^^^^^^^^^\x1b[0m\n")
                original.append(line)
            elif array([ElementList[i] in sline[i][:len(ElementList[i])] or sline[i] == "X" for i in range(len(ElementList))]).all():
                if score > 0:
                    filtered.append(line.replace("\n","")+"\x1b[91m%s\x1b[0m\n" % (''.join(["^" for i in range(score)])))
                    original.append(line)
                else:
                    filtered.append(line)
                    original.append(line)
            elif array([elemrev[i] in sline[i][:len(elemrev[i])] or sline[i] == "X" for i in range(len(elemrev))]).all():
                if score > 0:
                    filtered.append(line.replace("\n","")+"\x1b[91m%s\x1b[0m\n" % (''.join(["^" for i in range(score)])))
                    original.append(line)
                else:
                    filtered.append(line)
                    original.append(line)
        except: pass
    return filtered, original

# Default interaction for when there are no input parameters.
# Usually returns all zeros, but also gives default improper
# dihedrals for aromatic rings.
def getdefaults(Type,SectionName,Positions):
    Answer = zeros(len(Positions))
    Comment = 'Default: all zeros'
    if SectionName == 'dihedral':
        Answer[0] = 30.33400
        Answer[2] = -30.33400
        Comment = 'Default: aromatic ring'
    return Answer, Comment

# Takes in the text from OPLS_Filter and gives back recommendations.
# This is very simple; it just gives back the line with the highest number of points.
# If the automatic switch is turned on, then it will choose this line automatically.
# If there is no line with ten points, then the answer is ambiguous;
# it will then prompt the user to choose (the user isn't restricted to choosing
# the most recommended one.)
def recommend(textlist,manual):
    goodlines = []
    currcount = 0
    for line in textlist:
        starcount = line.count("^")
        if starcount > currcount:
            currcount = starcount
            goodlines = [textlist.index(line)]
        elif starcount == currcount:
            goodlines.append(textlist.index(line))
    if len(goodlines) == 1 and currcount == 10:
        return manual and 1 or 0, goodlines
    elif len(goodlines) >= 1:
        return 1, goodlines
    else:
        return 1, goodlines

# In: A list of atom types separated by periods, as for an interaction type
# Out: The ordered list of atom types
# This function takes care to swap atom types in dihedrals where A and D are the same,
# but B and C are not necessarily the same.
def Order(string):
    ssd = string.split(".")
    if ssd[0] > ssd[-1]:
        newlist = ssd[::-1]
    else:
        if len(ssd) == 4:
            if ssd[0] == ssd[3] and ssd[1] > ssd[2]:
                newlist = [ssd[0],ssd[2],ssd[1],ssd[3]]
            else:
                newlist = ssd
        else:
            newlist = ssd
    return ".".join(newlist)

# In: SectionName (bondtypes, angletypes, dihedraltypes) and SectionDict (from DefineParameters)
# and prints out a formatted block of parameters, either to the itp file or to screen.
def PrintTypeSection(SectionName,SectionDict):
    answer = []
    answer.append("[ %s ]" % SectionName)
    for i in sorted(SectionDict):
        line = ""
        for j in i.split("."):
            line += "%5s" % j
        line += "%5i" % SectionDict[i][0]
        for j in SectionDict[i][1]:
            line += "% 14.4e" % j
        if len(SectionDict[i][2]) > 0:
            line += " ; %s" % SectionDict[i][2]
        answer.append(line)
    answer.append("")
    return answer

# In: SectionName, list of 2 atoms (bonds) or 3 atoms (angles)
# Out: Either the distance between 2 atoms or the angles between 3 atoms
def GetBaseVals(SectionName,Atoms):
    if SectionName == "bonds":
        return "%10.5f" % dist(Atoms[0],Atoms[1])
    elif SectionName == "angles":
        return "%12.5f" % angle(Atoms[0],Atoms[1],Atoms[2])
    else:
        return ""

# Prints the 1-4 interactions using the dihedral list.
def PrintPairs(blist,dlist):
    answer = ["[ pairs ]"]
    printed = []
    for i in sorted(dlist):
        if sorted([i[0],i[-1]]) not in blist and sorted([i[0],i[-1]]) not in printed:
            printed.append(sorted([i[0],i[-1]]))
    for i in sorted(printed):
        line = "%5i%5i" % (i[0]+1,i[-1]+1)
        answer.append(line)
    answer.append("")
    return answer

# In: SectionName (bonds, angles, dihedrals), SectionDict (from DefineParameters),
# SectionList (a list of the specific interactions), PrintBaseVals (a switch),
# Sub180 (substitute parameters for large angles)
# In this section, the specific interaction is printed out, along with the
# base value (bond length, angle) and force constant if desired.
def PrintItemSection(SectionName,SectionDict,SectionList,PrintBaseVals,Sub180):
    answer = []
    answer.append("[ %s ]" % SectionName)
    for i in sorted(SectionList):
        line = ""
        for j in i:
            line += "%5i" % (j+1)
        line += "%5i" % SectionDict[Order(".".join([BAT[T[j]] for j in i]))][0]
        if SectionName == "angles" and Sub180 and float(GetBaseVals(SectionName,i)) > ThreshAngle:
            try:
                line += "% 14.4e" % altangles[Order(".".join([BAT[T[j]] for j in i]))][1][-2]
                line += "% 14.4e" % altangles[Order(".".join([BAT[T[j]] for j in i]))][1][-1]
            except:
                print "Sub180 has failed for angle %s" % (".".join(["%i" % j for j in i]))
        elif PrintBaseVals:
            line += GetBaseVals(SectionName,i)
            line += "% 14.4e" % SectionDict[Order(".".join([BAT[T[j]] for j in i]))][1][-1]
        answer.append(line)
    answer.append("")
    return answer

# This is the big lunker!
# In: InteractionList (a list of the specific interaction), TypeList (list of the bonded interaction types 'CA.CB.CT'), 
# InteractionName ('bondtypes','angletypes','dihedraltypes'), InteractionClass (1 for bonds, 1 for angles, 3 for dihedrals),
# Positions (integers corresponding to fields in line where the parameters belong)
# First ask the user whether to go into Automatic Mode
# For each interaction type (e.g. 'CA.CB.CT'), do the following:
# Get a selection of candidate lines from ffoplsaabon.itp and recommend parameters
# Pick the parameter automatically if in automatic mode and an unambiguous choice is made
# Ask the user if there are any decisions to be made
# When we are all done, give back a dictionary with InteractionTypes:Parameters
def DefineParameters(InteractionList,TypeList,InteractionName,SectionName,InteractionClass,Positions):
    SectionDict = {}
    choice = raw_input("%i '%s' interactions of %i types have been found.  Enter 'Yes' to perform automatic parameterization. --> " % (len(InteractionList),InteractionName,len(TypeList)))
    if InteractionName == 'bonds' or InteractionName == 'angles':
        ReadBaseVals = raw_input("Enter 'Yes' to get equilibrium values from source coordinate file (Useful if you did a geometry optimization!) -->") + " "
    if len(choice) > 0:
        if 'y' == choice[0] or 'Y' == choice[0]:
            Manual,ManualNow = 0,0
        else:
            Manual,ManualNow = 1,1
    else:
        Manual,ManualNow = 0,0
    try:
        if 'y' == ReadBaseVals[0] or 'Y' == ReadBaseVals[0] or ReadBaseVals == " ":
            ReadBaseVals = 1
        else:
            ReadBaseVals = 0
    except:
        ReadBaseVals = 0
    for Type in TypeList:
        Good = []
        TypeSplit = Type.split(".")
        Comment = 'OPLS'
        DefaultComment = 'OPLS'
        if ReadBaseVals:
            BaseVals = []
            for item in InteractionList:
                # This complicated line gets a list of base bond-lengths or angles from the atoms that match
                # Note that the data structures are apparent here: OPLSNicks[Bonded_Type[Base_Type[Atom_Number]]]
                if [OPLSNicks[BAT[T[i]]] for i in item] == [OPLSNicks[i] for i in TypeSplit] or [OPLSNicks[BAT[T[i]]] for i in item] == [OPLSNicks[i] for i in TypeSplit[::-1]]:
                    BaseVals.append(float(GetBaseVals(InteractionName,item)))
            BaseVals = array(BaseVals)
        OPLSChoices, RawLines = OPLS_Filter(SectionName,Type)
        ManualNow, Good = recommend(OPLSChoices,Manual)
        if not Manual:
            if ManualNow:
                print "Automatic parameterization failed (but don't worry), going to Manual Selection"
        if not ManualNow:
            Selection = Good[0]
            Parameters = [float(OPLSChoices[Selection].split()[i]) for i in Positions]
            if len(RawLines[Selection].split(';')) > 1:
                DefaultComment = "OPLS " + RawLines[Selection].split(';')[1].strip()
            if ReadBaseVals:
                Parameters[0] = mean(BaseVals)
                print "Automatic parameterization of %s %s using base geometry, std deviation is %.6f" % (Type,InteractionName,std(BaseVals))
                DefaultComment += ', geom from xyz'
            print "We got these parameters:",
            print "".join(["%-6s" % i for i in Type.split(".")]), "%5i" % InteractionClass, "".join(["% 10.5f" % i for i in Parameters])
            print
        else:
            DefaultParams, DefaultComment = getdefaults(Type,InteractionName,Positions) # getdefaults code returns zeros most of the time.
            try:
                try:
                    DefaultParams = [float(OPLSChoices[Good[0]].split()[i]) for i in Positions]
                    if len(RawLines[Good[0]].split(';')) > 1:
                        DefaultComment = "OPLS " + RawLines[Good[0]].split(';')[1].strip()
                except:
                    DefaultParams = [float(OPLSChoices[0].split()[i]) for i in Positions]
                    if len(RawLines[0].split(';')) > 1:
                        DefaultComment = "OPLS " + RawLines[0].split(';')[1].strip()
            except:
                pass
            if ReadBaseVals:
                DefaultParams[0] = mean(BaseVals)
                DefaultComment += ', geom from xyz'
            for line in OPLSChoices:
                print "%5i" % OPLSChoices.index(line),line,
            if len(Good) == 0:
                print " No recommended choices! "
            else:
                print " --- RECOMMENDED CHOICES --- "
                for item in Good:
                    print item, OPLSChoices[item],
            print " --- DEFAULT PARAMETERS  ---"
            print "".join(["%-6s" % i for i in Type.split(".")]), "%5i" % InteractionClass, "".join(["% 10.5f" % i for i in DefaultParams])
            if ReadBaseVals:
                print "Standard deviation of all equilibrium %s of this type is %.6f" % (InteractionName,std(BaseVals))
            line = raw_input("\nFor a '%s' interaction involving %s (OPLS types %s), \nselect line number from above, type your own parameters, \nor hit Enter to accept default. (# of \x1b[91m^\x1b[0m symbols indicate score) --> " % (InteractionName,"-".join(Type.split(".")),"-".join([OPLSNicks[i] for i in Type.split(".")])))
            try:
                Selection = int(line.strip())
                Parameters = [float(OPLSChoices[Selection].split()[i]) for i in Positions]
                print "Going with the following parameter selection: "
                print Selection, OPLSChoices[Selection]
                if len(RawLines[Selection].split(';')) > 1:
                    Comment = "OPLS " + RawLines[Selection].split(';')[1].strip()
            except:
                try:
                    if len(line.split()) == len(Positions):
                        print "Going with user-specified parameters."
                        Parameters = [float(i) for i in line.split()]
                        Comment = 'User Specified Parameter'
                    else:
                        print "No selection or user-specified parameters given, going with the default parameters."
                        Parameters = DefaultParams
                        Comment = DefaultComment
                except:
                    print "No selection or user-specified parameters given, going with the default parameters."
                    Parameters = DefaultParams
                    Comment = DefaultComment
            print
        SectionDict[Type] = [InteractionClass,Parameters,Comment]
    print "--- Finished Selecting %s Parameters ---" % InteractionName
    for line in PrintTypeSection(SectionName,SectionDict):
        print line
    return SectionDict

### END FUNCTION DEFINITIONS ###

### BUILDING CONNECTIVITY MATRIX ###
        
for i in range(na):
    sbvec = [k for k in abonds[i]]
    for j in range(na):
        if i in abonds[j]:
            sbvec.append(j)
    sbonds.append([k for k in sort(sbvec)])

### DEFINING ATOM TYPES ###

if len(preatom) == len(sbonds):
    print "Using pre-defined atom types in Gaussian .com file (syntax Element X Y Z ! AtomType)"
    AtomTypeSwitch = 1
else:
    AtomTypeSwitch = raw_input("Enter '1' for neighbor-based atom type generation or '0' for simple element-based atom types. -->")[0] == '1'

print "\n--- Number --- Element --- Neighbors --- Assigned Type ---\n"
for i in range(len(sbonds)): # [ELEMENT NEIGHBOR1 NEIGHBOR2...]
    AllNeighbors,AKey = BuildAKey(i)
    if len(preatom) == len(sbonds):
        AtomType = preatom[i]
    else:
        try:
            AtomType = TDict[AKey]
        except:
            AtomType = atomtype(E[i],E[i])
            TDict[AKey] = AtomType
    # LPW added a switch to make all carbon-bonded hydrogen "HC" and all oxygen-bonded hydrogen "HO"
    if AtomTypeSwitch:
        T.append(AtomType)
    else:
        T.append(E[i])
    PrintInfo(i)
    TEDict[AtomType] = E[i]

choice = raw_input("Modify the list? (Enter for no) --> ") + " "
print
while 'y' == choice[0] or 'Y' == choice[0]:
    print "\n--- Number --- Element --- Neighbors --- Assigned Type ---\n"
    for i in range(len(T)):
        PrintInfo(i)
    line = raw_input("Enter the index of the atom you wish to change, or enter Q to quit. -->")
    print
    if line == "Q" or line == 'q':
        break
    Index = int(line.strip())
    line = raw_input("Enter the new type of atom %i. --> " % Index)
    print
    Type = line.strip()
    try:
        T[Index-1] = Type
        TEDict[Type] = E[Index-1]
    except:
        choice = raw_input("There was an error.  Do you want to continue? --> ")
        print

raw_input("\nAtom Types Saved. Press Enter")

### SEARCHING FOR RINGS ###

ring.ratoms = []
ring.rsort = []

print "\nFinding Rings...",
for i in range(na):
    ring(6,0,i,[])
    ring(5,0,i,[])
for i in ring.ratoms:
    print "Ring found at",i
print "...Done"

### SELECTING NONBONDED PARAMETERS ###

# This section should probably be its own function, but it's only called once.
# This loop goes through the base atom types and selects the bonded types and OPLS nicknames.
# If the types are not chosen automatically, then the user gets to pick from a list.
print "%i atoms of %i distinct types have been found.  Press Enter." % (len(T),len(set(T)))
raw_input()
for i in sorted(list(set(T))):
    Element = TEDict[i]
    OPLSChoices = list(os.popen("awk '$2 ~ /%s/' %s | awk '(($4 - %.3f) < 0.1 && ($4 - %.3f) > -0.1)'" % (Element,oplsnb,PeriodicTable[Element],PeriodicTable[Element])).readlines())
    ChoicesByName = {}
    for line in OPLSChoices:
        ChoicesByName[line.split()[0]] = line
    try:
        # The contents of this if statement are invoked only if a valid OPLS type is indicated in the Gaussian .com file
        # using either ! ATYPE custom_type opls_xxx or Element x y z ! opls_xxx
        if predict[i] in ChoicesByName:
            print "ATYPE definition found for %s!" % i
            Selection = predict[i]
            SelectedLine = ChoicesByName[Selection]
            sline = SelectedLine.split()
            print "%s -> %10s%5s%5i%10.5f%10.3f%15.5e%15.5e  " % (i, sline[0],sline[1],int(sline[2]),float(sline[3]),float(sline[4]),float(sline[6]),float(sline[7])), OPLSExplanations[sline[0]]
            print
            ANum,Q,SIG,EPS = int(SelectedLine.split()[2]),float(SelectedLine.split()[4]),float(SelectedLine.split()[6]),float(SelectedLine.split()[7])
            COM = "OPLS " + OPLSExplanations[sline[0]]
            # "Orthodox" OPLS; base types are opls_xxx.  In this case, go to new bonded atom types.
            if "opls" in i:
                BAT[i] = SelectedLine.split()[1]#OPLSNicks[i]
                OPLSNicks[BAT[i]] = BAT[i]
            # Base types and bonded types are custom names; don't have to use opls_xxx, but still assign OPLS nicknames
            else:
                BAT[i] = i
                OPLSNicks[BAT[i]] = SelectedLine.split()[1]
            TEDict[BAT[i]] = Element
            M[i] = [ANum,PeriodicTable[Element],SIG,EPS,COM]
            QDict[i] = float("%.4f" % Q)
            continue
    except: pass
    for line in OPLSChoices:
        sline = line.split()
        print "-> %5i <- %10s%5s%5i%10.5f%10.3f%15.5e%15.5e  " % (OPLSChoices.index(line),sline[0],sline[1],int(sline[2]),float(sline[3]),float(sline[4]),float(sline[6]),float(sline[7])), OPLSExplanations[line.split()[0]]
    line = raw_input("For Atomtype %s, please select the corresponding OPLS atomtype from above by NUMBER (sequential) or NAME (opls_xxx), or enter three parameters of your own: Q SIG EPS --> " % i)
    print
    if len(line.split()) == 3:
        try:
            # If the user enters custom parameters, then there is no corresponding OPLS atomtype.
            # In this case, there is no OPLS nickname either, and we will attempt to get one by simply setting it equal to the base type.
            Q,SIG,EPS = [float(j) for j in line.split()]
            COM = 'User Specified Atomtype'
            BAT[i] = i
            OPLSNicks[BAT[i]] = i
            TEDict[BAT[i]] = Element
        except:
            print "An error has occurred in getting nonbonded parameters."
    else:
        try:
            try:
                Selection = int(line.strip())
                SelectedLine = OPLSChoices[Selection]
            except:
                Selection = line.strip()
                SelectedLine = ChoicesByName[Selection]
        except:
            print "An error has occurred in getting nonbonded parameters."
        print Selection,SelectedLine
        ANum,Q,SIG,EPS = int(SelectedLine.split()[2]),float(SelectedLine.split()[4]),float(SelectedLine.split()[6]),float(SelectedLine.split()[7])
        COM = "OPLS " + OPLSExplanations[sline[0]]
        BAT[i] = i
        OPLSNicks[BAT[i]] = SelectedLine.split()[1]
        TEDict[BAT[i]] = Element
    M[i] = [ANum,PeriodicTable[Element],SIG,EPS,COM]
    QDict[i] = float("%.4f" % Q)

# Try to zero out the atomic charge.

print "The following code aims to get an integer-charge molecule using precision 4 floating points."
TQ = sum([QDict[T[i]] for i in range(na)])
print "All atom types have been selected; molecule has a net charge of %.4f" % TQ
Corr1 = (float("%.4f" % (-1*TQ/na)))
choice = raw_input("Create overall integer charge (enter integer), enter your own (enter float), or skip this step (hit Enter)? --> ") + " "
#if 'y' == choice[0] or 'Y' == choice[0]:
if isint(choice.strip()):
    WantQ = int(choice)
    Corr1 = (float("%.4f" % ((WantQ - 1*TQ)/na)))
    for i in QDict:
        QDict[i] += Corr1
    TQ = sum([QDict[T[i]] for i in range(na)])
    print "Adding a charge of %.4f to each atom" % Corr1
    print "Now all atom types have been selected; molecule has a net charge of %.4f" % TQ
else:
    try:
        Corr1 = float(choice)
        for i in QDict:
            QDict[i] += Corr1
        TQ = sum([QDict[T[i]] for i in range(na)])
        print "Now all atom types have been selected; molecule has a net charge of %.4f" % TQ
    except: print "You didn't enter a number, exiting this section"
                
for i in QDict:
    AtomTypeCount[i] = sum(i == array(T))

PrintSwitch = 0
while abs(TQ - int(TQ)) > 1e-5:
    print
    print "Atom Counts:", AtomTypeCount
    print "Total charge: % .4f" %  TQ
    choice = raw_input("Manually add charge to an atom type using: Atype dQ (Q or Enter quits) ? --> ") + " "
    if 'q' == choice[0] or 'Q' == choice[0] or choice == " ": break
    else:
        try:
            QDict[choice.split()[0]] += float(choice.split()[1])
        except: print "The input cannot be parsed"
    TQ = sum([QDict[T[i]] for i in range(na)])

print
print "Charges are GOOD, molecule has a net charge of %.4f" % TQ
print

print "--- FINISHED SELECTING NONBONDED PARAMETERS ---"
for i in M:
    print i, ": ANum = %5i M = %.4f Q = % .3f SIG = %.4f EPS = %.4f" % (M[i][0],M[i][1],QDict[i],M[i][2],M[i][3])

### DETECTION AND PARAMETERIZATION OF BONDS ###

for i in range(na):
    for j in abonds[i]:
        BList.append(sorted([i,j]))
        BType = ".".join(sorted([BAT[T[i]],BAT[T[j]]]))
        if BType not in BTypeList:
            BTypeList.append(BType)

BTypeList.sort()
BondTypes = DefineParameters(BList,BTypeList,'bonds','bondtypes',1,[3,4])

### DETECTION AND PARAMETERIZATION OF ANGLES ###
    
for i in range(na):
    alist = acomb(i,sbonds[i])
    for j in alist:
        AList.append(j)
        acs = sorted([BAT[T[j[0]]],BAT[T[j[2]]]])
        AType = '.'.join([acs[0],BAT[T[j[1]]],acs[1]])
        if AType not in ATypeList:
            ATypeList.append(AType)

ATypeList.sort()
AngleTypes = DefineParameters(AList,ATypeList,'angles','angletypes',1,[4,5])

### DETECTION AND PARAMETERIZATION OF DIHEDRALS ###

for i in range(len(BList)):
    dlist = dcomb(BList[i][0],BList[i][1],sbonds[BList[i][0]],sbonds[BList[i][1]])
    for j in dlist:
        DList.append(j)
        if BAT[T[j[0]]] < BAT[T[j[3]]]:
            DType = '.'.join([BAT[T[j[i]]] for i in range(4)])
        elif BAT[T[j[0]]] > BAT[T[j[3]]]:
            DType = '.'.join([BAT[T[j[i]]] for i in range(3,-1,-1)])
        else:
            if BAT[T[j[1]]] <= BAT[T[j[2]]]:
                DType = '.'.join([BAT[T[j[i]]] for i in range(4)])
            else:
                DType = '.'.join([BAT[T[j[i]]] for i in range(3,-1,-1)])
        if DType not in DTypeList:
            DTypeList.append(DType)

DTypeList.sort()
DihedralTypes = DefineParameters(DList,DTypeList,'dihedral','dihedraltypes',3,[5,6,7,8,9,10])


def printitp():
    itpfile = open(argv[1].replace('.com','.itp'),'w')
    # Prints some header stuff.
    print >> itpfile, '[ defaults ]'
    print >> itpfile, '%-5i%-5i%-10s%5.1f%5.1f' % (1,2,'yes',0.5,0.5) # Defaults
    # Prints atom types and VdW parameters.
    print >> itpfile
    print >> itpfile, '[ atomtypes ]'
    for i in sorted([j for j in M]):
        if i == BAT[i]:
            print >> itpfile, '%-10s%5i%10.4f%10.4f%5s%14.4e%14.4e ; %s' % (i,M[i][0],M[i][1],0.0,'A',M[i][2],M[i][3],M[i][4])
        else:
            Orthodox = 1
            print >> itpfile, '%-10s%5s%5i%10.4f%10.4f%5s%14.4e%14.4e ; %s' % (i,BAT[i],M[i][0],M[i][1],0.0,'A',M[i][2],M[i][3],M[i][4])
    choice = raw_input("Do you want to add atomtypes for a water model?? 0 for SPC(/E), 1 for TIP4P, anything else for No: -->")
    try:
        choice = int(choice)
        if choice == 0:
            print >> itpfile, '%-10s%5s%5i%10.4f%10.4f%5s%14.4e%14.4e ; SPC/E Oxygen' % ("opls_116","OW",8,15.99940,-0.820,"A",3.16557e-01,6.50194e-01)
            print >> itpfile, '%-10s%5s%5i%10.4f%10.4f%5s%14.4e%14.4e ; SPC/E Hydrogen' % ("opls_117","HW",1,1.00800,0.410,"A",0.00000e+00,0.00000e+00)
        elif choice == 1:
            print >> itpfile, '%-10s%5s%5i%10.4f%10.4f%5s%14.4e%14.4e ; TIP4P Oxygen' % ("opls_113","OW",8,15.99940,0.000,"A",3.15365e-01,6.48520e-01)
            print >> itpfile, '%-10s%5s%5i%10.4f%10.4f%5s%14.4e%14.4e ; TIP4P Hydrogen' % ("opls_114","HW",1,1.00800,0.520,"A",0.00000e+00,0.00000e+00)
            print >> itpfile, '%-10s%5s%5i%10.4f%10.4f%5s%14.4e%14.4e ; TIP4P Virtual Site' % ("opls_115","MW",0,0.00000,-1.040,"D",0.00000e+00,0.00000e+00)
        else:
            pass
    except: pass
    print >> itpfile
    # Prints parameter type definitions for bonds, angles, dihedrals.
    for line in PrintTypeSection("bondtypes",BondTypes):
        print >> itpfile, line
    for line in PrintTypeSection("angletypes",AngleTypes):
        print >> itpfile, line
    for line in PrintTypeSection("dihedraltypes",DihedralTypes):
        print >> itpfile, line
    # Prints moleculetype.
    print >> itpfile
    print >> itpfile, '[ moleculetype ]'
    print >> itpfile, '%-10s%5i' % (molname,3) # Number of exclusions
    print >> itpfile
    # Prints atoms.
    print >> itpfile, '[ atoms ]'
    SumCharges = 0.0
    for i in range(na):
        # atom = BAT[T[i]] + "%i" % (i+1)
        atom = E[i] + "%i" % (i+1)
        print >> itpfile, '%5i%10s%5i%10s%7s%5i%10.4f%10.4f' % (i+1,T[i],1,molname,atom,i+1,QDict[T[i]],M[T[i]][1])
        SumCharges += QDict[T[i]]
    if abs(SumCharges) > 1e-6:
        print >> itpfile, "; The total charge of the generated FF is %.4f." % SumCharges
    print >> itpfile
    # Prints atom lists for bonds, angles, dihedrals.
    choice = raw_input("Print structural bond lengths into force field file? -->") + " "
    if 'y' == choice[0] or 'Y' == choice[0]:
        PrintBaseVals = 1
    else:
        PrintBaseVals = 0
    for line in PrintItemSection("bonds",BondTypes,BList,PrintBaseVals,0):
        print >> itpfile, line
    choice = raw_input("Print structural angles into force field file? (Enter no for sub180.) -->") + " "
    Sub180 = 0
    if 'y' == choice[0] or 'Y' == choice[0]:
        PrintBaseVals = 1
    else:
        PrintBaseVals = 0
        choice = raw_input("Try to substitute alternate parameters for linear angles (threshold is 150 degrees)? -->") + " "
        if 'y' == choice[0] or 'Y' == choice[0]:
            Sub180 = 1
    for line in PrintItemSection("angles",AngleTypes,AList,PrintBaseVals,Sub180):
        print >> itpfile, line
    PrintBaseVals = 0
    for line in PrintItemSection("dihedrals",DihedralTypes,DList,0,0):
        print >> itpfile, line
    choice = raw_input("Print 1-4 pairs into force field file? (Seems the only way to make 1-4 interactions work.) -->") + " "
    if 'y' == choice[0] or 'Y' == choice[0]:
        for line in PrintPairs(BList,DList):
            print >> itpfile,line
    print "Parameter file written to %s" % (argv[1].replace('.com','.itp'))
    itpfile.close()

def printgro():
    grofile = open(argv[1].replace('.com','.gro'),'w')
    print >> grofile, 'Generated by gauss2gro: %s' % molname
    print >> grofile, na
    res = "1"+molname
    for i in range(na):
        #atom = BAT[T[i]] + "%i" % (i+1)
        atom = E[i] + "%i" % (i+1)
        print >> grofile, "%8s%7s%5i%12.7f%12.7f%12.7f" % (res,atom,i+1,R[i][0],R[i][1],R[i][2])
    print >> grofile, "   3.00000   3.00000   3.00000"
    print "Coordinate file written to %s" % (argv[1].replace('.com','.gro'))
    grofile.close()
    
def main():
    printitp()
    printgro()
    print "Program Finished Successfully.  Have fun!"

main()
