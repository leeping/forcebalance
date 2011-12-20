""" @package gmxio GROMACS input/output.

@todo Even more stuff from forcefield.py needs to go into here.

@author Lee-Ping Wang
@date 12/2011
"""

from re import match
from nifty import isint
from numpy import array

## VdW interaction function types
nftypes = [None, 'VDW', 'VDW_BHAM']
## Pairwise interaction function types
pftypes = [None, 'VPAIR', 'VPAIR_BHAM']
## Bonded interaction function types
bftypes = [None, 'BONDS', 'G96BONDS', 'MORSE']
## Angle interaction function types
aftypes = [None, 'ANGLES', 'G96ANGLES', 'CROSS_BOND_BOND',
           'CROSS_BOND_ANGLE', 'UREY_BRADLEY', 'QANGLES']
## Dihedral interaction function types
dftypes = [None, 'PDIHS', 'IDIHS', 'RBDIHS']

## Section -> Interaction type dictionary.
## Based on the section you're in
## and the integer given on the current line, this looks up the
## 'interaction type' - for example, within bonded interactions
## there are four interaction types: harmonic, G96, Morse, and quartic
## interactions.
fdict = {
    'atomtypes'     : nftypes,
    'nonbond_params': pftypes,
    'bonds'         : bftypes,
    'bondtypes'     : bftypes,
    'angles'        : aftypes,
    'angletypes'    : aftypes,
    'dihedrals'     : dftypes,
    'dihedraltypes' : dftypes,
    'virtual_sites2': ['NONE','VSITE2'],
    'virtual_sites3': ['NONE','VSITE3','VSITE3FD','VSITE3FAD','VSITE3OUT'],
    'virtual_sites4': ['NONE','VSITE4FD']
    }

## Interaction type -> Parameter Dictionary.
## A list of supported GROMACS interaction types in force matching.
## The keys in this dictionary (e.g. 'BONDS','ANGLES') are values
## in the interaction type dictionary.  As the program loops through
## the force field file, it first looks up the interaction types in
## 'fdict' and then goes here to do the parameter lookup by field.
## @todo This needs to become more flexible because the parameter isn't
## always in the same field.  Still need to figure out how to do this.
## @todo How about making the PDIHS less ugly?
pdict = {'BONDS':{3:'B', 4:'K'},
         'G96BONDS':{},
         'MORSE':{3:'B', 4:'C', 5:'E'},
         'ANGLES':{4:'B', 5:'K'},
         'G96ANGLES':{},
         'CROSS_BOND_BOND':{4:'1', 5:'2', 6:'K'},
         'CROSS_BOND_ANGLE':{4:'1', 5:'2', 6:'3', 7:'K'},
         'QANGLES':{4:'B', 5:'K0', 6:'K1', 7:'K2', 8:'K3', 9:'K4'},
         'UREY_BRADLEY':{4:'T', 5:'K1', 6:'B', 7:'K2'},
         'PDIHS1':{5:'B', 6:'K'},
         'PDIHS2':{5:'B', 6:'K'},
         'PDIHS3':{5:'B', 6:'K'},
         'PDIHS4':{5:'B', 6:'K'},
         'PDIHS5':{5:'B', 6:'K'},
         'PDIHS6':{5:'B', 6:'K'},
         'IDIHS':{5:'B', 6:'K'},
         'VDW':{4:'S', 5:'T'},
         'VPAIR':{3:'S', 4:'T'},
         'COUL':{6:''},
         'RBDIHS':{6:'K1', 7:'K2', 8:'K3', 9:'K4', 10:'K5'},
         'VDW_BHAM':{4:'A', 5:'B', 6:'C'},
         'VPAIR_BHAM':{3:'A', 4:'B', 5:'C'},
         'QTPIE':{1:'C', 2:'H', 3:'A'},
         'VSITE2':{4:'A'},
         'VSITE3':{5:'A',6:'B'},
         'VSITE3FD':{5:'A',6:'D'},
         'VSITE3FAD':{5:'T',6:'D'},
         'VSITE3OUT':{5:'A',6:'B',7:'C'},
         'VSITE4FD':{6:'A',7:'B',8:'D'},
         }

def parse_atomtype_line(line):
    """ Parses the 'atomtype' line.
    
    Parses lines like this:\n
    <tt> opls_135     CT    6   12.0107    0.0000    A    3.5000e-01    2.7614e-01\n
    C       12.0107    0.0000    A    3.7500e-01    4.3932e-01\n
    Na  11    22.9897    0.0000    A    6.068128070229e+03  2.662662556402e+01  0.0000e+00 ; PARM 5 6\n </tt>
    Look at all the variety!

    @param[in] line Input line.
    @return answer Dictionary containing:\n
    atom type\n
    bonded atom type (if any)\n
    atomic number (if any)\n
    atomic mass\n
    charge\n
    particle type\n
    force field parameters\n
    number of optional fields
    """
    # First split the line up to the comment.  We don't care about the comment at this time
    sline = line.split(';')[0].split()
    # The line must contain at least six fields to be considered data.
    if len(sline) < 6:
        return
    # Using variable "wrd" because the line has a variable number of fields
    # Can you think of a better way?
    wrd = 0
    bonus = 0
    atomtype = sline[wrd]
    batomtype = sline[wrd]
    wrd += 1
    # The bonded atom type, a pecularity of OPLS-AA
    # Test if it begins with a letter.  Seems to work. :)
    if match('[A-Za-z]',sline[wrd]):
        batomtype = sline[wrd]
        wrd += 1
        bonus += 1
    # Now to test if the next line is an atomic number or a mass.
    # Atomic numbers never have decimals...
    atomicnum = -1
    if isint(sline[wrd]):
        atomicnum = int(sline[wrd])
        wrd += 1
        bonus += 1
    # The mass can be overridden in the 'atoms' section.
    mass = float(sline[wrd])
    wrd += 1
    # Atom types have a default charge though this is almost always overridden
    chg  = float(sline[wrd])
    wrd += 1
    # Particle type. Actual atom or virtual site?
    ptp  = sline[wrd]
    wrd += 1
    param = [float(i) for i in sline[wrd:]]
    answer = {'atomtype':atomtype, 'batomtype':batomtype, 'atomicnum':atomicnum, 'mass':mass, 'chg':chg, 'ptp':ptp, 'param':param, 'bonus':bonus}
    return answer

def makeaftype(line, section):
    """ Given a line, section name and list of atom names, return the interaction type and the atoms involved.
    
    For example, we want \n
    <tt> H    O    H    5    1.231258497536e+02    4.269161426840e+02   -1.033397697685e-02   1.304674117410e+04 ; PARM 4 5 6 7 </tt> \n
    to return [H,O,H],'UREY_BRADLEY'
    
    If we are in a TypeSection, it returns a list of atom types; \n
    If we are in a TopolSection, it returns a list of atom names.

    The section is essentially a case statement that picks out the
    appropriate interaction type and makes a list of the atoms
    involved Note that we can call gmxdump for this as well, but I
    prefer to read the force field file directly.

    ToDo: [ atoms ] section might need to be more flexible to accommodate optional fields
    
    """
    sline = line.split()
    atom = []
    ftype = None
    # No sense in doing anything for an empty line or a comment line.
    if len(sline) == 0 or match('^;',line): return None, None
    # Now go through all the cases.
    if section == 'defaults':
        makeaftype.nbtype = int(sline[0])
        makeaftype.crule  = int(sline[1])
    elif section == 'moleculetype':
        makeaftype.res    = sline[0]
    elif section == 'atomtypes':
        atype = parse_atomtype_line(line)
        # This kind of makes me uncomfortable.  Basically we're shifting the word positions
        # based on the syntax of the line in 'atomtype', but it allows the parameter typing to
        # keep up with the flexibility of the syntax of these lines.
        if atype['bonus'] > 0:
            pdict['VDW'] = {4+atype['bonus']:'S',5+atype['bonus']:'T'}
            pdict['VDW_BHAM'] = {4+atype['bonus']:'A', 5+atype['bonus']:'B', 6+atype['bonus']:'C'}
        atom = atype['atomtype']
        ftype = fdict[section][makeaftype.nbtype]
    elif section == 'nonbond_params':
        atom = sline[0] < sline[1] and [sline[0], sline[1]] or [sline[1], sline[0]]
        ftype = pftypes[makeaftype.nbtype]
    elif section == 'atoms':
        atom = [sline[4]]
        ftype = 'COUL'
        # Build the adict here.
        makeaftype.adict.setdefault(sline[3],[]).append(sline[4])
    elif section == 'qtpie':
        # The atom involved is labeled by the atomic number.
        atom = [sline[0]]
        ftype = 'QTPIE'
    elif section == 'bonds':
        atom = [makeaftype.adict[makeaftype.res][int(i) - 1] for i in sline[:2]]
        ftype = fdict[section][int(sline[2])]
    elif section == 'bondtypes':
        # We have several of these conditional expressions.
        # Their purpose is to enforce a unique ordering of atom names
        # so as to avoid duplication of interaction types.
        atom = sline[0] < sline[1] and [sline[0], sline[1]] or [sline[1], sline[0]]
        ftype = fdict[section][int(sline[2])]
    elif section == 'angles':
        atom = [makeaftype.adict[makeaftype.res][int(i) - 1] for i in sline[:3]]
        ftype = fdict[section][int(sline[3])]
    elif section == 'angletypes':
        atom = sline[0] < sline[2] and [sline[0], sline[1], sline[2]] or [sline[2], sline[1], sline[0]]
        ftype = fdict[section][int(sline[3])]
    elif section == 'dihedrals':
        atom = [makeaftype.adict[makeaftype.res][int(i)-1] for i in sline[:4]]
        ftype = fdict[section][int(sline[4])]
        if ftype == 'PDIHS' and len(sline) >= 7:
            # Add the multiplicity of the dihedrals to the interaction type. :)
            ftype += sline[7]
    elif section == 'dihedraltypes':
        atom = sline[0] < sline[3] and [sline[0], sline[1], sline[2], sline[3]] or [sline[3], sline[2], sline[1], sline[0]]
        ftype = fdict[section][int(sline[4])]
        if ftype == 'PDIHS' and len(sline) >= 7:
            ftype += sline[7]
    elif section == 'virtual_sites2':
        atom = [sline[0]]
        ftype = fdict[section][int(sline[3])]
    elif section == 'virtual_sites3':
        atom = [sline[0]]
        ftype = fdict[section][int(sline[4])]
    elif section == 'virtual_sites4':
        atom = [sline[0]]
        ftype = fdict[section][int(sline[5])]
    else:
        return [],"Confused"
    return atom, ftype
# Nonbonded type
makeaftype.nbtype = None
# Combination rule
makeaftype.crule  = None
# The current residue (set by the moleculetype keyword)
makeaftype.res    = None
# The mapping of (this residue, atom number) to (atom name) for building atom-specific interactions in [ bonds ], [ angles ] etc.
makeaftype.adict  = {}

def gmxprint(fnm, vec, type):
    """ Prints a vector to a file to feed it to the modified GROMACS.
    Ported over from the old version so it is a bit archaic for my current taste.

    @param[in] fnm The file name that we're printing the data to
    @param[in] vec 1-D array of data
    @param[in] type Either 'int' or 'double', indicating the type of data.
    """
    fobj = open(fnm, 'w')
    vec = array(vec)
    print >> fobj, vec.shape[0],
    if type == "int":
        for i in vec:
            print >> fobj, i,
    elif type == "double":
        for i in vec:
            print >> fobj, "% .12e" % i,
    fobj.close()
