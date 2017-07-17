""" @package forcebalance.gmxio GROMACS input/output.

@todo Even more stuff from forcefield.py needs to go into here.

@author Lee-Ping Wang
@date 12/2011
"""

import os, sys
import re
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from forcebalance import BaseReader
from forcebalance.engine import Engine
from forcebalance.abinitio import AbInitio
from forcebalance.binding import BindingEnergy
from forcebalance.liquid import Liquid
from forcebalance.lipid import Lipid
from forcebalance.interaction import Interaction
from forcebalance.moments import Moments
from forcebalance.vibration import Vibration
from forcebalance.molecule import Molecule
from forcebalance.thermo import Thermo
from copy import deepcopy
from forcebalance.qchemio import QChem_Dielectric_Energy
import itertools
from collections import defaultdict, OrderedDict
import traceback
import random
#import IPython

from forcebalance.output import getLogger
logger = getLogger(__name__)

def edit_mdp(fin=None, fout=None, options={}, defaults={}, verbose=False):
    """
    Read, create or edit a Gromacs MDP file.  The MDP file contains GROMACS run parameters.
    If the input file exists, it is parsed and options are replaced where "options" overrides them.
    If the "options" dictionary contains more options, they are added at the end.
    If the "defaults" dictionary contains more options, they are added at the end.
    Keys are standardized to lower-case strings where all dashes are replaced by underscores.
    The output file contains the same comments and "dressing" as the input.
    Also returns a dictionary with the final key/value pairs.

    Parameters
    ----------
    fin : str, optional
        Input .mdp file name containing options that are more important than "defaults", but less important than "options"
    fout : str, optional
        Output .mdp file name.
    options : dict, optional
        Dictionary containing mdp options. Existing options are replaced, new options are added at the end, None values are deleted from output mdp.
    defaults : dict, optional
        defaults Dictionary containing "default" mdp options, added only if they don't already exist.
    verbose : bool, optional
        Print out additional information        

    Returns
    -------
    OrderedDict
        Key-value pairs combined from the input .mdp and the supplied options/defaults and equivalent to what's printed in the output mdp.
    """
    clashes = ["pbc"]
    # Make sure that the keys are lowercase, and the values are all strings.
    options = OrderedDict([(key.lower().replace('-','_'), str(val) if val is not None else None) for key, val in options.items()])
    # List of lines in the output file.
    out = []
    # List of options in the output file.
    haveopts = []
    # List of all options in dictionary form, to be returned.
    all_options = OrderedDict()
    if fin is not None and os.path.isfile(fin):
        for line in open(fin).readlines():
            line    = line.strip().expandtabs()
            # The line structure should look something like this:
            # key   = value  ; comments
            # First split off the comments.
            if len(line) == 0:
                out.append('')
                continue
            s = line.split(';',1)
            data = s[0]
            comms = s[1] if len(s) > 1 else None
            # Pure comment lines or empty lines get appended to the output.
            if set(data).issubset([' ']):
                out.append(line)
                continue
            # Now split off the key and value fields at the equals sign.
            keyf, valf = data.split('=',1)
            key = keyf.strip().lower().replace('-','_')
            haveopts.append(key)
            if key in options:
                val = options[key]
                val0 = valf.strip()
                if key in clashes and val != val0:
                    logger.error("edit_mdp tried to set %s = %s but its original value was %s = %s\n" % (key, val, key, val0))
                    raise RuntimeError
                # Passing None as the value causes the option to be deleted
                if val is None: continue
                if len(val) < len(valf):
                    valf = ' ' + val + ' '*(len(valf) - len(val)-1)
                else:
                    valf = ' ' + val + ' '
                lout = [keyf, '=', valf]
                if comms is not None:
                    lout += [';',comms]
                out.append(''.join(lout))
            else:
                out.append(line)
                val = valf.strip()
            all_options[key] = val
    for key, val in options.items():
        key = key.lower().replace('-','_')
        if key not in haveopts:
            haveopts.append(key)
            out.append("%-20s = %s" % (key, val))
            all_options[key] = val
    # Fill in some default options.
    for key, val in defaults.items():
        key = key.lower().replace('-','_')
        options[key] = val
        if key not in haveopts:
            out.append("%-20s = %s" % (key, val))
            all_options[key] = val
    if fout != None:
       file_out = wopen(fout) 
       for line in out:
           print >> file_out, line
       file_out.close()
    if verbose:
        printcool_dictionary(options, title="%s -> %s with options:" % (fin, fout))
    return all_options

def write_ndx(fout, grps, fin=None):
    """
    Create or edit a Gromacs ndx file.
    @param[in] fout Output file name, can be the same as input file name.
    @param[in] grps Dictionary containing key : atom selections.
    @param[in] fin Input file name.
    """
    ndxgrps = OrderedDict()
    atoms = []
    grp = None
    if fin is not None and os.path.isfile(fin):
        for line in open(fin):
            s = line.split()
            if len(s) == 0: continue
            if line.startswith('['):
                grp = s[1]
                ndxgrps[grp] = []
            elif all([isint(i) for i in s]):
                ndxgrps[grp] += [int(i) for i in s]
    ndxgrps.update(grps)
    outf = wopen(fout)
    for name, nums in ndxgrps.items():
        print >> outf, '[ %s ]' % name
        for subl in list(grouper(nums, 15)):
            print >> outf, ' '.join(["%4i" % i for i in subl]) + ' '
    outf.close()

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
dftypes = [None, 'PDIHS', 'IDIHS', 'RBDIHS', 'PIMPDIHS', 'FOURDIHS', None, None, 'TABDIHS', 'PDIHMULS']

## Section -> Interaction type dictionary.
## Based on the section you're in
## and the integer given on the current line, this looks up the
## 'interaction type' - for example, within bonded interactions
## there are four interaction types: harmonic, G96, Morse, and quartic
## interactions.
fdict = {
    'atomtypes'     : nftypes,
    'nonbond_params': pftypes,
    'pairtypes'     : pftypes,
    'bonds'         : bftypes,
    'bondtypes'     : bftypes,
    'angles'        : aftypes,
    'angletypes'    : aftypes,
    'dihedrals'     : dftypes,
    'dihedraltypes' : dftypes,
    'virtual_sites2': ['NONE','VSITE2'],
    'virtual_sites3': ['NONE','VSITE3','VSITE3FD','VSITE3FAD','VSITE3OUT'],
    'virtual_sites4': ['NONE','VSITE4FD','VSITE4FDN']
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
         'PDIHS1':{5:'B', 6:'K'}, 'PDIHS2':{5:'B', 6:'K'}, 'PDIHS3':{5:'B', 6:'K'},
         'PDIHS4':{5:'B', 6:'K'}, 'PDIHS5':{5:'B', 6:'K'}, 'PDIHS6':{5:'B', 6:'K'},
         'PIMPDIHS1':{5:'B', 6:'K'}, 'PIMPDIHS2':{5:'B', 6:'K'}, 'PIMPDIHS3':{5:'B', 6:'K'},
         'PIMPDIHS4':{5:'B', 6:'K'}, 'PIMPDIHS5':{5:'B', 6:'K'}, 'PIMPDIHS6':{5:'B', 6:'K'},
         'FOURDIHS1':{5:'B', 6:'K'}, 'FOURDIHS2':{5:'B', 6:'K'}, 'FOURDIHS3':{5:'B', 6:'K'},
         'FOURDIHS4':{5:'B', 6:'K'}, 'FOURDIHS5':{5:'B', 6:'K'}, 'FOURDIHS6':{5:'B', 6:'K'},
         'PDIHMULS1':{5:'B', 6:'K'}, 'PDIHMULS2':{5:'B', 6:'K'}, 'PDIHMULS3':{5:'B', 6:'K'},
         'PDIHMULS4':{5:'B', 6:'K'}, 'PDIHMULS5':{5:'B', 6:'K'}, 'PDIHMULS6':{5:'B', 6:'K'},
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
         'VSITE4FDN':{6:'A',7:'B',8:'C'},
         'DEF':{3:'FLJ',4:'FQQ'},
         'POL':{3:'ALPHA'},
         'DEFINE':dict([(i, '') for i in range(100)])
         }

def parse_atomtype_line(line):
    """ Parses the 'atomtype' line.
    
    Parses lines like this:\n
    <tt> opls_135     CT    6   12.0107    0.0000    A    3.5000e-01    2.7614e-01\n
    C       12.0107    0.0000    A    3.7500e-01    4.3932e-01\n
    Na  11    22.9897    0.0000    A    6.068128070229e+03  2.662662556402e+01  0.0000e+00 ; PRM 5 6\n </tt>
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
    if re.match('[A-Za-z]',sline[wrd]):
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

class ITP_Reader(BaseReader):

    """Finite state machine for parsing GROMACS force field files.
    
    We open the force field file and read all of its lines.  As we loop
    through the force field file, we look for two types of tags: (1) section
    markers, in GMX indicated by [ section_name ], which allows us to determine
    the section, and (2) parameter tags, indicated by the 'PRM' or 'RPT' keywords.
    
    As we go through the file, we figure out the atoms involved in the interaction
    described on each line.
    
    When a 'PRM' keyword is indicated, it is followed by a number which is the field
    in the line to be modified, starting with zero.  Based on the field number and the
    section name, we can figure out the parameter type.  With the parameter type
    and the atoms in hand, we construct a 'parameter identifier' or pid which uniquely
    identifies that parameter.  We also store the physical parameter value in an array
    called 'pvals0' and the precise location of that parameter (by filename, line number,
    and field number) in a list called 'pfields'.
    
    An example: Suppose in 'my_ff.itp' I encounter the following on lines 146 and 147:
    
    @code
    [ angletypes ]
    CA   CB   O   1   109.47  350.00  ; PRM 4 5
    @endcode
    
    From reading <tt>[ angletypes ]</tt> I know I'm in the 'angletypes' section.
    
    On the next line, I notice two parameters on fields 4 and 5.
    
    From the atom types, section type and field number I know the parameter IDs are <tt>'ANGLESBCACBO'</tt> and <tt>'ANGLESKCACBO'</tt>.
    
    After building <tt>map={'ANGLESBCACBO':1,'ANGLESKCACBO':2}</tt>, I store the values in
    an array: <tt>pvals0=array([109.47,350.00])</tt>, and I put the parameter locations in
    pfields: <tt>pfields=[['my_ff.itp',147,4,1.0],['my_ff.itp',146,5,1.0]]</tt>.  The 1.0
    is a 'multiplier' and I will explain it below.
    
    Note that in the creation of parameter IDs, we run into the issue that the atoms
    involved in the interaction may be labeled in reverse order (e.g. <tt>OCACB</tt>).  Thus,
    we store both the normal and the reversed parameter ID in the map.
    
    Parameter repetition and multiplier:
    
    If <tt>'RPT'</tt> is encountered in the line, it is always in the syntax:
    <tt>'RPT 4 ANGLESBCACAH 5 MINUS_ANGLESKCACAH /RPT'</tt>.  In this case, field 4 is replaced by
    the stored parameter value corresponding to <tt>ANGLESBCACAH</tt> and field 5 is replaced by
    -1 times the stored value of <tt>ANGLESKCACAH</tt>.  Now I just picked this as an example,
    I don't think people actually want a negative angle force constant .. :) the <tt>MINUS</tt>
    keyword does come in handy for assigning atomic charges and virtual site positions.
    In order to achieve this, a multiplier of -1.0 is stored into pfields instead of 1.0.
    
    @todo Note that I can also create the opposite virtual site position by changing the atom
    labeling, woo!
    
    """
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(ITP_Reader,self).__init__(fnm)
        ## The current section that we're in
        self.sec = None
        ## Nonbonded type
        self.nbtype = None
        ## The current molecule (set by the moleculetype keyword)
        self.mol    = None
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict
        ## Listing of all atom names in the file, (probably unnecessary)
        self.atomnames = []
        ## Listing of all atom types in the file, (probably unnecessary)
        self.atomtypes = []
        ## A dictionary of atomic masses
        self.atomtype_to_mass = {}

    def feed(self, line):
        """ Given a line, determine the interaction type and the atoms involved (the suffix).
        
        For example, we want \n
        <tt> H    O    H    5    1.231258497536e+02    4.269161426840e+02   -1.033397697685e-02   1.304674117410e+04 ; PRM 4 5 6 7 </tt> \n
        to give us itype = 'UREY_BRADLEY' and suffix = 'HOH'
        
        If we are in a TypeSection, it returns a list of atom types; \n
        If we are in a TopolSection, it returns a list of atom names.
        
        The section is essentially a case statement that picks out the
        appropriate interaction type and makes a list of the atoms
        involved

        Note that we can call gmxdump for this as well, but I
        prefer to read the force field file directly.
        
        ToDo: [ atoms ] section might need to be more flexible to accommodate optional fields
        
        """
        s          = line.split()
        atom       = []
        self.itype = None
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or re.match('^ *;',line): return None, None
        # Now go through all the cases.
        if re.match('^#', line):
            self.overpfx = 'DEFINE'
            self.oversfx = s[1]
        elif hasattr(self, 'overpfx'):
            delattr(self, 'overpfx')
            delattr(self, 'oversfx')

        if re.match('^ *\[.*\]',line):
            # Makes a word like "atoms", "bonds" etc.
            self.sec = re.sub('[\[\] \n]','',line.strip())
        elif self.sec == 'defaults':
            self.itype = 'DEF'
            self.nbtype = int(s[0])
        elif self.sec == 'moleculetype':
            self.mol    = s[0]
        elif self.sec == 'atomtypes':
            atype = parse_atomtype_line(line)
            # Basically we're shifting the word positions
            # based on the syntax of the line in 'atomtype', but it allows the parameter typing to
            # keep up with the flexibility of the syntax of these lines.
            if atype['bonus'] > 0:
                pdict['VDW'] = {4+atype['bonus']:'S',5+atype['bonus']:'T'}
                pdict['VDW_BHAM'] = {4+atype['bonus']:'A', 5+atype['bonus']:'B', 6+atype['bonus']:'C'}
            atom = atype['atomtype']
            self.atomtype_to_mass[atom] = atype['mass']
            self.itype = fdict[self.sec][self.nbtype]
            self.AtomTypes[atype['atomtype']] = {'AtomClass'    : atype['batomtype'], 
                                                 'AtomicNumber' : atype['atomicnum'], 
                                                 'Mass'         : atype['mass'],
                                                 'Charge'       : atype['chg'],
                                                 'ParticleType' : atype['ptp']}
        elif self.sec == 'nonbond_params':
            atom = [s[0], s[1]]
            self.itype = pftypes[self.nbtype]
        elif self.sec == 'pairtypes':
            atom = [s[0], s[1]]
            self.itype = pftypes[self.nbtype]
        elif self.sec == 'atoms':
            # Ah, this is the atom name, not the atom number.
            # Maybe I should use the atom number.
            atom = [int(s[0])]
            self.atomnames.append(s[4])
            self.itype = 'COUL'
            # Build dictionaries where the key is the residue name
            # and the value is a list of atom numbers, atom types, and atomic masses.
            self.adict.setdefault(self.mol,[]).append(int(s[0]))
            ffAtom = {'AtomType' : s[1], 'ResidueNumber' : int(s[2]), 'ResidueName' : s[3], 'AtomName' : s[4], 'ChargeGroupNumber' : int(s[5]), 'Charge' : float(s[6])}
            self.Molecules.setdefault(self.mol,[]).append(ffAtom)
        elif self.sec == 'polarization':
            atom = [int(s[1])]
            self.itype = 'POL'
        elif self.sec == 'qtpie':
            # The atom involved is labeled by the atomic number.
            atom = [int(s[0])]
            self.itype = 'QTPIE'
        elif self.sec == 'bonds':
            atom = [self.adict[self.mol][int(i)-1] for i in s[:2]]
            self.itype = fdict[self.sec][int(s[2])]
        elif self.sec == 'bondtypes':
            atom = [s[0], s[1]]
            self.itype = fdict[self.sec][int(s[2])]
        elif self.sec == 'angles':
            atom = [self.adict[self.mol][int(i)-1] for i in s[:3]]
            self.itype = fdict[self.sec][int(s[3])]
        elif self.sec == 'angletypes':
            atom = [s[0], s[1], s[2]]
            self.itype = fdict[self.sec][int(s[3])]
        elif self.sec == 'dihedrals':
            atom = [self.adict[self.mol][int(i)-1] for i in s[:4]]
            self.itype = fdict[self.sec][int(s[4])]
            if self.itype in ['PDIHS', 'PIMPDIHS', 'FOURDIHS', 'PDIHMULS'] and len(s) >= 7:
                # Add the multiplicity of the dihedrals to the interaction type.
                self.itype += s[7]
        elif self.sec == 'dihedraltypes':
            # LPW: This needs to be fixed, because some dihedraltypes lines only have 2 atom types.
            # for i in range(len(s)):
            #     if isint(s[i]):
            #         nat = i
            # atom = [s[i] for i in range(nat)]
            # self.itype = fdict[self.sec][int(s[nat+1])]
            # if self.itype in ['PDIHS', 'PIMPDIHS', 'FOURDIHS', 'PDIHMULS'] and len(s) > (nat+3):
            #     self.itype += s[nat+3]
            atom = [s[0], s[1], s[2], s[3]]
            self.itype = fdict[self.sec][int(s[4])]
            if self.itype in ['PDIHS', 'PIMPDIHS', 'FOURDIHS', 'PDIHMULS'] and len(s) >= 7:
                self.itype += s[7]
        elif self.sec == 'virtual_sites2':
            atom = [self.adict[self.mol][int(i)-1] for i in s[:1]]
            #atom = [s[0]]
            self.itype = fdict[self.sec][int(s[3])]
        elif self.sec == 'virtual_sites3':
            atom = [self.adict[self.mol][int(i)-1] for i in s[:1]]
            #atom = [self.adict[self.mol][int(i)-1] for i in s[:3]]
            #atom = [s[0]]
            self.itype = fdict[self.sec][int(s[4])]
        elif self.sec == 'virtual_sites4':
            atom = [self.adict[self.mol][int(i)-1] for i in s[:1]]
            #atom = [self.adict[self.mol][int(i)-1] for i in s[:4]]
            #atom = [s[0]]
            self.itype = fdict[self.sec][int(s[5])]
        else:
            return [],"Confused"
        if type(atom) is list and (len(atom) > 1 and atom[0] > atom[-1]):
            # Enforce a canonical ordering of the atom labels in a parameter ID
            atom = atom[::-1]
        if self.mol is None:
            self.suffix = ':' + ''.join(["%s" % i for i in atom])
        elif self.sec == 'qtpie':
            self.suffix = ':' + '.'.join(["%s" % i for i in atom])
        else:
            self.suffix = ':' + '-'.join([self.mol,'.'.join(["%s" % i for i in atom])])
        self.molatom = (self.mol, atom if type(atom) is list else [atom])

def rm_gmx_baks(dir):
    # Delete the #-prepended files that GROMACS likes to make
    for root, dirs, files in os.walk(dir):
        for file in files:
            if re.match('^#',file):
                os.remove(file)

class GMX(Engine):

    """ Derived from Engine object for carrying out general purpose GROMACS calculations. """

    def __init__(self, name="gmx", **kwargs):
        ## Valid GROMACS-specific keywords.
        self.valkwd = ['gmxsuffix', 'gmxpath', 'gmx_top', 'gmx_mdp', 'gmx_ndx', 'gmx_eq_barostat', 'gmx_pme_order']
        super(GMX,self).__init__(name=name, **kwargs)

    def setopts(self, **kwargs):

        """ Called by __init__ ; Set GROMACS-specific options. """

        ## Disable some optimizations
        os.environ["GMX_MAXBACKUP"] = "-1"
        os.environ["GMX_NO_SOLV_OPT"] = "TRUE"
        os.environ["GMX_NO_ALLVSALL"] = "TRUE"

        ## The suffix to GROMACS executables, e.g. '_d' for double precision.
        if 'gmxsuffix' in kwargs:
            self.gmxsuffix = kwargs['gmxsuffix']
        else:
            warn_once("The 'gmxsuffix' option were not provided; using default.")
            self.gmxsuffix = ''
        
        ## Barostat keyword for equilibration
        if 'gmx_eq_barostat' in kwargs:
            self.gmx_eq_barostat = kwargs['gmx_eq_barostat']

        ## The directory containing GROMACS executables (e.g. mdrun)
        havegmx = False
        if 'gmxpath' in kwargs:
            self.gmxpath = kwargs['gmxpath']

            if type(self.gmxpath) is str and len(self.gmxpath) > 0:
                ## Figure out the GROMACS version - it will determine how programs are called.
                if os.path.exists(os.path.join(self.gmxpath,"gmx"+self.gmxsuffix)):
                    self.gmxversion = 5
                    havegmx = True
                elif os.path.exists(os.path.join(self.gmxpath,"mdrun"+self.gmxsuffix)):
                    self.gmxversion = 4
                    havegmx = True
                else:
                    warn_press_key("The mdrun executable indicated by %s doesn't exist! (Check gmxpath and gmxsuffix)" \
                                   % os.path.join(self.gmxpath,"mdrun"+self.gmxsuffix))

        if not havegmx:
            warn_once("The 'gmxpath' option was not specified; using default.")
            if which('gmx'+self.gmxsuffix) != '':
                self.gmxpath = which('gmx'+self.gmxsuffix)
                self.gmxversion = 5
                havegmx = True
            elif which('mdrun'+self.gmxsuffix) != '':
                self.gmxpath = which('mdrun'+self.gmxsuffix)
                self.gmxversion = 4
                havegmx = True
            else:
                warn_press_key("Please add GROMACS executables to the PATH or specify gmxpath.")
                logger.error("Cannot find the GROMACS executables!\n")
                raise RuntimeError

    def readsrc(self, **kwargs):
        """ Called by __init__ ; read files from the source directory. """

        ## Attempt to determine file names of .gro, .top, and .mdp files
        self.top = onefile(kwargs.get('gmx_top'), 'top')
        self.mdp = onefile(kwargs.get('gmx_mdp'), 'mdp')
        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        else:
            self.mol = Molecule(onefile(kwargs.get('coords'), 'gro', err=True))

    def prepare(self, pbc=False, **kwargs):
        """ Called by __init__ ; prepare the temp directory and figure out the topology. """

        self.gmx_defs = OrderedDict([("integrator", "md"), ("dt", "0.001"), ("nsteps", "0"),
                                     ("nstxout", "0"), ("nstfout", "0"), ("nstenergy", "1"), 
                                     ("nstxtcout", "0"), ("constraints", "none"), ("cutoff-scheme", "group")])
        gmx_opts = OrderedDict([])
        warnings = []
        self.pbc = pbc
        self.have_constraints = False
        if pbc:
            minbox = min([self.mol.boxes[0].a, self.mol.boxes[0].b, self.mol.boxes[0].c])
            if minbox <= 10:
                warn_press_key("Periodic box is set to less than 1.0 ")
            if minbox <= 21:
                # Cutoff diameter should be at least one angstrom smaller than the box size
                # Translates to 0.85 Angstrom for the SPC-216 water box
                rlist = 0.05*(float(int(minbox - 1)))
            else:
                rlist = 1.0
            # Override with user-provided nonbonded_cutoff if exist
            # Since user provides in Angstrom, divide by a factor of 10.
            if 'nonbonded_cutoff' in kwargs:
                rlist = kwargs['nonbonded_cutoff'] / 10
            # Gromacs likes rvdw to be a bit smaller than rlist
            rvdw = rlist - 0.05
            if rlist > 0.05*(float(int(minbox - 1))):
                warn_press_key("nonbonded_cutoff = %.1f should be smaller than half the box size = %.1f Angstrom" % (rlist*10, minbox))
            # Override with user-provided vdw_cutoff if exist
            if 'vdw_cutoff' in kwargs:
                rvdw = kwargs['vdw_cutoff'] / 10
            if rvdw > 0.05*(float(int(minbox - 1))):
                warn_press_key("vdw_cutoff = %.1f should be smaller than half the box size = %.1f Angstrom" % (rvdw*10, minbox))
            rvdw_switch = rvdw-0.05
            gmx_opts["pbc"] = "xyz"
            self.gmx_defs["ns_type"] = "grid"
            self.gmx_defs["nstlist"] = 20
            self.gmx_defs["rlist"] = "%.2f" % rlist
            self.gmx_defs["coulombtype"] = "pme"
            self.gmx_defs["rcoulomb"] = "%.2f" % rlist
            # self.gmx_defs["coulombtype"] = "pme-switch"
            # self.gmx_defs["rcoulomb"] = "%.2f" % (rlist - 0.05)
            # self.gmx_defs["rcoulomb_switch"] = "%.2f" % (rlist - 0.1)
            self.gmx_defs["vdwtype"] = "switch"
            self.gmx_defs["rvdw"] = "%.2f" % rvdw
            self.gmx_defs["rvdw_switch"] = "%.2f" % rvdw_switch
            self.gmx_defs["DispCorr"] = "EnerPres"
        else:
            if 'nonbonded_cutoff' in kwargs:
                warn_press_key("Not using PBC, your provided nonbonded_cutoff will not be used")
            if 'vdw_cutoff' in kwargs:
                warn_press_key("Not using PBC, your provided vdw_cutoff will not be used")
            gmx_opts["pbc"] = "no"
            self.gmx_defs["ns_type"] = "simple"
            self.gmx_defs["nstlist"] = 0
            self.gmx_defs["rlist"] = "0.0"
            self.gmx_defs["coulombtype"] = "cut-off"
            self.gmx_defs["rcoulomb"] = "0.0"
            self.gmx_defs["vdwtype"] = "cut-off"
            self.gmx_defs["rvdw"] = "0.0"
        
        ## Link files into the temp directory.
        if self.top is not None:
            LinkFile(os.path.join(self.srcdir, self.top), self.top, nosrcok=True)
        if self.mdp is not None:
            LinkFile(os.path.join(self.srcdir, self.mdp), self.mdp, nosrcok=True)

        ## Read the .mdp file to determine if there are constraints.
        mdp_dict = edit_mdp(fin=self.mdp)
        if 'constraints' in mdp_dict.keys() and mdp_dict['constraints'].replace('-','_').lower() in ['h_bonds', 'all_bonds', 'h_angles', 'all_angles']:
            self.have_constraints = True

        itptmp = False

        ## Write the appropriate coordinate files.
        if hasattr(self,'FF'):
            # Create the force field in this directory if the force field object is provided.  
            # This is because the .mdp and .top file can be force field files!
            # This bit affects how the geometry optimization is performed, but we should have
            # a more comprehensive way to pass constraint settings through.
            if self.FF.rigid_water:
                self.have_constraints = True
            if not all([os.path.exists(f) for f in self.FF.fnms]):
                # If the parameter files don't already exist, create them for the purpose of
                # preparing the engine, but then delete them afterward.
                itptmp = True
                self.FF.make(np.zeros(self.FF.np))
            self.top = onefile(self.top, 'top')
            self.mdp = onefile(self.mdp, 'mdp')
            # Sanity check; the force fields should be referenced by the .top file.
            if self.top is not None and os.path.exists(self.top):
                if self.top not in self.FF.fnms and (not any([any([fnm in line for fnm in self.FF.fnms]) for line in open(self.top)])):
                    logger.warning('Force field file is not referenced in the .top file\nAssuming the first .itp file is to be included\n')
                    for itpfnm in self.FF.fnms:
                        if itpfnm.endswith(".itp"):
                            break
                    topol = open(self.top).readlines()
                    with wopen(self.top) as f:
                        print >> f, "#include \"%s\"" % itpfnm
                        print >> f
                        for line in topol:
                            print >> f, line,
                    # warn_press_key("None of the force field files %s are referenced in the .top file. "
                    #                "Are you referencing the files through C preprocessor directives?" % self.FF.fnms)

        ## Write out the trajectory coordinates to a .gro file.
        if hasattr(self, 'target') and hasattr(self.target,'shots'):
            self.mol.write("%s-all.gro" % self.name, select=range(self.target.shots))
        else:
            self.mol.write("%s-all.gro" % self.name)
        self.mol[0].write('%s.gro' % self.name)

        ## At this point, we could have gotten a .mdp file from the
        ## target folder or as part of the force field.  If it still
        ## missing, then we may write a default.
        if self.top is not None and os.path.exists(self.top):
            LinkFile(self.top, '%s.top' % self.name)
        else:
            logger.error("No .top file found, cannot continue.\n")
            raise RuntimeError
        edit_mdp(fin=self.mdp, fout="%s.mdp" % self.name, options=gmx_opts, defaults=self.gmx_defs)

        ## Call grompp followed by gmxdump to read the trajectory
        o = self.warngmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name), warnings=warnings)
        self.double = 0
        for line in o:
            if 'double precision' in line:
                self.double = 1
        o = self.callgmx("gmxdump -s %s.tpr -sys" % self.name, copy_stderr=True)
        self.AtomMask = []
        self.AtomLists = defaultdict(list)
        ptype_dict = {'atom': 'A', 'vsite': 'D', 'shell': 'S'}

        ## Here we recognize the residues and charge groups.
        for line in o:
            line = line.replace("=", "= ")
            if "ptype=" in line:
                s = line.split()
                ptype = s[s.index("ptype=")+1].replace(',','').lower()
                resind = int(s[s.index("resind=")+1].replace(',','').lower())
                mass = float(s[s.index("m=")+1].replace(',','').lower())
                charge = float(s[s.index("q=")+1].replace(',','').lower())
                # Gather data for residue number.
                self.AtomMask.append(ptype=='atom')
                self.AtomLists['ResidueNumber'].append(resind)
                self.AtomLists['ParticleType'].append(ptype_dict[ptype])
                self.AtomLists['Charge'].append(charge)
                self.AtomLists['Mass'].append(mass)
            if "cgs[" in line:
                ai = [int(i) for i in line.split("{")[1].split("}")[0].split("..")]
                cg = int(line.split('[')[1].split(']')[0])
                for i in range(ai[1]-ai[0]+1) : self.AtomLists['ChargeGroupNumber'].append(cg)
            if "mols[" in line:
                ai = [int(i) for i in line.split("{")[1].split("}")[0].split("..")]
                mn = int(line.split('[')[1].split(']')[0])
                for i in range(ai[1]-ai[0]+1) : self.AtomLists['MoleculeNumber'].append(mn)
        os.unlink('mdout.mdp')
        os.unlink('%s.tpr' % self.name)
        if hasattr(self,'FF') and itptmp:
            for f in self.FF.fnms: 
                os.unlink(f)

    def get_charges(self):
        # Call gmxdump to get system information and read the charges.
        self.warngmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name))
        o = self.callgmx("gmxdump -s %s.tpr -sys" % self.name, copy_stderr=True)
        # List of charges obtained from reading gmxdump.
        charges = []
        for line in o:
            line = line.replace("=", "= ")
            # These lines contain the charges
            if "ptype=" in line:
                s = line.split()
                charge = float(s[s.index("q=")+1].replace(',','').lower())
                charges.append(charge)
        os.unlink('%s.tpr' % self.name)
        return np.array(charges)

    def links(self):
        topfile = onefile('%s.top' % self.name, 'top', err=True)
        LinkFile(topfile, "%s.top" % self.name)
        mdpfile = onefile('%s.mdp' % self.name, 'mdp', err=True)
        LinkFile(mdpfile, "%s.mdp" % self.name, nosrcok=True)

    def callgmx(self, command, stdin=None, print_to_screen=False, print_command=False, **kwargs):

        """ Call GROMACS; prepend the gmxpath to the call to the GROMACS program. """

        ## Always, always remove backup files.
        rm_gmx_baks(os.getcwd())
        ## Create symbolic links (mainly for the case of .top and .mdp
        ## files which don't exist at object creation)
        self.links()
        ## Call a GROMACS program as you would from the command line.
        csplit = command.split()
        prog = os.path.join(self.gmxpath, csplit[0])
        if self.gmxversion == 5:
            csplit[0] = csplit[0].replace('g_','').replace('gmxdump','dump')
            csplit = ['gmx' + self.gmxsuffix] + csplit
        elif self.gmxversion == 4:
            csplit[0] = prog + self.gmxsuffix
        else:
            raise RuntimeError('gmxversion can only be 4 or 5')
        return _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, **kwargs)

    def warngmx(self, command, warnings=[], maxwarn=1, **kwargs):
        
        """ Call gromacs and allow for certain expected warnings. """

        # Common warning lines:
        # "You are generating velocities so I am assuming"
        # "You are not using center of mass motion removal"
        csplit = command.split()
        if '-maxwarn' in csplit:
            csplit[csplit.index('-maxwarn')+1] = '%i' % maxwarn
        else:
            csplit += ['-maxwarn', '%i' % maxwarn]
        command = ' '.join(csplit)
        o = self.callgmx(command, persist=True, copy_stderr=True, print_error=False, **kwargs)
        warnthis = []
        fatal = 0
        warn = 0
        for line in o:
            if warn:
                warnthis.append(line)
            warn = 0
            if 'Fatal error' in line:
                fatal = 1
            if 'WARNING' in line:
                warn = 1
        if fatal and all([any([a in w for a in warnings]) for w in warnthis]):
            maxwarn = len(warnthis)
            csplit = command.split()
            if '-maxwarn' in csplit:
                csplit[csplit.index('-maxwarn')+1] = '%i' % maxwarn
            else:
                csplit += ['-maxwarn', '%i' % maxwarn]
            command = ' '.join(csplit)
            o = self.callgmx(command, **kwargs)
        elif fatal:
            for line in o:
                logger.error(line+'\n')
            logger.error('grompp encountered a fatal error!\n')
            raise RuntimeError
        return o

    def energy_termnames(self, edrfile=None):

        """ Get a list of energy term names from the .edr file by parsing a system call to g_energy. """

        if edrfile is None:
            edrfile = "%s.edr" % self.name
        if not os.path.exists(edrfile):
            logger.error('Cannot determine energy term names without an .edr file\n')
            raise RuntimeError
        ## Figure out which energy terms need to be printed.
        o = self.callgmx("g_energy -f %s -xvg no" % (edrfile), stdin="Total-Energy\n", copy_stdout=False, copy_stderr=True)
        parsemode = 0
        energyterms = OrderedDict()
        for line in o:
            s = line.split()
            if "Select the terms you want from the following list" in line:
                parsemode = 1
            if parsemode == 1:
                if len(s) > 0 and all([isint(i) for i in s[::2]]):
                    parsemode = 2
            if parsemode == 2:
                if len(s) > 0:
                    try:
                        if all([isint(i) for i in s[::2]]):
                            for j in range(len(s))[::2]:
                                num = int(s[j])
                                name = s[j+1]
                                energyterms[name] = num
                    except: pass
        return energyterms

    def optimize(self, shot=0, crit=1e-4, **kwargs):
        
        """ Optimize the geometry and align the optimized geometry to the starting geometry. """

        ## Write the correct conformation.
        self.mol[shot].write("%s.gro" % self.name)
           
        if "min_opts" in kwargs:
            min_opts = kwargs["min_opts"]
        else:
            # algorithm = "steep"
            if self.have_constraints:
                algorithm = "steep"
            else:
                algorithm = "l-bfgs"
            # Arguments for running minimization.
            min_opts = {"integrator" : algorithm, "emtol" : crit, "nstxout" : 0, "nstfout" : 0, "nsteps" : 10000, "nstenergy" : 1}

        edit_mdp(fin="%s.mdp" % self.name, fout="%s-min.mdp" % self.name, options=min_opts)

        self.warngmx("grompp -c %s.gro -p %s.top -f %s-min.mdp -o %s-min.tpr" % (self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-min -nt 1" % self.name)
        self.callgmx("trjconv -f %s-min.trr -s %s-min.tpr -o %s-min.gro -ndec 9" % (self.name, self.name, self.name), stdin="System")
        self.callgmx("g_energy -xvg no -f %s-min.edr -o %s-min-e.xvg" % (self.name, self.name), stdin='Potential')
        
        E = float(open("%s-min-e.xvg" % self.name).readlines()[-1].split()[1])
        M = Molecule("%s.gro" % self.name, build_topology=False) + Molecule("%s-min.gro" % self.name, build_topology=False)
        if not self.pbc:
            M.align(center=False)
        rmsd = M.ref_rmsd(0)[1]
        M[1].write("%s-min.gro" % self.name)

        return E / 4.184, rmsd

    def evaluate_(self, force=False, dipole=False, traj=None):

        """ 
        Utility function for computing energy, and (optionally) forces and dipoles using GROMACS. 
        
        Inputs:
        force: Switch for calculating the force.
        dipole: Switch for calculating the dipole.
        traj: Trajectory file name.  If present, will loop over these snapshots.
        Otherwise will do a single point evaluation at the current geometry.

        Outputs:
        Result: Dictionary containing energies, forces and/or dipoles.
        """

        shot_opts = OrderedDict([("nsteps", 0), ("nstxout", 0), ("nstxtcout", 0), ("nstenergy", 1)])
        shot_opts["nstfout"] = 1 if force else 0
        edit_mdp(fin="%s.mdp" % self.name, fout="%s-1.mdp" % self.name, options=shot_opts)

        ## Call grompp followed by mdrun.
        self.warngmx("grompp -c %s.gro -p %s.top -f %s-1.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s -nt 1 -rerunvsite %s" % (self.name, "-rerun %s" % traj if traj else ''))

        ## Gather information
        Result = OrderedDict()

        ## Calculate and record energy
        self.callgmx("g_energy -xvg no -f %s.edr -o %s-e.xvg" % (self.name, self.name), stdin='Potential')
        Efile = open("%s-e.xvg" % self.name).readlines()
        Result["Energy"] = np.array([float(Eline.split()[1]) for Eline in Efile])

        ## Calculate and record force
        if force:
            self.callgmx("g_traj -xvg no -s %s.tpr -f %s.trr -of %s-f.xvg -fp" % (self.name, self.name, self.name), stdin='System')
            Result["Force"] = np.array([[float(j) for i, j in enumerate(line.split()[1:]) if self.AtomMask[i/3]] \
                                            for line in open("%s-f.xvg" % self.name).readlines()])
        ## Calculate and record dipole
        if dipole:
            self.callgmx("g_dipoles -s %s.tpr -f %s -o %s-d.xvg -xvg no" % (self.name, traj if traj else '%s.gro' % self.name, self.name), stdin="System\n")
            Result["Dipole"] = np.array([[float(i) for i in line.split()[1:4]] for line in open("%s-d.xvg" % self.name)])

        return Result

    def evaluate_snapshot(self, shot, force=False, dipole=False):

        """ Evaluate variables (energies, force and/or dipole) using GROMACS for a single snapshot. """

        ## Write the correct conformation.
        self.mol[shot].write("%s.gro" % self.name)
        return self.evaluate_(force, dipole)

    def evaluate_trajectory(self, force=False, dipole=False, traj=None):

        """ Evaluate variables (energies, force and/or dipole) using GROMACS over a trajectory. """

        if traj is None:
            if hasattr(self, 'mdtraj'):
                traj = self.mdtraj
            else:
                traj = "%s-all.gro" % self.name
        self.mol[0].write("%s.gro" % self.name)
        return self.evaluate_(force, dipole, traj)

    def make_gro_trajectory(self, fout=None):
        """ Return the MD trajectory as a Molecule object. """
        if fout is None:
            fout = "%s-mdtraj.gro" % self.name
        if not hasattr(self, 'mdtraj'):
            raise RuntimeError('Engine does not have an associated trajectory.')
        self.callgmx("trjconv -f %s -o %s -ndec 9 -pbc mol -novel -noforce" % (self.mdtraj, fout), stdin='System')
        return fout

    def energy_one(self, shot):

        """ Compute the energy using GROMACS for a snapshot. """

        return self.evaluate_snapshot(shot)["Energy"]

    def energy_force_one(self, shot):

        """ Compute the energy and force using GROMACS for a single snapshot; interfaces with AbInitio target. """

        Result = self.evaluate_snapshot(shot, force=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy(self, traj=None):

        """ Compute the energy using GROMACS over a trajectory. """

        return self.evaluate_trajectory(traj=traj)["Energy"]

    def energy_force(self, force=True, traj=None):

        """ Compute the energy and force using GROMACS over a trajectory. """

        Result = self.evaluate_trajectory(force=force, traj=traj)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy_dipole(self, traj=None):
        Result = self.evaluate_trajectory(force=False, dipole=True, traj=traj)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Dipole"]))

    def energy_rmsd(self, shot, optimize=True):

        """ Calculate energy of the selected structure (optionally minimize and return the minimized energy and RMSD). In kcal/mol. """

        if optimize: 
            return self.optimize(shot)
        else:
            self.mol[shot].write("%s.gro" % self.name)
            self.warngmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s-1.tpr" % (self.name, self.name, self.name, self.name))
            self.callgmx("mdrun -deffnm %s-1 -nt 1 -rerunvsite" % (self.name, self.name))
            self.callgmx("g_energy -xvg no -f %s-1.edr -o %s-1-e.xvg" % (self.name, self.name), stdin='Potential')
            E = float(open("%s-1-e.xvg" % self.name).readlines()[0].split()[1])
            return E, 0.0

    def interaction_energy(self, fraga, fragb):

        """ Computes the interaction energy between two fragments over a trajectory. """

        self.mol[0].write("%s.gro" % self.name)

        ## Create an index file with the requisite groups.
        write_ndx('%s.ndx' % self.name, OrderedDict([('A',[i+1 for i in fraga]),('B',[i+1 for i in fragb])]))

        ## .mdp files for fully interacting and interaction-excluded systems.
        edit_mdp(fin='%s.mdp' % self.name, fout='%s-i.mdp' % self.name, options={'xtc_grps':'A B', 'energygrps':'A B'})
        edit_mdp(fin='%s.mdp' % self.name, fout='%s-x.mdp' % self.name, options={'xtc_grps':'A B', 'energygrps':'A B', 'energygrp-excl':'A B'})

        ## Call grompp followed by mdrun for interacting system.
        self.warngmx("grompp -c %s.gro -p %s.top -f %s-i.mdp -n %s.ndx -o %s-i.tpr" % \
                         (self.name, self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-i -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("g_energy -f %s-i.edr -o %s-i-e.xvg -xvg no" % (self.name, self.name), stdin='Potential\n')
        I = []
        for line in open('%s-i-e.xvg' % self.name):
            I.append(sum([float(i) for i in line.split()[1:]]))
        I = np.array(I)

        ## Call grompp followed by mdrun for noninteracting system.
        self.warngmx("grompp -c %s.gro -p %s.top -f %s-x.mdp -n %s.ndx -o %s-x.tpr" % \
                         (self.name, self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-x -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("g_energy -f %s-x.edr -o %s-x-e.xvg -xvg no" % (self.name, self.name), stdin='Potential\n')
        X = []
        for line in open('%s-x-e.xvg' % self.name):
            X.append(sum([float(i) for i in line.split()[1:]]))
        X = np.array(X)

        return (I - X) / 4.184 # kcal/mol

    def multipole_moments(self, shot=0, optimize=True, polarizability=False):
        
        """ Return the multipole moments of the 1st snapshot in Debye and Buckingham units. """
        
        if not self.double:
            warn_once("Single-precision GROMACS detected - recommend that you use double precision build.")

        if polarizability:
            raise NotImplementedError

        if optimize: 
            self.optimize(shot)
            M = Molecule("%s-min.gro" % self.name)
        else:
            self.mol[shot].write("%s.gro" % self.name)
            M = Molecule("%s.gro" % self.name)
        
        #-----
        # g_dipoles uses a different reference point compared to TINKER
        #-----
        # self.callgmx("g_dipoles -s %s-d.tpr -f %s-d.gro -o %s-d.xvg -xvg no" % (self.name, self.name, self.name), stdin="System\n")
        # Dips = np.array([[float(i) for i in line.split()[1:4]] for line in open("%s-d.xvg" % self.name)])[0]
        #-----
        
        ea_debye = 4.803204255928332 # Conversion factor from e*nm to Debye
        q = self.get_charges()
        x = M.xyzs[0] - M.center_of_mass()[0]

        xx, xy, xz, yy, yz, zz = (x[:,i]*x[:,j] for i, j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)])
        # Multiply charges by positions to get dipole moment.
        dip = ea_debye * np.sum(x*q.reshape(-1,1),axis=0)
        dx = dip[0]
        dy = dip[1]
        dz = dip[2]
        qxx = 1.5 * ea_debye * np.sum(q*xx)
        qxy = 1.5 * ea_debye * np.sum(q*xy)
        qxz = 1.5 * ea_debye * np.sum(q*xz)
        qyy = 1.5 * ea_debye * np.sum(q*yy)
        qyz = 1.5 * ea_debye * np.sum(q*yz)
        qzz = 1.5 * ea_debye * np.sum(q*zz)
        tr = qxx+qyy+qzz
        qxx -= tr/3
        qyy -= tr/3
        qzz -= tr/3

        moments = [dx,dy,dz,qxx,qxy,qyy,qxz,qyz,qzz]
        dipole_dict = OrderedDict(zip(['x','y','z'], moments[:3]))
        quadrupole_dict = OrderedDict(zip(['xx','xy','yy','xz','yz','zz'], moments[3:10]))
        calc_moments = OrderedDict([('dipole', dipole_dict), ('quadrupole', quadrupole_dict)])
        # This ordering has to do with the way TINKER prints it out.
        return calc_moments

    def normal_modes(self, shot=0, optimize=True):

        if not self.double:
            warn_once("Single-precision GROMACS detected - recommend that you use double precision build.")

        edit_mdp(fin='%s.mdp' % self.name, fout='%s-nm.mdp' % self.name, options={'integrator':'nm'})

        if optimize:
            self.optimize(shot)
            self.warngmx("grompp -c %s-min.gro -p %s.top -f %s-nm.mdp -o %s-nm.tpr" % (self.name, self.name, self.name, self.name))
        else:
            warn_once("Asking for normal modes without geometry optimization?")
            self.mol[shot].write("%s.gro" % self.name)
            self.warngmx("grompp -c %s.gro -p %s.top -f %s-nm.mdp -o %s-nm.tpr" % (self.name, self.name, self.name, self.name))

        self.callgmx("mdrun -deffnm %s-nm -nt 1 -mtx %s-nm.mtx -v" % (self.name, self.name))
        self.callgmx("g_nmeig -s %s-nm.tpr -f %s-nm.mtx -of %s-nm.xvg -v %s-nm.trr -last 10000 -xvg no" % \
                         (self.name, self.name, self.name, self.name))
        self.callgmx("trjconv -s %s-nm.tpr -f %s-nm.trr -o %s-nm.gro -ndec 9" % (self.name, self.name, self.name), stdin="System")
        NM = Molecule("%s-nm.gro" % self.name, build_topology=False)
        
        calc_eigvals = np.array([float(line.split()[1]) for line in open("%s-nm.xvg" % self.name).readlines()])
        calc_eigvecs = NM.xyzs[1:]
        # Copied from tinkerio.py code
        calc_eigvals = np.array(calc_eigvals)
        calc_eigvecs = np.array(calc_eigvecs)
        # Sort by frequency absolute value and discard the six that are closest to zero
        calc_eigvecs = calc_eigvecs[np.argsort(np.abs(calc_eigvals))][6:]
        calc_eigvals = calc_eigvals[np.argsort(np.abs(calc_eigvals))][6:]
        # Sort again by frequency
        calc_eigvecs = calc_eigvecs[np.argsort(calc_eigvals)]
        calc_eigvals = calc_eigvals[np.argsort(calc_eigvals)]
        for i in range(len(calc_eigvecs)):
            calc_eigvecs[i] /= np.linalg.norm(calc_eigvecs[i])
        return calc_eigvals, calc_eigvecs

    def generate_positions(self):
        ## Call grompp followed by mdrun.
        self.warngmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("trjconv -f %s.trr -o %s-out.gro -ndec 9 -novel -noforce" % (self.name, self.name), stdin='System')
        NewMol = Molecule("%s-out.gro" % self.name)
        return NewMol.xyzs

    def n_snaps(self, nsteps, step_interval, timestep):
        return int((nsteps * 1.0 / step_interval) * timestep)

    def scd_persnap(self, ndx, timestep, final_frame):
        Scd = []
        for snap in range(0, final_frame + 1):
            self.callgmx("g_order -s %s-md.tpr -f %s-md.trr -n %s-%s.ndx -od %s-%s.xvg -b %i -e %i -xvg no" % (self.name, self.name, self.name, ndx, self.name, ndx, snap, snap))
            Scd_snap = []
            for line in open("%s-%s.xvg" % (self.name, ndx)):
                s = [float(i) for i in line.split()]
                Scd_snap.append(s[-1])
            Scd.append(Scd_snap)
        return Scd

    def calc_scd(self, n_snap, timestep):
        # Find deuterium order parameter via g_order.
        # Create index files for each lipid tail.
        sn1_ndx = ['a C15', 'a C17', 'a C18', 'a C19', 'a C20', 'a C21', 'a C22', 'a C23', 'a C24', 'a C25', 'a C26', 'a C27', 'a C28', 'a C29', 'a C30', 'a C31', 'del 0-5', 'q', '']
        sn2_ndx = ['a C34', 'a C36', 'a C37', 'a C38', 'a C39', 'a C40', 'a C41', 'a C42', 'a C43', 'a C44', 'a C45', 'a C46', 'a C47', 'a C48', 'a C49', 'a C50', 'del 0-5', 'q', '']
        self.callgmx("make_ndx -f %s-md.tpr -o %s-sn1.ndx" % (self.name, self.name), stdin="\n".join(sn1_ndx))
        self.callgmx("make_ndx -f %s-md.tpr -o %s-sn2.ndx" % (self.name, self.name), stdin="\n".join(sn2_ndx))
        
        # Loop over g_order for each frame.
        # Adjust nsteps to account for nstxout = 1000.
        sn1 = self.scd_persnap('sn1', timestep, n_snap)
        sn2 = self.scd_persnap('sn2', timestep, n_snap)
        for i in range(0, n_snap + 1):
            sn1[i].extend(sn2[i])
        Scds = np.abs(np.array(sn1))
        return Scds

    def n_nonwater(self, structure_file):
        mol = Molecule(structure_file)
        n_mol = len(mol.molecules)
        n_sol = len([i for i in mol.Data['resname'] if i == 'SOL']) * 1.0 / 3
        return n_mol - n_sol

    def molecular_dynamics(self, nsteps, timestep, temperature=None, pressure=None, nequil=0, nsave=0, minimize=True, threads=None, verbose=False, bilayer=False, **kwargs):
        
        """
        Method for running a molecular dynamics simulation.  

        Required arguments:
        nsteps      = (int)   Number of total time steps
        timestep    = (float) Time step in FEMTOSECONDS
        temperature = (float) Temperature control (Kelvin)
        pressure    = (float) Pressure control (atmospheres)
        nequil      = (int)   Number of additional time steps at the beginning for equilibration
        nsave       = (int)   Step interval for saving data
        minimize    = (bool)  Perform an energy minimization prior to dynamics
        threads     = (int)   Number of MPI-threads

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        Als         = (array)     Average area per lipid in nm^2
        Scds        = (Nx28 array) Deuterium order parameter
        """

        if verbose: logger.info("Molecular dynamics simulation with GROMACS engine.\n")

        # Set the number of threads.
        if threads is None:
            if "OMP_NUM_THREADS" in os.environ:
                threads = int(os.environ["OMP_NUM_THREADS"])
            else:
                threads = 1
                
        # Molecular dynamics options.
        md_opts = OrderedDict([("integrator", "md"), ("nsteps", nsteps), ("dt", timestep / 1000)])
        # Default options if not user specified.
        md_defs = OrderedDict()

        warnings = []
        if temperature is not None:
            md_opts["ref_t"] = temperature
            md_opts["gen_vel"] = "no"
            md_defs["tc_grps"] = "System"
            md_defs["tcoupl"] = "v-rescale"
            md_defs["tau_t"] = 1.0
        if self.pbc:
            md_opts["comm_mode"] = "linear"
            if pressure is not None:
                md_opts["ref_p"] = pressure
                md_defs["pcoupl"] = "parrinello-rahman"
                md_defs["tau_p"] = 1.5
        else:
            md_opts["comm_mode"] = "None"
            md_opts["nstcomm"] = 0

        # In gromacs version 5, default cutoff scheme becomes verlet. 
        # Need to set to group for backwards compatibility
        md_defs["cutoff-scheme"] = 'group'
        md_opts["nstenergy"] = nsave
        md_opts["nstcalcenergy"] = nsave
        md_opts["nstxout"] = nsave
        md_opts["nstvout"] = nsave
        md_opts["nstfout"] = 0
        md_opts["nstxtcout"] = 0

        # Minimize the energy.
        if minimize:
            min_opts = OrderedDict([("integrator", "steep"), ("emtol", 10.0), ("nsteps", 10000)])
            if verbose: logger.info("Minimizing energy... ")
            self.optimize(min_opts=min_opts)
            if verbose: logger.info("Done\n")
            gro1="%s-min.gro" % self.name
        else:
            gro1="%s.gro" % self.name
            self.mol[0].write(gro1)
      
        # Run equilibration.
        if nequil > 0:
            if verbose: logger.info("Equilibrating...\n")
            eq_opts = deepcopy(md_opts)
            eq_opts.update({"nsteps" : nequil, "nstenergy" : 0, "nstxout" : 0,
                            "gen-vel": "yes", "gen-temp" : temperature, "gen-seed" : random.randrange(100000,999999)})
            eq_defs = deepcopy(md_defs)
            if "pcoupl" in eq_defs and hasattr(self, 'gmx_eq_barostat'):
                eq_opts["pcoupl"] = self.gmx_eq_barostat
            edit_mdp(fin='%s.mdp' % self.name, fout="%s-eq.mdp" % self.name, options=eq_opts, defaults=eq_defs)
            self.warngmx("grompp -c %s -p %s.top -f %s-eq.mdp -o %s-eq.tpr" % (gro1, self.name, self.name, self.name), warnings=warnings, print_command=verbose)
            self.callgmx("mdrun -v -deffnm %s-eq -nt %i -stepout %i" % (self.name, threads, nsave), print_command=verbose, print_to_screen=verbose)
            gro2="%s-eq.gro" % self.name
        else:
            gro2=gro1

        # Run production.
        if verbose: logger.info("Production run...\n")
        edit_mdp(fin="%s.mdp" % self.name, fout="%s-md.mdp" % self.name, options=md_opts, defaults=md_defs)
        self.warngmx("grompp -c %s -p %s.top -f %s-md.mdp -o %s-md.tpr" % (gro2, self.name, self.name, self.name), warnings=warnings, print_command=verbose)
        self.callgmx("mdrun -v -deffnm %s-md -nt %i -stepout %i" % (self.name, threads, nsave), print_command=verbose, print_to_screen=verbose)

        self.mdtraj = '%s-md.trr' % self.name

        if verbose: logger.info("Production run finished, calculating properties...\n")
        # Figure out dipoles - note we use g_dipoles and not the multipole_moments function.
        self.callgmx("g_dipoles -s %s-md.tpr -f %s-md.trr -o %s-md-dip.xvg -xvg no" % (self.name, self.name, self.name), stdin="System\n")

        # Figure out which energy terms need to be printed.
        energyterms = self.energy_termnames(edrfile="%s-md.edr" % self.name)
        ekeep = [k for k,v in energyterms.items() if v <= energyterms['Total-Energy']]
        ekeep += ['Temperature', 'Volume', 'Density']

        # Calculate deuterium order parameter for bilayer optimization.
        if bilayer:
            # Figure out how many lipids in simulation.
            n_lip = self.n_nonwater('%s.gro' % self.name)
            n_snap = self.n_snaps(nsteps, 1000, timestep)
            Scds = self.calc_scd(n_snap, timestep)
            al_vars = ['Box-Y', 'Box-X']
            self.callgmx("g_energy -f %s-md.edr -o %s-md-energy-xy.xvg -xvg no" % (self.name, self.name), stdin="\n".join(al_vars))
            Xs = []
            Ys = []
            for line in open("%s-md-energy-xy.xvg" % self.name):
                s = [float(i) for i in line.split()]
                Xs.append(s[-1])
                Ys.append(s[-2])
            Xs = np.array(Xs)
            Ys = np.array(Ys)
            Als = 2 * (Xs * Ys) / n_lip
        else:
            Scds = 0
            Als = 0

        # Perform energy component analysis and return properties.
        self.callgmx("g_energy -f %s-md.edr -o %s-md-energy.xvg -xvg no" % (self.name, self.name), stdin="\n".join(ekeep))
        ecomp = OrderedDict()
        Rhos = []
        Volumes = []
        Kinetics = []
        Potentials = []
        for line in open("%s-md-energy.xvg" % self.name):
            s = [float(i) for i in line.split()]
            for i in range(len(ekeep) - 2):
                val = s[i+1]
                if ekeep[i] in ecomp:
                    ecomp[ekeep[i]].append(val)
                else:
                    ecomp[ekeep[i]] = [val]
            Rhos.append(s[-1])
            Volumes.append(s[-2])
        Rhos = np.array(Rhos)
        Volumes = np.array(Volumes)
        Potentials = np.array(ecomp['Potential'])
        Kinetics = np.array(ecomp['Kinetic-En.'])
        Dips = np.array([[float(i) for i in line.split()[1:4]] for line in open("%s-md-dip.xvg" % self.name)])
        Ecomps = OrderedDict([(key, np.array(val)) for key, val in ecomp.items()])
        # Initialized property dictionary.
        prop_return = OrderedDict()
        prop_return.update({'Rhos': Rhos, 'Potentials': Potentials, 'Kinetics': Kinetics, 'Volumes': Volumes, 'Dips': Dips, 'Ecomps': Ecomps, 'Als': Als, 'Scds': Scds})
        if verbose: logger.info("Finished!\n")
        return prop_return

    def md(self, nsteps=0, nequil=0, verbose=False, deffnm=None, **kwargs):
        
        """
        Method for running a molecular dynamics simulation.  A little different than molecular_dynamics (for thermo.py)

        Required arguments:

        nsteps = (int) Number of total time steps
        
        nequil = (int) Number of additional time steps at the beginning
        for equilibration

        verbose = (bool) Be loud and noisy

        deffnm = (string) default names for simulation output files
        
        The simulation data is written to the working directory.
                
        """

        if verbose:
            logger.info("Molecular dynamics simulation with GROMACS engine.\n")

        # Molecular dynamics options.
        md_opts = OrderedDict()
        # Default options 
        md_defs = OrderedDict(**kwargs)

        if nsteps > 0:
            md_opts["nsteps"] =  nsteps
            
        warnings = []

        if "gmx_ndx" in kwargs:
            ndx_flag = "-n " + kwargs["gmx_ndx"]
        else:
            ndx_flag = ""
            
        gro1 = "%s.gro" % self.name
      
        # Run equilibration.
        if nequil > 0:
            if verbose:
                logger.info("Equilibrating...\n")
            eq_opts = deepcopy(md_opts)
            eq_opts.update({"nsteps" : nequil, "nstenergy" : 0, "nstxout" : 0})
            eq_defs = deepcopy(md_defs)
            edit_mdp(fin='%s.mdp' % self.name,
                     fout="%s-eq.mdp" % self.name,
                     options=eq_opts,
                     defaults=eq_defs)

            self.warngmx(("grompp " +
                          "-c %s " % gro1 +
                          "-f %s-eq.mdp " % self.name +
                          "-p %s.top " % self.name +
                          "%s " % ndx_flag +
                          "-o %s-eq.tpr" % self.name),
                          warnings=warnings,
                          print_command=verbose)
            self.callgmx(("mdrun -v " +
                          "-deffnm %s-eq" % self.name),
                          print_command=verbose,
                          print_to_screen=verbose)
            
            gro2 = "%s-eq.gro" % self.name
        else:
            gro2 = gro1
            
        self.mdtraj = '%s-md.trr' % self.name
        self.mdene  = '%s-md.edr' % self.name
         
        # Run production.
        if verbose:
            logger.info("Production run...\n")
        edit_mdp(fin="%s.mdp" % self.name,
                 fout="%s-md.mdp" % self.name,
                 options=md_opts,
                 defaults=md_defs)
        self.warngmx(("grompp " +
                      "-c %s " % gro2 + 
                      "-f %s-md.mdp " % self.name +
                      "-p %s.top " % self.name +
                      "%s " % ndx_flag +
                      "-o %s-md.tpr" % self.name),
                      warnings=warnings,
                      print_command=verbose)
        self.callgmx(("mdrun -v " +
                      "-deffnm %s-md " % self.name),
                      print_command=verbose,
                      print_to_screen=verbose)

        self.mdtraj = '%s-md.trr' % self.name
        
        if verbose:
            logger.info("Production run finished...\n")                

class Liquid_GMX(Liquid):
    def __init__(self,options,tgt_opts,forcefield):
        # Path to GROMACS executables.
        self.set_option(options,'gmxpath')
        # Suffix for GROMACS executables.
        self.set_option(options,'gmxsuffix')
        # Number of threads for mdrun.
        self.set_option(tgt_opts,'md_threads')
        # Name of the liquid coordinate file.
        self.set_option(tgt_opts,'liquid_coords',default='liquid.gro',forceprint=True)
        # Name of the gas coordinate file.
        self.set_option(tgt_opts,'gas_coords',default='gas.gro',forceprint=True)
        # Whether to use a different barostat for equilibration (default berendsen)
        self.set_option(tgt_opts,'gmx_eq_barostat',forceprint=True)
        # Class for creating engine object.
        self.engine_ = GMX
        # Name of the engine to pass to npt.py.
        self.engname = "gromacs"
        # Command prefix.
        self.nptpfx = "bash rungmx.sh"
        if tgt_opts['remote_backup']:
            self.nptpfx += " -b"
        # Extra files to be linked into the temp-directory.
        self.nptfiles = ['%s.mdp' % os.path.splitext(f)[0] for f in [self.liquid_coords, self.gas_coords]]
        self.nptfiles += ['%s.top' % os.path.splitext(f)[0] for f in [self.liquid_coords, self.gas_coords]]
        # Set some options for the polarization correction calculation.
        self.gas_engine_args = {'gmx_top' : '%s.top' % os.path.splitext(self.gas_coords)[0],
                                'gmx_mdp' : '%s.mdp' % os.path.splitext(self.gas_coords)[0]}
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ['rungmx.sh']
        # Initialize the base class.
        super(Liquid_GMX,self).__init__(options,tgt_opts,forcefield)
        # Error checking.
        for i in self.nptfiles:
            if not os.path.exists(os.path.join(self.root, self.tgtdir, i)):
                logger.error('Please provide %s; it is needed to proceed.\n' % i)
                raise RuntimeError
        # Send back last frame of production trajectory.
        self.extra_output = ['liquid-md.gro']
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output += ['liquid-md.trr']
        # Dictionary of last frames.
        self.LfDict = OrderedDict()
        self.LfDict_New = OrderedDict()

    def npt_simulation(self, temperature, pressure, simnum):
            """ Submit a NPT simulation to the Work Queue. """
            if self.goodstep and (temperature, pressure) in self.LfDict_New:
                self.LfDict[(temperature, pressure)] = self.LfDict_New[(temperature, pressure)]
            if (temperature, pressure) in self.LfDict:
                lfsrc = self.LfDict[(temperature, pressure)]
                lfdest = os.path.join(os.getcwd(), 'liquid.gro')
                logger.info("Copying previous iteration final geometry .gro file: %s to %s\n" % (lfsrc, lfdest))
                shutil.copy2(lfsrc,lfdest)
                self.nptfiles.append(lfdest)
            self.LfDict_New[(temperature, pressure)] = os.path.join(os.getcwd(),'liquid-md.gro')
            super(Liquid_GMX, self).npt_simulation(temperature, pressure, simnum)
            self.last_traj = [i for i in self.last_traj if '.gro' not in i]

class Lipid_GMX(Lipid):
    def __init__(self,options,tgt_opts,forcefield):
        # Path to GROMACS executables.
        self.set_option(options,'gmxpath')
        # Suffix for GROMACS executables.
        self.set_option(options,'gmxsuffix')
        # Number of threads for mdrun.
        self.set_option(tgt_opts,'md_threads')
        # Name of the lipid coordinate file.
        self.set_option(tgt_opts,'lipid_coords',default='lipid.gro',forceprint=True)
        # Whether to use a different barostat for equilibration (default berendsen)
        self.set_option(tgt_opts,'gmx_eq_barostat',forceprint=True)
        # Class for creating engine object.
        self.engine_ = GMX
        # Name of the engine to pass to npt.py.
        self.engname = "gromacs"
        # Command prefix.
        self.nptpfx = "bash rungmx.sh"
        if tgt_opts['remote_backup']:
            self.nptpfx += " -b"
        # Extra files to be linked into the temp-directory.
        self.nptfiles = ['%s.mdp' % os.path.splitext(f)[0] for f in [self.lipid_coords]]
        self.nptfiles += ['%s.top' % os.path.splitext(f)[0] for f in [self.lipid_coords]]
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ['rungmx.sh']
        # Initialize the base class.
        super(Lipid_GMX,self).__init__(options,tgt_opts,forcefield)
        # Error checking.
        for i in self.nptfiles:
            if not os.path.exists(os.path.join(self.root, self.tgtdir, i)):
                logger.error('Please provide %s; it is needed to proceed.\n' % i)
                raise RuntimeError
        # Send back last frame of production trajectory.
        self.extra_output = ['lipid-md.gro']
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output += ['lipid-md.trr']
        # Dictionary of last frames.
        self.LfDict = OrderedDict()
        self.LfDict_New = OrderedDict()

    def npt_simulation(self, temperature, pressure, simnum):
            """ Submit a NPT simulation to the Work Queue. """
            if "n_ic" in self.RefData:
                # Get PT state information.
                phase_reorder = zip(*self.PhasePoints)
                t_index = [i for i, x in enumerate(phase_reorder[0]) if x == temperature] 
                p_index = [i for i, x in enumerate(phase_reorder[1]) if x == pressure] 
                p_u = phase_reorder[-1][list(set(t_index) & set(p_index))[0]]
                PT_vals = (temperature, pressure, p_u)
                # Build dictionary of current iteration final frames, indexed by phase points.
                if not PT_vals in self.lipid_mols_new:
                    self.lipid_mols_new[PT_vals] = [os.path.join(os.getcwd(),'lipid-md.gro')]
                else:
                    self.lipid_mols_new[PT_vals].append(os.path.join(os.getcwd(),'lipid-md.gro'))
                # Run NPT simulation.
                super(Lipid_GMX, self).npt_simulation(temperature, pressure, simnum)
                # When lipid_mols_new is full, move values to lipid_mols for the next iteration.
                if len(self.lipid_mols_new[PT_vals]) == int(self.RefData['n_ic'][PT_vals]):
                    self.lipid_mols[PT_vals] = self.lipid_mols_new[PT_vals]
                    self.lipid_mols_new.pop(PT_vals)
            else:
                if self.goodstep and (temperature, pressure) in self.LfDict_New:
                    self.LfDict[(temperature, pressure)] = self.LfDict_New[(temperature, pressure)]
                if (temperature, pressure) in self.LfDict:
                    lfsrc = self.LfDict[(temperature, pressure)]
                    lfdest = os.path.join(os.getcwd(), 'lipid.gro')
                    logger.info("Copying previous iteration final geometry .gro file: %s to %s\n" % (lfsrc, lfdest))
                    shutil.copy2(lfsrc,lfdest)
                    self.nptfiles.append(lfdest)
                self.LfDict_New[(temperature, pressure)] = os.path.join(os.getcwd(),'lipid-md.gro')
                super(Lipid_GMX, self).npt_simulation(temperature, pressure, simnum)
            self.last_traj = [i for i in self.last_traj if '.gro' not in i]

class AbInitio_GMX(AbInitio):
    """ Subclass of AbInitio for force and energy matching using GROMACS. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates, top and mdp files.
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'gmx_top',default="topol.top")
        self.set_option(tgt_opts,'gmx_mdp',default="shot.mdp")
        self.engine_ = GMX
        ## Initialize base class.
        super(AbInitio_GMX,self).__init__(options,tgt_opts,forcefield)

class BindingEnergy_GMX(BindingEnergy):
    """ Binding energy matching using Gromacs. """
    def __init__(self,options,tgt_opts,forcefield):
        self.engine_ = GMX
        ## Initialize base class.
        super(BindingEnergy_GMX,self).__init__(options,tgt_opts,forcefield)

class Interaction_GMX(Interaction):
    """ Interaction energy matching using GROMACS. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates, top and mdp files.
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'gmx_top',default="topol.top")
        self.set_option(tgt_opts,'gmx_mdp',default="shot.mdp")
        self.engine_ = GMX
        ## Initialize base class.
        super(Interaction_GMX,self).__init__(options,tgt_opts,forcefield)

class Moments_GMX(Moments):
    """ Multipole moment matching using GROMACS. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="conf.gro")
        self.set_option(tgt_opts,'gmx_top',default="topol.top")
        self.set_option(tgt_opts,'gmx_mdp',default="shot.mdp")
        self.engine_ = GMX
        ## Initialize base class.
        super(Moments_GMX,self).__init__(options,tgt_opts,forcefield)
    
class Vibration_GMX(Vibration):
    """ Vibrational frequency matching using GROMACS. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="conf.gro")
        self.engine_ = GMX
        ## Initialize base class.
        super(Vibration_GMX,self).__init__(options,tgt_opts,forcefield)

class Thermo_GMX(Thermo):
    """ Thermodynamical property matching using GROMACS. """
    def __init__(self,options,tgt_opts,forcefield):
        # Path to GROMACS executables.
        self.set_option(options,'gmxpath')
        # Suffix for GROMACS executables.
        self.set_option(options,'gmxsuffix')
        self.engine_ = GMX
        # Name of the engine to pass to scripts.
        self.engname = "gromacs"
        # Command prefix.
        self.mdpfx = "bash gmxprefix.bash"
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ['gmxprefix.bash', 'md_chain.py']
        ## Initialize base class.
        super(Thermo_GMX,self).__init__(options,tgt_opts,forcefield)
