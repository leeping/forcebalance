""" @package forcebalance.gmxio GROMACS input/output.

@todo Even more stuff from forcefield.py needs to go into here.

@author Lee-Ping Wang
@date 12/2011
"""

import os
import re
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from forcebalance import BaseReader
from forcebalance.engine import Engine
from forcebalance.abinitio import AbInitio
from forcebalance.liquid import Liquid
from forcebalance.interaction import Interaction
from forcebalance.vibration import Vibration
from forcebalance.molecule import Molecule
from copy import deepcopy
from forcebalance.qchemio import QChem_Dielectric_Energy
import itertools
from collections import OrderedDict
import traceback
#import IPython

from forcebalance.output import getLogger
logger = getLogger(__name__)

def write_mdp(fout, options, fin=None, defaults={}, verbose=False):
    """
    Create or edit a Gromacs MDP file.
    @param[in] fout Output file name, can be the same as input file name.
    @param[in] options Dictionary containing mdp options. Existing options are replaced, new options are added at the end.
    @param[in] fin Input file name.
    @param[in] defaults Default options to add to the mdp only if they don't already exist.
    """
    clashes = ["pbc"]
    # Make sure that the keys are lowercase, and the values are all strings.
    options = OrderedDict([(key.lower(), str(val)) for key, val in options.items()])
    # List of lines in the output file.
    out = []
    # List of options in the output file.
    haveopts = []
    if fin != None and os.path.isfile(fin):
        for line in open(fin).readlines():
            line    = line.strip().expandtabs()
            # The line structure should look something like this:
            # key   = value  ; comments
            # First split off the comments.
            if len(line) == 0:
                out.append('')
            s = line.split(';',1)
            data = s[0]
            comms = s[1] if len(s) > 1 else None
            # Pure comment lines or empty lines get appended to the output.
            if set(data).issubset([' ']):
                out.append(line)
                continue
            # Now split off the key and value fields at the equals sign.
            keyf, valf = data.split('=',1)
            key = keyf.strip().lower()
            haveopts.append(key)
            if key in options:
                val = options[key]
                val0 = valf.strip()
                if key in clashes and val != val0:
                    raise RuntimeError("write_mdp tried to set %s = %s but its original value was %s = %s" % (key, val, key, val0))
                # Passing None as the value causes the option to be deleted
                if val == None: continue
                if len(val) < len(valf):
                    valf = ' ' + val + ' '*(len(valf) - len(val)-1)
                else:
                    valf = ' ' + val + ' '
                lout = [keyf, '=', valf]
                if comms != None:
                    lout += [';',comms]
                out.append(''.join(lout))
            else:
                out.append(line)
    for key, val in options.items():
        if key not in haveopts:
            haveopts.append(key)
            out.append("%-20s = %s" % (key, val))
    # Fill in some default options.
    for key, val in defaults.items():
        if key not in haveopts:
            out.append("%-20s = %s" % (key, val))
    file_out = wopen(fout) 
    for line in out:
        print >> file_out, line
    if verbose:
        printcool_dictionary(options, title="%s -> %s with options:" % (fin, fout))
    file_out.close()

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
    if fin != None and os.path.isfile(fin):
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
        # Also skip C preprocessor lines.
        if len(s) == 0 or re.match('^ *;',line) or re.match('^#',line): return None, None
        # Now go through all the cases.
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
            atom = [int(s[0]), int(s[1])]
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
        if self.mol == None:
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
        self.valkwd = ['gmxsuffix', 'gmxpath', 'gmx_top', 'gmx_mdp']
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

        ## The directory containing GROMACS executables (e.g. mdrun)
        if 'gmxpath' in kwargs:
            self.gmxpath = kwargs['gmxpath']
            if not os.path.exists(os.path.join(self.gmxpath,"mdrun"+self.gmxsuffix)):
                warn_press_key("The mdrun executable indicated by %s doesn't exist! (Check gmxpath and gmxsuffix)" \
                                   % os.path.join(self.gmxpath,"mdrun"+self.gmxsuffix))
        else:
            warn_once("The 'gmxpath' option was not specified; using default.")
            if which('mdrun'+self.gmxsuffix) == '':
                warn_press_key("Please add GROMACS executables to the PATH or specify gmxpath.")
            self.gmxpath = which('mdrun'+self.gmxsuffix)

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory. """

        ## Attempt to determine file names of .gro, .top, and .mdp files
        self.top = onefile('top', kwargs['gmx_top'] if 'gmx_top' in kwargs else None)
        self.mdp = onefile('mdp', kwargs['gmx_mdp'] if 'gmx_mdp' in kwargs else None)
        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs and os.path.exists(kwargs['coords']):
            self.mol = Molecule(kwargs['coords'])
        else:
            grofile = onefile('gro')
            self.mol = Molecule(grofile)

    def prepare(self, **kwargs):

        """ Called by __init__ ; prepare the temp directory and figure out the topology. """

        mdp_default = OrderedDict([("integrator", "md"), ("dt", "0.001"), ("nsteps", "0"), ("nstxout", "0"), 
                                   ("nstfout", "0"), ("nstenergy", "1"), ("nstxtcout", "0"), ("nstlist", "0"), 
                                   ("ns_type", "simple"), ("rlist", "0.0"), ("coulombtype", "cut-off"), ("rcoulomb", "0.0"), 
                                   ("vdwtype", "cut-off"), ("rvdw", "0.0"), ("constraints", "none"), ("pbc", "no")])
        
        ## Link files into the temp directory because it's good for reproducibility.
        if self.top != None:
            LinkFile(os.path.join(self.srcdir, self.top), "%s.top" % self.name, nosrcok=True)
        if self.mdp != None:
            LinkFile(os.path.join(self.srcdir, self.mdp), "%s.mdp" % self.name, nosrcok=True)

        ## Write the appropriate coordinate files.
        if hasattr(self,'target'):
            # Create the force field in this directory if the force field object is provided.  
            # This is because the .mdp and .top file can be force field files!
            FF = self.target.FF
            FF.make(np.zeros(FF.np))
            if not os.path.exists('%s.top' % self.name):
                topfile = onefile('top')
                if topfile != None:
                    LinkFile(topfile, "%s.top" % self.name, nosrcok=True)
            if not os.path.exists('%s.mdp' % self.name):
                mdpfile = onefile('mdp')
                if mdpfile != None:
                    LinkFile(mdpfile, "%s.mdp" % self.name, nosrcok=True)
            # Sanity check; the force fields should be referenced by the .top file.
            if os.path.exists("%s.top" % self.name):
                if not any([any([fnm in line for fnm in FF.fnms]) for line in open("%s.top" % self.name)]):
                    warn_press_key("None of the force field files %s are referenced in the .top file. "
                                   "Are you referencing the files through C preprocessor directives?" % FF.fnms)
            if hasattr(self.target,'shots'):
                self.mol.write("%s-all.gro" % self.name, select=range(self.target.shots))
            else:
                self.mol.write("%s-all.gro" % self.name)
        else:
            self.mol.write("%s-all.gro" % self.name)
        self.mol[0].write("%s.gro" % self.name)

        ## At this point, we could have gotten a .mdp file from the
        ## target folder or as part of the force field.  If it still
        ## missing, then we may write a default.
        if not os.path.exists('%s.top' % self.name):
            raise RuntimeError("No .top file found, cannot continue.")
        if not os.path.exists("%s.mdp" % self.name):
            logger.warn("No .mdp file found, writing default.")
            write_mdp("%s.mdp" % self.name, {}, fin=self.mdp, defaults=mdp_default)
        
        ## Call grompp followed by gmxdump to read the trajectory
        o = self.callgmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name), copy_stderr=True)
        double = 0
        for line in o:
            if 'double precision' in line:
                double = 1
        if not double:
            warn_once("Single-precision GROMACS detected - recommend that you use double precision build.")
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
        os.unlink('%s.gro' % self.name)
        os.unlink('mdout.mdp')
        os.unlink('%s.tpr' % self.name)
        # Delete force field files.
        if hasattr(self, 'target'):
            for f in FF.fnms:
                os.unlink(f)

    def links(self):
        if not os.path.exists('%s.top' % self.name):
            topfile = onefile('top')
            if topfile != None:
                LinkFile(topfile, "%s.top" % self.name)
            else:
                raise RuntimeError("No .top file found, cannot continue.")
        if not os.path.exists('%s.mdp' % self.name):
            mdpfile = onefile('mdp')
            if mdpfile != None:
                LinkFile(mdpfile, "%s.mdp" % self.name, nosrcok=True)
            else:
                raise RuntimeError("No .mdp file found, cannot continue.")

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
        csplit[0] = prog + self.gmxsuffix
        return _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, **kwargs)

    def energy_termnames(self, edrfile=None):

        """ Get a list of energy term names from the .edr file by parsing a system call to g_energy. """

        if edrfile == None:
            edrfile = "%s.edr" % self.name
        if not os.path.exists(edrfile):
            raise RuntimeError('Cannot determine energy term names without an .edr file')
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
            # Arguments for running minimization.
            min_opts = {"integrator" : "l-bfgs", "emtol" : crit, "nstxout" : 0, "nstfout" : 0, "nsteps" : 10000, "nstenergy" : 1}
        
        write_mdp("%s-min.mdp" % self.name, min_opts, fin="%s.mdp" % self.name)

        self.callgmx("grompp -c %s.gro -p %s.top -f %s-min.mdp -o %s-min.tpr" % (self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-min -nt 1" % self.name)
        self.callgmx("trjconv -f %s-min.trr -s %s-min.tpr -o %s-min.gro -ndec 9" % (self.name, self.name, self.name), stdin="System")
        self.callgmx("g_energy -xvg no -f %s-min.edr -o %s-min-e.xvg" % (self.name, self.name), stdin='Potential')
        
        E = float(open("%s-min-e.xvg" % self.name).readlines()[-1].split()[1])
        M = Molecule("%s.gro" % self.name, build_topology=False) + Molecule("%s-min.gro" % self.name, build_topology=False)
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
        write_mdp("%s-1.mdp" % self.name, shot_opts, fin="%s.mdp" % self.name)

        ## Call grompp followed by mdrun.
        self.callgmx("grompp -c %s.gro -p %s.top -f %s-1.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name))
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
            self.callgmx("g_dipoles -s %s.tpr -f %s.gro -o %s-d.xvg -xvg no" % (self.name, self.name, self.name), stdin="System\n")
            Result["Dipole"] = np.array([[float(i) for i in line.split()[1:4]] for line in open("%s-d.xvg" % self.name)])

        return Result

    def evaluate_snapshot(self, shot, force=False, dipole=False):

        """ Evaluate variables (energies, force and/or dipole) using GROMACS for a single snapshot. """

        ## Write the correct conformation.
        self.mol[shot].write("%s.gro" % self.name)
        return self.evaluate_(force, dipole)

    def evaluate_trajectory(self, force=False, dipole=False, traj=None):

        """ Evaluate variables (energies, force and/or dipole) using GROMACS over a trajectory. """

        if traj == None:
            traj = "%s-all.gro" % self.name
        self.mol[0].write("%s.gro" % self.name)
        return self.evaluate_(force, dipole, traj)

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
            self.callgmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s-1.tpr" % (self.name, self.name, self.name, self.name))
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
        write_mdp('%s-i.mdp' % self.name, {'xtc_grps':'A B', 'energygrps':'A B'}, fin='%s.mdp' % self.name)
        write_mdp('%s-x.mdp' % self.name, {'xtc_grps':'A B', 'energygrps':'A B', 'energygrp-excl':'A B'}, fin='%s.mdp' % self.name)

        ## Call grompp followed by mdrun for interacting system.
        self.callgmx("grompp -c %s.gro -p %s.top -f %s-i.mdp -n %s.ndx -o %s-i.tpr" % \
                         (self.name, self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-i -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("g_energy -f %s-i.edr -o %s-i-e.xvg -xvg no" % (self.name, self.name), stdin='Potential\n')
        I = []
        for line in open('%s-i-e.xvg' % self.name):
            I.append(sum([float(i) for i in line.split()[1:]]))
        I = np.array(I)

        ## Call grompp followed by mdrun for noninteracting system.
        self.callgmx("grompp -c %s.gro -p %s.top -f %s-x.mdp -n %s.ndx -o %s-x.tpr" % \
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
        q = np.array(self.AtomLists['Charge'])
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

        write_mdp('%s-nm.mdp' % self.name, {'integrator':'nm'}, fin='%s.mdp' % self.name)

        if optimize:
            self.optimize(shot)
            self.callgmx("grompp -c %s-min.gro -p %s.top -f %s-nm.mdp -o %s-nm.tpr" % (self.name, self.name, self.name, self.name))
        else:
            warn_once("Asking for normal modes without geometry optimization?")
            self.mol[shot].write("%s.gro" % self.name)
            self.callgmx("grompp -c %s.gro -p %s.top -f %s-nm.mdp -o %s-nm.tpr" % (self.name, self.name, self.name, self.name))

        self.callgmx("mdrun -deffnm %s-nm -mtx %s-nm.mtx -v" % (self.name, self.name))
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

    def generate_vsite_positions(self):
        ## Call grompp followed by mdrun.
        self.callgmx("grompp -c %s.gro -p %s.top -f %s.mdp -o %s.tpr" % (self.name, self.name, self.name, self.name))
        self.callgmx("mdrun -deffnm %s -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("trjconv -f %s.trr -o %s-out.gro -ndec 9 -novel -noforce" % (self.name, self.name), stdin='System')
        NewMol = Molecule("%s-out.gro" % self.name)
        return NewMol.xyzs

    def molecular_dynamics(self, nsteps, timestep, temperature=None, pressure=None, nequil=0, nsave=0, minimize=True, pbc=True, threads=None, **kwargs):
        
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
        pbc         = (bool)  Periodic boundary conditions; remove COM motion
        threads     = (int)   Number of MPI-threads

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        """

        # Set the number of threads.
        if threads == None:
            if "OMP_NUM_THREADS" in os.environ:
                threads = int(os.environ["OMP_NUM_THREADS"])
            else:
                threads = 1
                
        # Molecular dynamics options.
        md_opts = OrderedDict([("integrator", "md"), ("nsteps", nsteps), ("dt", timestep / 1000)])
        # Default options if not user specified.
        md_defs = OrderedDict()
        eq_defs = OrderedDict()
        if temperature != None:
            md_opts["ref_t"] = temperature
            md_opts["gen_temp"] = temperature
            md_defs["tc_grps"] = "System"
            md_defs["tcoupl"] = "v-rescale"
            md_defs["tau_t"] = 1.0
        md_opts["nstenergy"] = nsave
        md_opts["nstcalcenergy"] = nsave
        md_opts["nstxout"] = nsave
        md_opts["nstvout"] = nsave
        md_opts["nstfout"] = 0
        md_opts["nstxtcout"] = 0
        if pbc:
            if minbox <= 10:
                warn_press_key("Periodic box is set to less than 1.0 ")
            elif minbox <= 21:
                # Cutoff diameter should be at least one angstrom smaller than the box size
                # Translates to 0.85 Angstrom for the SPC-216 water box
                rlist = 0.05*(float(int(minbox - 1)))
            else:
                rlist = 1.0
            md_opts["pbc"] = "xyz"
            md_opts["comm_mode"] = "linear"
            if pressure != None:
                md_opts["ref_p"] = pressure
                md_defs["pcoupl"] = "parrinello-rahman"
                md_defs["tau_p"] = 1.5
            md_defs["ns_type"] = "grid"
            minbox = min([self.mol.boxes[0].a, self.mol.boxes[0].b, self.mol.boxes[0].c])
            md_defs["nstlist"] = 20
            md_defs["rlist"] = "%.2f" % rlist
            md_defs["coulombtype"] = "pme-switch"
            md_defs["rcoulomb"] = "%.2f" % (rlist - 0.05)
            md_defs["rcoulomb_switch"] = "%.2f" % (rlist - 0.1)
            md_defs["vdwtype"] = "switch"
            md_defs["rvdw"] = "%.2f" % (rlist - 0.05)
            md_defs["rvdw_switch"] = "%.2f" % (rlist - 0.1)
            md_defs["DispCorr"] = "EnerPres"
        else:
            if pressure != None:
                raise RuntimeError("To set a pressure, enable periodic boundary conditions.")
            md_opts["pbc"] = "no"
            md_opts["comm_mode"] = "None"
            md_opts["nstcomm"] = 0
            md_defs["ns_type"] = "simple"
            md_defs["nstlist"] = 0
            md_defs["rlist"] = "0.0"
            md_defs["coulombtype"] = "cut-off"
            md_defs["rcoulomb"] = "0.0"
            md_defs["vdwtype"] = "switch"
            md_defs["rvdw"] = "0.0"

        # Minimize the energy.
        if minimize:
            min_opts = OrderedDict([("integrator", "steep"), ("emtol", 10.0), ("nsteps", 10000)])
            self.optimize(min_opts=min_opts)
            gro1="%s-min.gro" % self.name
        else:
            gro1="%s.gro" % self.name
            self.mol[0].write(gro1)
      
        # Run equilibration.
        if nequil > 0:
            eq_opts = deepcopy(md_opts)
            eq_opts.update({"nsteps" : nequil, "nstenergy" : 0, "nstcalcenergy" : 0, "nstxout" : 0})
            eq_defs = deepcopy(md_defs)
            if "pcoupl" in eq_defs: eq_defs["pcoupl"] = "berendsen"
            write_mdp("%s-eq.mdp" % self.name, eq_opts, fin='%s.mdp' % self.name, defaults=eq_defs)
            self.callgmx("grompp -c %s -p %s.top -f %s-eq.mdp -o %s-eq.tpr" % (gro1, self.name, self.name, self.name))
            self.callgmx("mdrun -v -deffnm %s-eq -nt %i -stepout %i" % (self.name, threads, nsave))
            gro2="%s-eq.gro" % self.name
        else:
            gro2=gro1

        # Run production.
        write_mdp("%s-md.mdp" % self.name, md_opts, fin="%s.mdp" % self.name, defaults=md_defs)
        self.callgmx("grompp -c %s -p %s.top -f %s-md.mdp -o %s-md.tpr" % (gro2, self.name, self.name, self.name))
        self.callgmx("mdrun -v -deffnm %s-md -nt %i -stepout %i" % (self.name, threads, nsave))

        # Figure out dipoles - note we use g_dipoles and not the multipole_moments function.
        self.callgmx("g_dipoles -s %s-md.tpr -f %s-md.trr -o %s-md-dip.xvg -xvg no" % (self.name, self.name, self.name), stdin="System\n")
        
        # Figure out which energy terms need to be printed.
        energyterms = self.energy_termnames(edrfile="%s-md.edr" % self.name)
        ekeep = [k for k,v in energyterms.items() if v <= energyterms['Total-Energy']]
        ekeep += ['Volume', 'Density']

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
        return Rhos, Potentials, Kinetics, Volumes, Dips, OrderedDict([(key, np.array(val)) for key, val in ecomp.items()])

class AbInitio_GMX(AbInitio):
    """ Subclass of AbInitio for force and energy matching using normal GROMACS.
    Implements the prepare_temp_directory and energy_force_driver methods."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates, top and mdp files.
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'gmx_top',default="topol.top")
        self.set_option(tgt_opts,'gmx_mdp',default="shot.mdp")

        ## Initialize base class.
        super(AbInitio_GMX,self).__init__(options,tgt_opts,forcefield)

        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
    
        ## Create engine object.
        self.engine = GMX(target=self, **engine_args)
        self.AtomLists = self.engine.AtomLists
        self.AtomMask = self.engine.AtomMask
        
    def read_topology(self):
        self.topology_flag = True

    def generate_vsite_positions(self):
        """ Call mdrun in order to update the virtual site positions. """
        self.engine.generate_vsite_positions()

class Liquid_GMX(Liquid):
    def __init__(self,options,tgt_opts,forcefield):
        super(Liquid_GMX,self).__init__(options,tgt_opts,forcefield)
        # Number of threads in mdrun
        self.set_option(tgt_opts,'mdrun_threads')
        self.liquid_fnm = "liquid.gro"
        self.liquid_conf = Molecule(os.path.join(self.root, self.tgtdir,"liquid.gro"))
        self.liquid_mol = None
        self.gas_fnm = "gas.gro"
        if os.path.exists(os.path.join(self.root, self.tgtdir,"all.gro")):
            self.liquid_mol = Molecule(os.path.join(self.root, self.tgtdir,"all.gro"))
            print "Found collection of starting conformations, length %i!" % len(self.liquid_mol)
        if self.do_self_pol:
            warn_press_key("Self-polarization correction not implemented yet when using GMX")
        # Command prefix.
        self.nptpfx = 'sh rungmx.sh'
         # Suffix to command string for launching NPT simulations.
        self.nptsfx += ["--nt %i" % self.mdrun_threads]
        # List of extra files to upload to Work Queue.
        self.nptfiles += ['rungmx.sh', 'liquid.top', 'liquid.mdp', 'gas.top', 'gas.mdp']
        # MD engine argument supplied to command string for launching NPT simulations.
        self.engine = "gromacs"
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output = ['liquid-md.trr']

    def prepare_temp_directory(self,options,tgt_opts):
        """ Prepare the temporary directory by copying in important files. """
        os.environ["GMX_NO_SOLV_OPT"] = "TRUE"
        os.environ["GMX_NO_ALLVSALL"] = "TRUE"
        abstempdir = os.path.join(self.root,self.tempdir)
        if options['gmxpath'] == None or options['gmxsuffix'] == None:
            warn_press_key('Please set the options gmxpath and gmxsuffix in the input file!')
        if not os.path.exists(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix'])):
            warn_press_key('The mdrun executable pointed to by %s doesn\'t exist! (Check gmxpath and gmxsuffix)' % os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']))
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","npt.py"),os.path.join(abstempdir,"npt.py"))
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","rungmx.sh"),os.path.join(abstempdir,"rungmx.sh"))
        # Link the run files
        for phase in ["liquid","gas"]:
            LinkFile(os.path.join(self.root,self.tgtdir,"%s.mdp" % phase),os.path.join(abstempdir,"%s.mdp" % phase))
            LinkFile(os.path.join(self.root,self.tgtdir,"%s.top" % phase),os.path.join(abstempdir,"%s.top" % phase))
            LinkFile(os.path.join(self.root,self.tgtdir,"%s.gro" % phase),os.path.join(abstempdir,"%s.gro" % phase))

    def polarization_correction(self,mvals):
        # This needs to be implemented
        return 0

class Interaction_GMX(Interaction):
    """ Subclass of Interaction for interaction energy matching using GROMACS. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates, top and mdp files.
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'gmx_top',default="topol.top")
        self.set_option(tgt_opts,'gmx_mdp',default="shot.mdp")

        ## Initialize base class.
        super(Interaction_GMX,self).__init__(options,tgt_opts,forcefield)

        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
    
        ## Create engine object.
        self.engine = GMX(target=self, **engine_args)
    
    def interaction_driver(self, shot):
        """ Computes the energy and force using GROMACS for a single
        snapshot.  This does not require GROMACS-X2. """
        raise NotImplementedError('Per-snapshot interaction energies not implemented, consider using all-at-once')

    def interaction_driver_all(self, dielectric=False):
        """ Computes the energy and force using GROMACS for a trajectory.  This does not require GROMACS-X2. """
        return self.engine.interaction_energy(self.select1, self.select2)
        ## Now we have the MM interaction energy.
        ## We need the COSMO component of the interaction energy now...
        # if dielectric:
        #     traj_dimer = deepcopy(self.mol)
        #     traj_dimer.add_quantum("qtemp_D.in")
        #     traj_dimer.write("qchem_dimer.in",ftype="qcin")
        #     traj_monoA = deepcopy(self.mol)
        #     traj_monoA.add_quantum("qtemp_A.in")
        #     traj_monoA.write("qchem_monoA.in",ftype="qcin")
        #     traj_monoB = deepcopy(self.mol)
        #     traj_monoB.add_quantum("qtemp_B.in")
        #     traj_monoB.write("qchem_monoB.in",ftype="qcin")
        #     wq = getWorkQueue()
        #     if wq == None:
        #         warn_press_key("To proceed past this point, a Work Queue must be present")
        #     print "Computing the dielectric energy"
        #     Diel_D = QChem_Dielectric_Energy("qchem_dimer.in",wq)
        #     Diel_A = QChem_Dielectric_Energy("qchem_monoA.in",wq)
        #     # The dielectric energy for a water molecule should never change.
        #     if hasattr(self,"Diel_B"):
        #         Diel_B = self.Diel_B
        #     else:
        #         Diel_B = QChem_Dielectric_Energy("qchem_monoB.in",self.wq)
        #         self.Diel_B = Diel_B
        #     self.Dielectric = Diel_D - Diel_A - Diel_B
        # M += self.Dielectric

class Vibration_GMX(Vibration):

    """Subclass of Target for vibrational frequency matching
    using GROMACS.  Provides optimized geometry, vibrational frequencies (in cm-1),
    and eigenvectors."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="conf.gro")
        ## Initialize base class.
        super(Vibration_GMX,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
        ## Create engine object.
        self.engine = GMX(target=self, **engine_args)
