""" @package forcebalance.gmxio GROMACS input/output.

@todo Even more stuff from forcefield.py needs to go into here.

@author Lee-Ping Wang
@date 12/2011
"""

import os
import re
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from numpy import array
from forcebalance import BaseReader
from forcebalance.engine import Engine
from forcebalance.abinitio import AbInitio
from forcebalance.liquid import Liquid
from forcebalance.interaction import Interaction
from forcebalance.molecule import Molecule
from copy import deepcopy
from forcebalance.qchemio import QChem_Dielectric_Energy
import itertools
from collections import OrderedDict
#import IPython

from forcebalance.output import getLogger
logger = getLogger(__name__)

def edit_mdp(fin, fout, options, verbose=False):
    """
    Create or edit a Gromacs MDP file.
    @param[in] fin Input file name.
    @param[in] fout Output file name, can be the same as input file name.
    @param[in] options Dictionary containing mdp options. Existing options are replaced, new options are added at the end.
    """
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
            valen = len(valf)
            if key in options:
                val = options[key]
                if len(val) < len(valf):
                    valf = ' ' + val + ' '*(len(valf) - len(val)-1)
                else:
                    valf = ' ' + val + ' '
                lout = [keyf, '=', valf]
                if comms != None:
                    lout += [';',comms]
                out.append(''.join(lout))
                haveopts.append(key)
            else:
                out.append(line)
    for key, val in options.items():
        if key not in haveopts:
            out.append("%-20s = %s" % (key, val))
    file_out = open(fout,'w')
    for line in out:
        print >> file_out, line
    if verbose:
        printcool_dictionary(options, title="%s -> %s with options:" % (fin, fout))
    file_out.close()

def edit_ndx(fin, fout, newgrps):
    """
    Create or edit a Gromacs ndx file.
    @param[in] fin Input file name.
    @param[in] fout Output file name, can be the same as input file name.
    @param[in] newgrps Dictionary containing key : atom selections.
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
    ndxgrps.update(newgrps)
    outf = open(fout,"w")
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
            atom = [s[0], s[1]]
            self.itype = pftypes[self.nbtype]
        elif self.sec == 'atoms':
            # Ah, this is the atom name, not the atom number.
            # Maybe I should use the atom number.
            atom = [s[0]]
            self.atomnames.append(s[4])
            self.itype = 'COUL'
            # Build dictionaries where the key is the residue name
            # and the value is a list of atom numbers, atom types, and atomic masses.
            self.adict.setdefault(self.mol,[]).append(s[0])
            ffAtom = {'AtomType' : s[1], 'ResidueNumber' : int(s[2]), 'ResidueName' : s[3], 'AtomName' : s[4], 'ChargeGroupNumber' : int(s[5]), 'Charge' : float(s[6])}
            self.Molecules.setdefault(self.mol,[]).append(ffAtom)
        elif self.sec == 'polarization':
            atom = [s[1]]
            self.itype = 'POL'
        elif self.sec == 'qtpie':
            # The atom involved is labeled by the atomic number.
            atom = [s[0]]
            self.itype = 'QTPIE'
        elif self.sec == 'bonds':
            # print self.adict
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
            self.suffix = ':' + ''.join(atom)
        elif self.sec == 'qtpie':
            self.suffix = ':' + ''.join(atom)
        else:
            self.suffix = ':' + '-'.join([self.mol,''.join(atom)])
        self.molatom = (self.mol, atom if type(atom) is list else [atom])

def rm_gmx_baks(dir):
    # Delete the #-prepended files that GROMACS likes to make
    for root, dirs, files in os.walk(dir):
        for file in files:
            if re.match('^#',file):
                os.remove(file)

# Default hard-coded .mdp file for energy and force matching.
shot_mdp = """integrator	= md
dt		= 0.001
nsteps		= 0
nstxout 	= 0
nstfout		= 1
nstenergy	= 1
nstxtcout	= 0
xtc_grps	= System
energygrps	= System

nstlist		= 0
ns_type		= simple
rlist		= 0.0
vdwtype		= cut-off
coulombtype	= cut-off
rcoulomb	= 0.0
rvdw		= 0.0
constraints	= none
pbc		= no
"""

class GMX(Engine):
    """ Derived from Engine object for carrying out general purpose GROMACS calculations. """
    def __init__(self, name="gmx", **kwargs):
        kwargs = {i:j for i,j in kwargs.items() if j != None} 
        super(GMX,self).__init__(name=name, **kwargs)
        
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

        cwd = os.getcwd()
        os.chdir(self.srcdir)
        ## Attempt to determine file names of .gro, .top, and .mdp files
        self.top = onefile('top', kwargs['gmx_top'] if 'gmx_top' in kwargs else None)
        self.mdp = onefile('mdp', kwargs['gmx_mdp'] if 'gmx_mdp' in kwargs else None)
        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs:
            self.mol = Molecule(kwargs['coords'])
        else:
            grofile = onefile('gro')
            self.mol = Molecule(grofile)
        os.chdir(cwd)
        self.postinit()

    def callgmx(self, command, stdin=None, print_to_screen=False, print_command=False, **kwargs):
        """ Call GROMACS; prepend the gmxpath to the call to the GROMACS program. """
        ## Always, always remove backup files.
        rm_gmx_baks(os.getcwd())

        ## Call a GROMACS program as you would from the command line.
        csplit = command.split()
        prog = os.path.join(self.gmxpath, csplit[0])
        csplit[0] = prog
        return _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, **kwargs)

    def prepare(self):

        """ Prepare the calculation.  Write coordinates to the temp-directory.  Read the topology. """

        ## First move into the temp directory if specified by the input arguments.
        cwd = os.getcwd()
        if hasattr(self,'target'):
            dnm = os.path.join(self.root, self.target.tempdir)
        else:
            warn_once("Running in current directory (%s)." % os.getcwd())
            dnm = os.getcwd()
        os.chdir(dnm)

        ## Link files into the temp directory because it's good for reproducibility.
        LinkFile(os.path.join(self.srcdir, self.mdp), os.path.join(dnm, self.mdp), nosrcok=True)
        LinkFile(os.path.join(self.srcdir, self.top), os.path.join(dnm, self.top), nosrcok=True)

        ## Write the appropriate coordinate files.
        if hasattr(self,'target'):
            # Create the force field in this directory if the force field object is provided.  
            # This is because the .mdp and .top file can be force field files! :)
            FF = self.target.FF
            FF.make(np.zeros(FF.np, dtype=float))
            if hasattr(self.target,'shots'):
                self.mol.write(os.path.join(dnm, "%s-all.gro" % self.name), select=range(self.target.shots))
            else:
                self.mol.write(os.path.join(dnm, "%s-all.gro" % self.name))
        else:
            self.mol.write(os.path.join(dnm, "%s-all.gro" % self.name))
        self.mol[0].write(os.path.join(dnm, "%s.gro" % self.name))

        ## Call grompp followed by gmxdump to read the trajectory
        self.callgmx("grompp -c %s.gro -p %s -f %s -o %s.tpr" % (self.name, self.top, self.mdp, self.name))
        o = self.callgmx("gmxdump -s %s.tpr -sys" % self.name, copy_stderr=True)
        self.AtomMask = []
        self.AtomLists = defaultdict(list)
        ptype_dict = {'atom': 'A', 'vsite': 'D', 'shell': 'S'}

        ## Here we recognize the residues and charge groups.
        for line in o:
            if "ptype=" in line:
                s = line.split()
                ptype = s[s.index("ptype=")+1].replace(',','').lower()
                resind = int(s[s.index("resind=")+1].replace(',','').lower())
                mass = float(s[s.index("m=")+1].replace(',','').lower())
                # Gather data for residue number.
                self.AtomMask.append(ptype=='atom')
                self.AtomLists['ResidueNumber'].append(resind)
                self.AtomLists['ParticleType'].append(ptype_dict[ptype])
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
        os.chdir(cwd)
        if hasattr(self,'target'):
            self.target.AtomLists = self.AtomLists
            self.target.AtomMask = self.AtomMask

    def energy_termnames(self):
        if not os.path.exists('%s.edr' % self.name):
            raise RuntimeError('Cannot determine energy term names without an .edr file')
        ## Figure out which energy terms need to be printed.
        o = self.callgmx("g_energy -f %s.edr -xvg no" % (self.name), stdin="Total-Energy\n", copy_stdout=False, copy_stderr=True)
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

    def energy_force_one(self, shot):

        """ Computes the energy and force using GROMACS for a single snapshot. """

        ## Write the correct conformation.
        self.mol.write('%s.gro',select=[shot])

        ## Call grompp followed by mdrun.
        self.callgmx("grompp -c %s.gro -p %s -f %s -o %s.tpr" % (self.name, self.top, self.mdp, self.name))
        self.callgmx("mdrun -deffnm %s -nt 1 -rerunvsite" % self.name)

        ## Gather information
        self.callgmx("g_energy -xvg no -f %s.edr -o %s-energy.xvg" % (self.name, self.name), stdin='Potential')
        self.callgmx("g_traj -xvg no -s %s.tpr -f %s.trr -of %s-force.xvg -fp" % (self.name, self.name, self.name), stdin='System')
        E = [float(open("%s-energy.xvg" % self.name).readlines()[0].split()[1])]
        ## When we read in the force, make sure that we only read in the forces on real atoms.
        F = [float(j) for i, j in enumerate(open("%s-force.xvg" % self.name).readlines()[0].split()[1:]) if self.AtomMask[i/3]]
        M = array(E + F)

        return M

    def energy_force(self):

        """ Computes the energy and force using GROMACS over a trajectory. """

        ## Call grompp followed by mdrun.
        self.callgmx("grompp -c %s.gro -p %s -f %s -o %s.tpr" % (self.name, self.top, self.mdp, self.name))
        self.callgmx("mdrun -deffnm %s -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))

        ## Gather information
        self.callgmx("g_energy -xvg no -f %s.edr -o %s-energy.xvg" % (self.name, self.name), stdin='Potential')
        self.callgmx("g_traj -xvg no -s %s.tpr -f %s.trr -of %s-force.xvg -fp" % (self.name, self.name, self.name), stdin='System')
        M = []
        Efile = open("%s-energy.xvg" % self.name).readlines()
        Ffile = open("%s-force.xvg" % self.name).readlines()
        # Loop through the snapshots
        for Eline, Fline in zip(Efile, Ffile):
            # Compute the potential energy and append to list
            Energy = [float(Eline.split()[1])]
            # When we read in the force, make sure that we only read in the forces on real atoms.
            Force = [float(j) for i, j in enumerate(Fline.split()[1:]) if self.AtomMask[i/3]]
            M.append(array(Energy + Force))
        return array(M)

    def interaction_energy(self, fraga, fragb):

        """ Computes the interaction energy between two fragments over a trajectory. """

        ## Create an index file with the requisite groups.
        edit_ndx(None,'%s.ndx' % self.name, OrderedDict([('A',[i+1 for i in fraga]),('B',[i+1 for i in fragb])]))

        ## .mdp files for fully interacting and interaction-excluded systems.
        imdp = '%s-i.mdp' % os.path.splitext(self.mdp)[0]
        edit_mdp(self.mdp, imdp, {'xtc_grps':'A B', 'energygrps':'A B'})
        xmdp = '%s-x.mdp' % os.path.splitext(self.mdp)[0]
        edit_mdp(self.mdp, xmdp, {'xtc_grps':'A B', 'energygrps':'A B', 'energygrp-excl':'A B'})

        ## Call grompp followed by mdrun for interacting system.
        self.callgmx("grompp -c %s.gro -p %s -f %s -n %s.ndx -o %s-i.tpr" % (self.name, self.top, imdp, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-i -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("g_energy -f %s-i.edr -o %s-i-energy.xvg -xvg no" % (self.name, self.name), stdin='Potential\n')
        I = []
        for line in open('%s-i-energy.xvg' % self.name):
            I.append(sum([float(i) for i in line.split()[1:]]))
        I = array(I)

        ## Call grompp followed by mdrun for noninteracting system.
        self.callgmx("grompp -c %s.gro -p %s -f %s -n %s.ndx -o %s-x.tpr" % (self.name, self.top, xmdp, self.name, self.name))
        self.callgmx("mdrun -deffnm %s-x -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("g_energy -f %s-x.edr -o %s-x-energy.xvg -xvg no" % (self.name, self.name), stdin='Potential\n')
        X = []
        for line in open('%s-x-energy.xvg' % self.name):
            X.append(sum([float(i) for i in line.split()[1:]]))
        X = array(X)

        return I - X

    def generate_vsite_positions(self):
        ## Call grompp followed by mdrun.
        self.callgmx("grompp -c %s.gro -p %s -f %s -o %s.tpr" % (self.name, self.top, self.mdp, self.name))
        self.callgmx("mdrun -deffnm %s -nt 1 -rerunvsite -rerun %s-all.gro" % (self.name, self.name))
        self.callgmx("trjconv -f %s.trr -o %s-out.gro -ndec 6 -novel -noforce" % (self.name, self.name), stdin='System')
        NewMol = Molecule("%s-out.gro" % self.name)
        return NewMol.xyzs

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
    
        ## Create engine object.
        self.engine = GMX(target=self, **engine_args)
        
    def read_topology(self):
        self.topology_flag = True

    def energy_force_driver(self, shot):
        """ Computes the energy and force using GROMACS for a single
        snapshot.  This does not require GROMACS-X2. """
        return self.engine.energy_force_one(shot)

    def energy_force_driver_all(self):
        """ Computes the energy and force using GROMACS for a trajectory.  This does not require GROMACS-X2. """
        return self.engine.energy_force()

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
