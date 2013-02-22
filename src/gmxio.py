""" @package gmxio GROMACS input/output.

@todo Even more stuff from forcefield.py needs to go into here.

@author Lee-Ping Wang
@date 12/2011
"""

import os
from re import match, sub
from nifty import isint, _exec, warn_press_key, getWorkQueue, LinkFile
from numpy import array
from basereader import BaseReader
from abinitio import AbInitio
from interaction import Interaction
from molecule import Molecule
from copy import deepcopy
from qchemio import QChem_Dielectric_Energy
import itertools
#import IPython

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

class ITP_Reader(BaseReader):

    """Finite state machine for parsing GROMACS force field files.
    
    We open the force field file and read all of its lines.  As we loop
    through the force field file, we look for two types of tags: (1) section
    markers, in GMX indicated by [ section_name ], which allows us to determine
    the section, and (2) parameter tags, indicated by the 'PARM' or 'RPT' keywords.
    
    As we go through the file, we figure out the atoms involved in the interaction
    described on each line.
    
    When a 'PARM' keyword is indicated, it is followed by a number which is the field
    in the line to be modified, starting with zero.  Based on the field number and the
    section name, we can figure out the parameter type.  With the parameter type
    and the atoms in hand, we construct a 'parameter identifier' or pid which uniquely
    identifies that parameter.  We also store the physical parameter value in an array
    called 'pvals0' and the precise location of that parameter (by filename, line number,
    and field number) in a list called 'pfields'.
    
    An example: Suppose in 'my_ff.itp' I encounter the following on lines 146 and 147:
    
    @code
    [ angletypes ]
    CA   CB   O   1   109.47  350.00  ; PARM 4 5
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
        <tt> H    O    H    5    1.231258497536e+02    4.269161426840e+02   -1.033397697685e-02   1.304674117410e+04 ; PARM 4 5 6 7 </tt> \n
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
        if len(s) == 0 or match('^ *;',line): return None, None
        # Now go through all the cases.
        if match('^ *\[.*\]',line):
            # Makes a word like "atoms", "bonds" etc.
            self.sec = sub('[\[\] \n]','',line.strip())
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
            if match('^#',file):
                os.remove(file)

class AbInitio_GMX(AbInitio):
    """ Subclass of AbInitio for force and energy matching using normal GROMACS.
    Implements the prepare_temp_directory and energy_force_driver methods."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory
        self.trajfnm = "all.gro"
        self.topfnm = "topol.top"
        super(AbInitio_GMX,self).__init__(options,tgt_opts,forcefield)
        
    def read_topology(self):
        section = None
        ResidueCounter = -1
        ChargeGroupCounter = -1
        MoleculeCounter = -1
        for line in open(os.path.join(self.root, self.tgtdir,  self.topfnm)).readlines():
            s          = line.split()
            # No sense in doing anything for an empty line or a comment line.
            if len(s) == 0 or match('^;',line): continue
            # Now go through all the cases.
            if match('^\[.*\]',line):
                # Makes a word like "atoms", "bonds" etc.
                section = sub('[\[\] \n]','',line)
            elif section == 'molecules':
                molname    = s[0]
                nummol     = int(s[1])
                FFMolecule = self.FF.FFMolecules[molname]
                mollen = len(FFMolecule)
                for i in range(nummol):
                    resnum = -1
                    cgnum  = -1
                    MoleculeCounter += 1
                    for j in FFMolecule:
                        if j['ResidueNumber'] != resnum:
                            resnum = j['ResidueNumber']
                            ResidueCounter += 1
                        if j['ChargeGroupNumber'] != cgnum:
                            cgnum = j['ChargeGroupNumber']
                            ChargeGroupCounter += 1
                        self.AtomLists['ResidueNumber'].append(ResidueCounter)
                        self.AtomLists['MoleculeNumber'].append(MoleculeCounter)
                        self.AtomLists['ChargeGroupNumber'].append(ChargeGroupCounter)
                        self.AtomLists['ParticleType'].append(j['ParticleType'])
                        self.AtomLists['Mass'].append(j['Mass'])
        self.topology_flag = True
        return

    def prepare_temp_directory(self, options, tgt_opts):
        os.environ["GMX_NO_SOLV_OPT"] = "TRUE"
        os.environ["GMX_NO_ALLVSALL"] = "TRUE"
        abstempdir = os.path.join(self.root,self.tempdir)
        if options['gmxpath'] == None or options['gmxsuffix'] == None:
            warn_press_key('Please set the options gmxpath and gmxsuffix in the input file!')
        if not os.path.exists(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix'])):
            warn_press_key('The mdrun executable pointed to by %s doesn\'t exist! (Check gmxpath and gmxsuffix)' % os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']))
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']),os.path.join(abstempdir,"mdrun"))
        LinkFile(os.path.join(options['gmxpath'],"grompp"+options['gmxsuffix']),os.path.join(abstempdir,"grompp"))
        LinkFile(os.path.join(options['gmxpath'],"g_energy"+options['gmxsuffix']),os.path.join(abstempdir,"g_energy"))
        LinkFile(os.path.join(options['gmxpath'],"g_traj"+options['gmxsuffix']),os.path.join(abstempdir,"g_traj"))
        LinkFile(os.path.join(options['gmxpath'],"trjconv"+options['gmxsuffix']),os.path.join(abstempdir,"trjconv"))
        # Link the run files
        LinkFile(os.path.join(self.root,self.tgtdir,"shot.mdp"),os.path.join(abstempdir,"shot.mdp"))
        LinkFile(os.path.join(self.root,self.tgtdir,self.topfnm),os.path.join(abstempdir,self.topfnm))
        # Write the trajectory to the temp-directory
        self.traj.write(os.path.join(abstempdir,"all.gro"),select=range(self.ns))
        # Print out the first conformation in all.gro to use as conf.gro
        self.traj.write(os.path.join(abstempdir,"conf.gro"),select=[0])

    def energy_force_driver(self, shot):
        """ Computes the energy and force using GROMACS for a single
        snapshot.  This does not require GROMACS-X2. """

        # Remove backup files.
        rm_gmx_baks(os.getcwd())
        # Write the correct conformation.
        self.traj.write('conf.gro',select=[shot])
        # Call grompp followed by mdrun.
        o, e = Popen(["./grompp", "-f", "shot.mdp"],stdout=PIPE,stderr=PIPE).communicate()
        o, e = Popen(["./mdrun", "-nt", "1", "-o", "shot.trr", "-rerunvsite"], stdout=PIPE, stderr=PIPE).communicate()
        # Gather information
        o, e = Popen(["./g_energy","-xvg","no"],stdin=PIPE,stdout=PIPE,stderr=PIPE).communicate('Potential')
        o, e = Popen(["./g_traj","-xvg","no","-f","shot.trr","-of","force.xvg","-fp"],stdin=PIPE,stdout=PIPE,stderr=PIPE).communicate('System')
        E = [float(open("energy.xvg").readlines()[0].split()[1])]
        # When we read in the force, virtual sites are distinguished by whether the force is zero.
        # However, sometimes the force really is exactly zero on an atom, so we have to be a bit tricksier.
        F0 = [float(i) for i in open("force.xvg").readlines()[0].split()[1:] if float(i) != 0.0]
        F1 = [float(i) for i in open("force.xvg").readlines()[0].split()[1:]]
        if len(F0) == len(F1) or len(F1) > 3*self.qmatoms:
            F = F0[:]
        elif len(F0) < 3*self.qmatoms:
            F = F1[:]
        M = array(E + F)
        M = M[:3*self.fitatoms+1]
        return M

    def energy_force_driver_all(self):
        """ Computes the energy and force using GROMACS for a trajectory.  This does not require GROMACS-X2. """
        # Remove backup files.
        rm_gmx_baks(os.getcwd())
        # Call grompp followed by mdrun.
        _exec(["./grompp", "-f", "shot.mdp"], print_command=False)
        _exec(["./mdrun", "-nt", "1", "-o", "shot.trr", "-rerunvsite", "-rerun", "all.gro"], print_command=False)
        # Gather information
        _exec(["./g_energy","-xvg","no"], stdin='Potential', print_command=False)
        _exec(["./g_traj","-xvg","no","-f","shot.trr","-of","force.xvg","-fp"], stdin='System', print_command=False)
        M = []
        Efile = open("energy.xvg").readlines()
        Ffile = open("force.xvg").readlines()
        # Loop through the snapshots
        for Eline, Fline in zip(Efile, Ffile):
            # Compute the potential energy and append to list
            Energy = [float(Eline.split()[1])]
            F0 = [float(i) for i in Fline.split()[1:] if float(i) != 0.0]
            F1 = [float(i) for i in Fline.split()[1:]]
            if len(F0) == len(F1) or len(F1) > 3*self.qmatoms:
                Force = F0[:]
            elif len(F0) < 3*self.qmatoms:
                Force = F1[:]
            M.append(array(Energy + Force)[:3*self.fitatoms+1])
        return array(M)

    def generate_vsite_positions(self):
        """ Call mdrun in order to update the virtual site positions. """
        # Remove backup files.
        rm_gmx_baks(os.getcwd())
        # Call grompp followed by mdrun.
        _exec(["./grompp", "-f", "shot.mdp"], print_command=False)
        _exec(["./mdrun", "-nt", "1", "-o", "shot.trr", "-rerunvsite", "-rerun", "all.gro"], print_command=False)
        # Gather information
        _exec(["./trjconv","-f","shot.trr","-o","trajout.gro","-ndec","6","-novel","-noforce"], stdin='System', print_command=False)
        NewMol = Molecule("trajout.gro")
        self.traj.xyzs = NewMol.xyzs

class Interaction_GMX(Interaction):
    """ Subclass of Interaction for interaction energy matching using GROMACS. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory
        self.trajfnm = "all.gro"
        self.topfnm = "topol.top"
        super(Interaction_GMX,self).__init__(options,tgt_opts,forcefield)
        self.Dielectric = 0.0
        raise Exception('This needs to be fixed')
    
    def prepare_temp_directory(self, options, tgt_opts):
        os.environ["GMX_NO_SOLV_OPT"] = "TRUE"
        abstempdir = os.path.join(self.root,self.tempdir)
        if options['gmxpath'] == None or options['gmxsuffix'] == None:
            warn_press_key('Please set the options gmxpath and gmxsuffix in the input file!')
        if not os.path.exists(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix'])):
            warn_press_key('The mdrun executable pointed to by %s doesn\'t exist! (Check gmxpath and gmxsuffix)' % os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']))
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']),os.path.join(abstempdir,"mdrun"))
        LinkFile(os.path.join(options['gmxpath'],"grompp"+options['gmxsuffix']),os.path.join(abstempdir,"grompp"))
        LinkFile(os.path.join(options['gmxpath'],"g_energy"+options['gmxsuffix']),os.path.join(abstempdir,"g_energy"))
        # Link the run files
        LinkFile(os.path.join(self.root,self.tgtdir,"index.ndx"),os.path.join(abstempdir,"index.ndx"))
        #LinkFile(os.path.join(self.root,self.tgtdir,"shot.mdp"),os.path.join(abstempdir,"shot.mdp"))
        LinkFile(os.path.join(self.root,self.tgtdir,self.topfnm),os.path.join(abstempdir,self.topfnm))
        # Write the trajectory to the temp-directory
        self.traj.write(os.path.join(abstempdir,"all.gro"),select=range(self.ns))
        # Print out the first conformation in all.gro to use as conf.gro
        self.traj.write(os.path.join(abstempdir,"conf.gro"),select=[0])

    def interaction_driver(self, shot):
        """ Computes the energy and force using GROMACS for a single
        snapshot.  This does not require GROMACS-X2. """
        raise NotImplementedError('Per-snapshot interaction energies not implemented, consider using all-at-once')

    def interaction_driver_all(self, dielectric=False):
        """ Computes the energy and force using GROMACS for a trajectory.  This does not require GROMACS-X2. """
        # Remove backup files.
        rm_gmx_baks(os.getcwd())
        # Do the interacting calculation.
        _exec(["./grompp", "-f", "interaction.mdp", "-n", "index.ndx"], print_command=False)
        _exec(["./mdrun", "-nt", "1", "-rerunvsite", "-rerun", "all.gro"], print_command=False)
        # Gather information
        _exec(["./g_energy","-xvg","no"], print_command=False, stdin="Potential\n")
        Interact = array([float(l.split()[1]) for l in open('energy.xvg').readlines()])
        # Do the excluded calculation.
        _exec(["./grompp", "-f", "excluded.mdp", "-n", "index.ndx"], print_command=False)
        _exec(["./mdrun", "-nt", "1", "-rerunvsite", "-rerun", "all.gro"], print_command=False)
        # Gather information
        _exec(["./g_energy","-xvg","no"], print_command=False, stdin="Potential\n")
        Excluded = array([float(l.split()[1]) for l in open('energy.xvg').readlines()])
        # The interaction energy.
        M = Interact - Excluded
        # Now we have the MM interaction energy.
        # We need the COSMO component of the interaction energy now...
        if dielectric:
            traj_dimer = deepcopy(self.traj)
            traj_dimer.add_quantum("qtemp_D.in")
            traj_dimer.write("qchem_dimer.in",ftype="qcin")
            traj_monoA = deepcopy(self.traj)
            traj_monoA.add_quantum("qtemp_A.in")
            traj_monoA.write("qchem_monoA.in",ftype="qcin")
            traj_monoB = deepcopy(self.traj)
            traj_monoB.add_quantum("qtemp_B.in")
            traj_monoB.write("qchem_monoB.in",ftype="qcin")
            wq = getWorkQueue()
            if wq == None:
                warn_press_key("To proceed past this point, a Work Queue must be present")
            print "Computing the dielectric energy"
            Diel_D = QChem_Dielectric_Energy("qchem_dimer.in",wq)
            Diel_A = QChem_Dielectric_Energy("qchem_monoA.in",wq)
            # The dielectric energy for a water molecule should never change.
            if hasattr(self,"Diel_B"):
                Diel_B = self.Diel_B
            else:
                Diel_B = QChem_Dielectric_Energy("qchem_monoB.in",self.wq)
                self.Diel_B = Diel_B
            self.Dielectric = Diel_D - Diel_A - Diel_B
        M += self.Dielectric
        return M
    
