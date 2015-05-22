""" @package forcebalance.amberio AMBER force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os, sys, re
import copy
from re import match, sub, split, findall
import networkx as nx
from forcebalance.nifty import isint, isfloat, _exec, LinkFile, warn_once, which, onefile, listfiles, warn_press_key, wopen
import numpy as np
from forcebalance import BaseReader
from forcebalance.engine import Engine
from forcebalance.abinitio import AbInitio
from forcebalance.interaction import Interaction
from forcebalance.vibration import Vibration
from forcebalance.molecule import Molecule
from collections import OrderedDict, defaultdict

from forcebalance.output import getLogger
logger = getLogger(__name__)

mol2_pdict = {'COUL':{'Atom':[1], 8:''}}

frcmod_pdict = {'BONDS': {'Atom':[0], 1:'K', 2:'B'},
                'ANGLES':{'Atom':[0], 1:'K', 2:'B'},
                'PDIHS1':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS2':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS3':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS4':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS5':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS6':{'Atom':[0], 2:'K', 3:'B'},
                'IDIHS' :{'Atom':[0], 1:'K', 3:'B'},
                'VDW':{'Atom':[0], 1:'S', 2:'T'}
                }

def is_mol2_atom(line):
    s = line.split()
    if len(s) < 9:
        return False
    return all([isint(s[0]), isfloat(s[2]), isfloat(s[3]), isfloat(s[4]), isfloat(s[8])])

def write_leap(fnm, mol2=[], frcmod=[], pdb=None, prefix='amber', spath = [], delcheck=False):
    """ Parse and edit an AMBER LEaP input file. Output file is written to inputfile_ (with trailing underscore.) """
    have_fmod = []
    have_mol2 = []
    # The lines that will be printed out to actually run tleap
    line_out = []
    aload = ['loadamberparams', 'source', 'loadoff']
    aload_eq = ['loadmol2']
    spath.append('.')
    # Default name for the "unit" that is written to prmtop/inpcrd
    ambername = 'amber'
    for line in open(fnm):
        # Skip comment lines
        if line.strip().startswith('#') : continue
        line = line.split('#')[0]
        s = line.split()
        ll = line.lower()
        ls = line.lower().split()
        # Check to see if all files being loaded are in the search path
        if '=' in line:
            if ll.split('=')[1].split()[0] in aload_eq:
                if not any([os.path.exists(os.path.join(d, s[-1])) for d in spath]):
                    logger.error("The file in this line cannot be loaded : " + line.strip())
                    raise RuntimeError
        elif len(ls) > 0 and ls[0] in aload:
            if not any([os.path.exists(os.path.join(d, s[-1])) for d in spath]):
                logger.error("The file in this line cannot be loaded : " + line.strip())
                raise RuntimeError
        if len(s) >= 2 and ls[0] == 'loadamberparams':
            have_fmod.append(s[1])
        if len(s) >= 2 and 'loadmol2' in ll:
            # Adopt the AMBER molecule name from the loadpdb line.
            ambername = line.split('=')[0].strip()
            have_mol2.append(s[-1])
        if len(s) >= 2 and 'loadpdb' in ll:
            # Adopt the AMBER molecule name from the loadpdb line.
            ambername = line.split('=')[0].strip()
            # If we pass in our own PDB, then this line is replaced.
            if pdb is not None:
                line = '%s = loadpdb %s\n' % (ambername, pdb)
        if len(s) >= 1 and ls[0] == 'check' and delcheck:
            # Skip over check steps if so decreed
            line = "# " + line
        if 'saveamberparm' in ll: 
            # We'll write the saveamberparm line ourselves
            continue
        if len(s) >= 1 and ls[0] == 'quit':
            # Don't write the quit line.
            break
        if not line.endswith('\n') : line += '\n'
        line_out.append(line)
    # Sanity checks: If frcmod and mol2 files are provided to this function,
    # they should be in the leap.cmd file as well.  There should be exactly
    # one PDB file being loaded.
    for i in frcmod:
        if i not in have_fmod:
            warn_press_key("WARNING: %s is not being loaded in %s" % (i, fnm))
    for i in mol2:
        if i not in have_mol2:
            warn_press_key("WARNING: %s is not being loaded in %s" % (i, fnm))

    fout = fnm+'_'
    line_out.append('saveamberparm %s %s.prmtop %s.inpcrd\n' % (ambername, prefix, prefix))
    line_out.append('quit\n')
    with wopen(fout) as f: print >> f, ''.join(line_out)

class Mol2_Reader(BaseReader):
    """Finite state machine for parsing Mol2 force field file. (just for parameterizing the charges)"""
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(Mol2_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = mol2_pdict
        ## The atom numbers in the interaction (stored in the parser)
        self.atom   = []
        ## The mol2 file provides a list of atom names
        self.atomnames = []
        ## The section that we're in
        self.section = None
        # The name of the molecule
        self.mol = None

    def feed(self, line):
        s          = line.split()
        self.ln   += 1
        # In mol2 files, the only defined interaction type is the Coulomb interaction.
        if line.strip().lower() == '@<tripos>atom':
            self.itype = 'COUL'
            self.section = 'Atom'
        elif line.strip().lower() == '@<tripos>bond':
            self.itype = 'None'
            self.section = 'Bond'
        elif line.strip().lower() == '@<tripos>substructure':
            self.itype = 'None'
            self.section = 'Substructure'
        elif line.strip().lower() == '@<tripos>molecule':
            self.itype = 'None'
            self.section = 'Molecule'
        elif self.section == 'Molecule' and self.mol is None:
            self.mol = '_'.join(s)
        elif not is_mol2_atom(line):
            self.itype = 'None'

        if is_mol2_atom(line) and self.itype == 'COUL':
            #self.atomnames.append(s[self.pdict[self.itype]['Atom'][0]])
            #self.adict.setdefault(self.mol,[]).append(s[self.pdict[self.itype]['Atom'][0]])
            self.atomnames.append(s[0])
            self.adict.setdefault(self.mol,[]).append(s[0])

        if self.itype in self.pdict:
            if 'Atom' in self.pdict[self.itype] and match(' *[0-9]', line):
                # List the atoms in the interaction.
                #self.atom = [s[i] for i in self.pdict[self.itype]['Atom']]
                self.atom = [s[0]]
                # The suffix of the parameter ID is built from the atom    #
                # types/classes involved in the interaction.
                self.suffix = ':' + '-'.join([self.mol,''.join(self.atom)])
            #self.suffix = '.'.join(self.atom)
                self.molatom = (self.mol, self.atom if type(self.atom) is list else [self.atom])

class FrcMod_Reader(BaseReader):
    """Finite state machine for parsing FrcMod force field file."""
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(FrcMod_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = frcmod_pdict
        ## The atom numbers in the interaction (stored in the parser)
        self.atom   = []
        ## Whether we're inside the dihedral section
        self.dihe  = False
        ## The frcmod file never has any atoms in it
        self.adict = {None:None}
        
    def Split(self, line):
        return split(' +(?!-(?![0-9.]))', line.strip().replace('\n',''))

    def Whites(self, line):
        return findall(' +(?!-(?![0-9.]))', line.replace('\n',''))

    def feed(self, line):
        s          = self.Split(line)
        self.ln   += 1

        if len(line.strip()) == 0: 
            return
        if match('^dihe', line.strip().lower()):
            self.dihe = True
            return
        elif match('^mass$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'MASS'
            return
        elif match('^bond$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'BONDS'
            return
        elif match('^angle$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'ANGLES'
            return
        elif match('^improper$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'IDIHS'
            return
        elif match('^nonbon$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'VDW'
            return
        elif len(s) == 0:
            self.dihe  = False
            return

        if self.dihe:
            try:
                self.itype = 'PDIHS%i' % int(np.abs(float(s[4])))
            except:
                self.itype = 'None'

        if self.itype in self.pdict:
            if 'Atom' in self.pdict[self.itype]:
                # List the atoms in the interaction.
                self.atom = [s[i].replace(" -","-") for i in self.pdict[self.itype]['Atom']]

            # The suffix of the parameter ID is built from the atom    #
            # types/classes involved in the interaction.
            self.suffix = ''.join(self.atom)

#=============================================================================================
# AMBER parmtop loader (from 'zander', by Randall J. Radmer)
#=============================================================================================

# A regex for extracting print format info from the FORMAT lines.
FORMAT_RE_PATTERN=re.compile("([0-9]+)([a-zA-Z]+)([0-9]+)\.?([0-9]*)")

# Pointer labels which map to pointer numbers at top of prmtop files
POINTER_LABELS  = """
              NATOM,  NTYPES, NBONH,  MBONA,  NTHETH, MTHETA,
              NPHIH,  MPHIA,  NHPARM, NPARM,  NEXT,   NRES,
              NBONA,  NTHETA, NPHIA,  NUMBND, NUMANG, NPTRA,
              NATYP,  NPHB,   IFPERT, NBPER,  NGPER,  NDPER,
              MBPER,  MGPER,  MDPER,  IFBOX,  NMXRS,  IFCAP
"""

# Pointer labels (above) as a list, not string.
POINTER_LABEL_LIST = POINTER_LABELS.replace(',', '').split()

class PrmtopLoader(object):
    """Parsed AMBER prmtop file.

    ParmtopLoader reads, parses and manages content from a AMBER prmtop file.

    EXAMPLES

    Parse a prmtop file of alanine dipeptide in implicit solvent.

    >>> import os, os.path
    >>> directory = os.path.join(os.getenv('YANK_INSTALL_DIR'), 'test', 'systems', 'alanine-dipeptide-gbsa')
    >>> prmtop_filename = os.path.join(directory, 'alanine-dipeptide.prmtop')
    >>> prmtop = PrmtopLoader(prmtop_filename)

    Parse a prmtop file of alanine dipeptide in explicit solvent.

    >>> import os, os.path
    >>> directory = os.path.join(os.getenv('YANK_INSTALL_DIR'), 'test', 'systems', 'alanine-dipeptide-explicit')
    >>> prmtop_filename = os.path.join(directory, 'alanine-dipeptide.prmtop')
    >>> prmtop = PrmtopLoader(prmtop_filename)    

    """
    def __init__(self, inFilename):
        """
        Create a PrmtopLoader object from an AMBER prmtop file.

        ARGUMENTS

        inFilename (string) - AMBER 'new-style' prmtop file, probably generated with one of the AMBER tleap/xleap/sleap        

        """

        self._prmtopVersion=None
        self._flags=[]
        self._raw_format={}
        self._raw_data={}

        fIn=open(inFilename)
        for line in fIn:
            if line.startswith('%VERSION'):
                tag, self._prmtopVersion = line.rstrip().split(None, 1)
            elif line.startswith('%FLAG'):
                tag, flag = line.rstrip().split(None, 1)
                self._flags.append(flag)
                self._raw_data[flag] = []
            elif line.startswith('%FORMAT'):
                format = line.rstrip()
                index0=format.index('(')
                index1=format.index(')')
                format = format[index0+1:index1]
                m = FORMAT_RE_PATTERN.search(format)
                self._raw_format[self._flags[-1]] = (format, m.group(1), m.group(2), m.group(3), m.group(4))
            elif self._flags \
                 and 'TITLE'==self._flags[-1] \
                 and not self._raw_data['TITLE']:
                self._raw_data['TITLE'] = line.rstrip()
            else:
                flag=self._flags[-1]
                (format, numItems, itemType,
                 itemLength, itemPrecision) = self._getFormat(flag)
                iLength=int(itemLength)
                line = line.rstrip()
                for index in range(0, len(line), iLength):
                    item = line[index:index+iLength]
                    if item:
                        self._raw_data[flag].append(item.strip())
        fIn.close()

    def _getFormat(self, flag=None):
        if not flag:
            flag=self._flags[-1]
        return self._raw_format[flag]

    def _getPointerValue(self, pointerLabel):
        """Return pointer value given pointer label

           Parameter:
            - pointerLabel: a string matching one of the following:

            NATOM  : total number of atoms 
            NTYPES : total number of distinct atom types
            NBONH  : number of bonds containing hydrogen
            MBONA  : number of bonds not containing hydrogen
            NTHETH : number of angles containing hydrogen
            MTHETA : number of angles not containing hydrogen
            NPHIH  : number of dihedrals containing hydrogen
            MPHIA  : number of dihedrals not containing hydrogen
            NHPARM : currently not used
            NPARM  : currently not used
            NEXT   : number of excluded atoms
            NRES   : number of residues
            NBONA  : MBONA + number of constraint bonds
            NTHETA : MTHETA + number of constraint angles
            NPHIA  : MPHIA + number of constraint dihedrals
            NUMBND : number of unique bond types
            NUMANG : number of unique angle types
            NPTRA  : number of unique dihedral types
            NATYP  : number of atom types in parameter file, see SOLTY below
            NPHB   : number of distinct 10-12 hydrogen bond pair types
            IFPERT : set to 1 if perturbation info is to be read in
            NBPER  : number of bonds to be perturbed
            NGPER  : number of angles to be perturbed
            NDPER  : number of dihedrals to be perturbed
            MBPER  : number of bonds with atoms completely in perturbed group
            MGPER  : number of angles with atoms completely in perturbed group
            MDPER  : number of dihedrals with atoms completely in perturbed groups
            IFBOX  : set to 1 if standard periodic box, 2 when truncated octahedral
            NMXRS  : number of atoms in the largest residue
            IFCAP  : set to 1 if the CAP option from edit was specified
        """
        index = POINTER_LABEL_LIST.index(pointerLabel) 
        return float(self._raw_data['POINTERS'][index])

    def getNumAtoms(self):
        """Return the number of atoms in the system"""
        return int(self._getPointerValue('NATOM'))

    def getNumTypes(self):
        """Return the number of AMBER atoms types in the system"""
        return int(self._getPointerValue('NTYPES'))

    def getIfBox(self):
        """Return True if the system was build with periodic boundary conditions (PBC)"""
        return int(self._getPointerValue('IFBOX'))

    def getIfCap(self):
        """Return True if the system was build with the cap option)"""
        return int(self._getPointerValue('IFCAP'))

    def getIfPert(self):
        """Return True if the system was build with the perturbation parameters)"""
        return int(self._getPointerValue('IFPERT'))

    def getMasses(self):
        """Return a list of atomic masses in the system"""
        try:
            return self._massList
        except AttributeError:
            pass

        self._massList=[]
        raw_masses=self._raw_data['MASS']
        for ii in range(self.getNumAtoms()):
            self._massList.append(float(raw_masses[ii]))
        self._massList = self._massList
        return self._massList

    def getCharges(self):
        """Return a list of atomic charges in the system"""
        try:
            return self._chargeList
        except AttributeError:
            pass

        self._chargeList=[]
        raw_charges=self._raw_data['CHARGE']
        for ii in range(self.getNumAtoms()):
            self._chargeList.append(float(raw_charges[ii])/18.2223)
        self._chargeList = self._chargeList
        return self._chargeList

    def getAtomName(self, iAtom):
        """Return the atom name for iAtom"""
        atomNames = self.getAtomNames()
        return atomNames[iAtom]

    def getAtomNames(self):
        """Return the list of the system atom names"""
        return self._raw_data['ATOM_NAME']

    def _getAtomTypeIndexes(self):
        try:
            return self._atomTypeIndexes
        except AttributeError:
            pass
        self._atomTypeIndexes=[]
        for atomTypeIndex in  self._raw_data['ATOM_TYPE_INDEX']:
            self._atomTypeIndexes.append(int(atomTypeIndex))
        return self._atomTypeIndexes

    def getAtomType(self, iAtom):
        """Return the AMBER atom type for iAtom"""
        atomTypes=self.getAtomTypes()
        return atomTypes[iAtom]

    def getAtomTypes(self):
        """Return the list of the AMBER atom types"""
        return self._raw_data['AMBER_ATOM_TYPE']

    def getResidueNumber(self, iAtom):
        """Return iAtom's residue number"""
        return self._getResiduePointer(iAtom)+1

    def getResidueLabel(self, iAtom=None, iRes=None):
        """Return residue label for iAtom OR iRes"""
        if iRes==None and iAtom==None:
            raise Exception("only specify iRes or iAtom, not both")
        if iRes!=None and iAtom!=None:
            raise Exception("iRes or iAtom must be set")
        if iRes!=None:
            return self._raw_data['RESIDUE_LABEL'][iRes]
        else:
            return self.getResidueLabel(iRes=self._getResiduePointer(iAtom))

    def _getResiduePointer(self, iAtom):
        try:
            return self.residuePointerDict[iAtom]
        except:
            pass
        self.residuePointerDict = {}
        resPointers=self._raw_data['RESIDUE_POINTER']
        firstAtom = [int(p)-1 for p in resPointers]
        firstAtom.append(self.getNumAtoms())
        res = 0
        for i in range(self.getNumAtoms()):
            while firstAtom[res+1] <= i:
                res += 1
            self.residuePointerDict[i] = res
        return self.residuePointerDict[iAtom]

    def getNonbondTerms(self):
        """Return list of all rVdw, epsilon pairs for each atom.  Work in the AMBER unit system."""
        try:
            return self._nonbondTerms
        except AttributeError:
            pass
        self._nonbondTerms=[]
        lengthConversionFactor = 1.0
        energyConversionFactor = 1.0
        for iAtom in range(self.getNumAtoms()):
            numTypes=self.getNumTypes()
            atomTypeIndexes=self._getAtomTypeIndexes()
            index=(numTypes+1)*(atomTypeIndexes[iAtom]-1)
            nbIndex=int(self._raw_data['NONBONDED_PARM_INDEX'][index])-1
            if nbIndex<0:
                raise Exception("10-12 interactions are not supported")
            acoef = float(self._raw_data['LENNARD_JONES_ACOEF'][nbIndex])
            bcoef = float(self._raw_data['LENNARD_JONES_BCOEF'][nbIndex])
            try:
                rMin = (2*acoef/bcoef)**(1/6.0)
                epsilon = 0.25*bcoef*bcoef/acoef
            except ZeroDivisionError:
                rMin = 1.0
                epsilon = 0.0
            rVdw = rMin/2.0*lengthConversionFactor
            epsilon = epsilon*energyConversionFactor
            self._nonbondTerms.append( (rVdw, epsilon) )
        return self._nonbondTerms

    def _getBonds(self, bondPointers):
        forceConstant=self._raw_data["BOND_FORCE_CONSTANT"]
        bondEquil=self._raw_data["BOND_EQUIL_VALUE"]
        returnList=[]
        forceConstConversionFactor = 1.0
        lengthConversionFactor = 1.0
        for ii in range(0,len(bondPointers),3):
             if int(bondPointers[ii])<0 or \
                int(bondPointers[ii+1])<0:
                 raise Exception("Found negative bonded atom pointers %s"
                                 % ((bondPointers[ii],
                                     bondPointers[ii+1]),))
             iType=int(bondPointers[ii+2])-1
             returnList.append((int(bondPointers[ii])/3,
                                int(bondPointers[ii+1])/3,
                                float(forceConstant[iType])*forceConstConversionFactor,
                                float(bondEquil[iType])*lengthConversionFactor))
        return returnList

    def getBondsWithH(self):
        """Return list of bonded atom pairs, K, and Rmin for each bond with a hydrogen"""
        try:
            return self._bondListWithH
        except AttributeError:
            pass
        bondPointers=self._raw_data["BONDS_INC_HYDROGEN"]
        self._bondListWithH = self._getBonds(bondPointers)
        return self._bondListWithH
        

    def getBondsNoH(self):
        """Return list of bonded atom pairs, K, and Rmin for each bond with no hydrogen"""
        try:
            return self._bondListNoH
        except AttributeError:
            pass
        bondPointers=self._raw_data["BONDS_WITHOUT_HYDROGEN"]
        self._bondListNoH = self._getBonds(bondPointers)
        return self._bondListNoH

    def getAngles(self):
        """Return list of atom triplets, K, and ThetaMin for each bond angle"""
        try:
            return self._angleList
        except AttributeError:
            pass
        forceConstant=self._raw_data["ANGLE_FORCE_CONSTANT"]
        angleEquil=self._raw_data["ANGLE_EQUIL_VALUE"]
        anglePointers = self._raw_data["ANGLES_INC_HYDROGEN"] \
                       +self._raw_data["ANGLES_WITHOUT_HYDROGEN"]
        self._angleList=[]
        forceConstConversionFactor = 1.0
        for ii in range(0,len(anglePointers),4):
             if int(anglePointers[ii])<0 or \
                int(anglePointers[ii+1])<0 or \
                int(anglePointers[ii+2])<0:
                 raise Exception("Found negative angle atom pointers %s"
                                 % ((anglePointers[ii],
                                     anglePointers[ii+1],
                                     anglePointers[ii+2]),))
             iType=int(anglePointers[ii+3])-1
             self._angleList.append((int(anglePointers[ii])/3,
                                int(anglePointers[ii+1])/3,
                                int(anglePointers[ii+2])/3,
                                float(forceConstant[iType])*forceConstConversionFactor,
                                float(angleEquil[iType])))
        return self._angleList

    def getDihedrals(self):
        """Return list of atom quads, K, phase and periodicity for each dihedral angle"""
        try:
            return self._dihedralList
        except AttributeError:
            pass
        forceConstant=self._raw_data["DIHEDRAL_FORCE_CONSTANT"]
        phase=self._raw_data["DIHEDRAL_PHASE"]
        periodicity=self._raw_data["DIHEDRAL_PERIODICITY"]
        dihedralPointers = self._raw_data["DIHEDRALS_INC_HYDROGEN"] \
                          +self._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
        self._dihedralList=[]
        forceConstConversionFactor = 1.0
        for ii in range(0,len(dihedralPointers),5):
             if int(dihedralPointers[ii])<0 or int(dihedralPointers[ii+1])<0:
                 raise Exception("Found negative dihedral atom pointers %s"
                                 % ((dihedralPointers[ii],
                                    dihedralPointers[ii+1],
                                    dihedralPointers[ii+2],
                                    dihedralPointers[ii+3]),))
             iType=int(dihedralPointers[ii+4])-1
             self._dihedralList.append((int(dihedralPointers[ii])/3,
                                int(dihedralPointers[ii+1])/3,
                                abs(int(dihedralPointers[ii+2]))/3,
                                abs(int(dihedralPointers[ii+3]))/3,
                                float(forceConstant[iType])*forceConstConversionFactor,
                                float(phase[iType]),
                                int(0.5+float(periodicity[iType]))))
        return self._dihedralList

    def get14Interactions(self):
        """Return list of atom pairs, chargeProduct, rMin and epsilon for each 1-4 interaction"""
        dihedralPointers = self._raw_data["DIHEDRALS_INC_HYDROGEN"] \
                          +self._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
        returnList=[]
        charges=self.getCharges()
        nonbondTerms = self.getNonbondTerms()
        for ii in range(0,len(dihedralPointers),5):
             if int(dihedralPointers[ii+2])>0 and int(dihedralPointers[ii+3])>0:
                 iAtom = int(dihedralPointers[ii])/3
                 lAtom = int(dihedralPointers[ii+3])/3
                 chargeProd = charges[iAtom]*charges[lAtom]
                 (rVdwI, epsilonI) = nonbondTerms[iAtom]
                 (rVdwL, epsilonL) = nonbondTerms[lAtom]
                 rMin = (rVdwI+rVdwL)
                 epsilon = math.sqrt(epsilonI*epsilonL)
                 returnList.append((iAtom, lAtom, chargeProd, rMin, epsilon))
        return returnList

    def getExcludedAtoms(self):
        """Return list of lists, giving all pairs of atoms that should have no non-bond interactions"""
        try:
            return self._excludedAtoms
        except AttributeError:
            pass
        self._excludedAtoms=[]
        numExcludedAtomsList=self._raw_data["NUMBER_EXCLUDED_ATOMS"]
        excludedAtomsList=self._raw_data["EXCLUDED_ATOMS_LIST"]
        total=0
        for iAtom in range(self.getNumAtoms()):
            index0=total
            n=int(numExcludedAtomsList[iAtom])
            total+=n
            index1=total
            atomList=[]
            for jAtom in excludedAtomsList[index0:index1]:
                j=int(jAtom)
                if j>0:
                    atomList.append(j-1)
            self._excludedAtoms.append(atomList)
        return self._excludedAtoms

    def getGBParms(self, symbls=None):
        """Return list giving GB params, Radius and screening factor"""
        try:
            return self._gb_List
        except AttributeError:
            pass
        self._gb_List=[]
        radii=self._raw_data["RADII"]
        screen=self._raw_data["SCREEN"]
        # Update screening parameters for GBn if specified
        if symbls:
            for (i, symbl) in enumerate(symbls):
                if symbl[0] == ('c' or 'C'):
                    screen[i] = 0.48435382330
                elif symbl[0] == ('h' or 'H'):
                    screen[i] = 1.09085413633
                elif symbl[0] == ('n' or 'N'):
                    screen[i] = 0.700147318409
                elif symbl[0] == ('o' or 'O'):
                    screen[i] = 1.06557401132
                elif symbl[0] == ('s' or 'S'):
                    screen[i] = 0.602256336067
                else:
                    screen[i] = 0.5
        lengthConversionFactor = 1.0
        for iAtom in range(len(radii)):
            self._gb_List.append((float(radii[iAtom])*lengthConversionFactor, float(screen[iAtom])))
        return self._gb_List

    def getBoxBetaAndDimensions(self):
        """Return periodic boundary box beta angle and dimensions"""
        beta=float(self._raw_data["BOX_DIMENSIONS"][0])
        x=float(self._raw_data["BOX_DIMENSIONS"][1])
        y=float(self._raw_data["BOX_DIMENSIONS"][2])
        z=float(self._raw_data["BOX_DIMENSIONS"][3])
        return (beta, x, y, z)

class AMBER(Engine):

    """ Engine for carrying out general purpose AMBER calculations. """

    def __init__(self, name="amber", **kwargs):
        ## Keyword args that aren't in this list are filtered out.
        self.valkwd = ['amberhome', 'pdb', 'mol2', 'frcmod', 'leapcmd', 'nbcut', 'reqpdb']
        super(AMBER,self).__init__(name=name, **kwargs)

    def setopts(self, **kwargs):
        
        """ Called by __init__ ; Set AMBER-specific options. """

        if 'amberhome' in kwargs:
            self.amberhome = kwargs['amberhome']
            if not os.path.exists(os.path.join(self.amberhome, "bin", "sander")):
                warn_press_key("The 'sander' executable indicated by %s doesn't exist! (Check amberhome)" \
                                   % os.path.join(self.amberhome,"bin","sander"))
        else:
            warn_once("The 'amberhome' option was not specified; using default.")
            if which('sander') == '':
                warn_press_key("Please add AMBER executables to the PATH or specify amberhome.")
            self.amberhome = os.path.split(which('sander'))[0]
        
        with wopen('.quit.leap') as f:
            print >> f, 'quit'

        self.nbcut = kwargs.get('nbcut', 9999)

        # AMBER search path
        self.spath = []
        for line in self.callamber('tleap -f .quit.leap'):
            if 'Adding' in line and 'to search path' in line:
                self.spath.append(line.split('Adding')[1].split()[0])
        os.remove('.quit.leap')

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory. """

        self.leapcmd = onefile(kwargs.get('leapcmd'), 'leap', err=True)
        self.absleap = os.path.abspath(self.leapcmd)

        # Name of the molecule, currently just call it a default name.
        self.mname = 'molecule'

        # Whether to throw an error if a PDB file doesn't exist.
        reqpdb = kwargs.get('reqpdb', 1)
        
        # Determine the PDB file name.  Amber could use this in tleap if it wants.
        # If 'pdb' is provided to Engine initialization, it will be used to 
        # copy over topology information (chain, atomname etc.).  If mol/coords
        # is not provided, then it will also provide the coordinates.
        pdbfnm = onefile(kwargs.get('pdb'), 'pdb' if reqpdb else None, err=reqpdb)

        # If the molecule object is provided as a keyword argument, it now
        # becomes an Engine attribute as well.  Otherwise, we create the
        # Engine.mol from the provided coordinates (if they exist).
        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        else:
            crdfile = None
            if 'coords' in kwargs:
                crdfile = onefile(kwargs.get('coords'), None, err=True)
            elif pdbfnm is not None:
                crdfile = pdbfnm
            if crdfile is None:
                logger.error("Cannot find a coordinate file to use\n")
                raise RuntimeError
            self.mol = Molecule(crdfile, top=pdbfnm, build_topology=False)

            
        # If a .pdb was not provided, we create one.
        if pdbfnm is None:
            pdbfnm = self.name + ".pdb"
            # AMBER doesn't always like the CONECT records
            self.mol[0].write(pdbfnm, write_conect=False)
        self.abspdb = os.path.abspath(pdbfnm)

        # Write the PDB that AMBER is going to read in.
        # This may or may not be identical to the one used to initialize the engine.
        # self.mol.write('%s.pdb' % self.name)
        # self.abspdb = os.path.abspath('%s.pdb' % self.name)

    def callamber(self, command, stdin=None, print_to_screen=False, print_command=False, **kwargs):

        """ Call TINKER; prepend the amberhome to calling the TINKER program. """

        csplit = command.split()
        # Sometimes the engine changes dirs and the inpcrd/prmtop go missing, so we link it.
        # Prepend the AMBER path to the program call.
        prog = os.path.join(self.amberhome, "bin", csplit[0])
        csplit[0] = prog
        # No need to catch exceptions since failed AMBER calculations will return nonzero exit status.
        o = _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, rbytes=1024, **kwargs)
        return o

    def leap(self, name=None, delcheck=False, read_prmtop=True):
        if not os.path.exists(self.leapcmd):
            LinkFile(self.absleap, self.leapcmd)
        pdb = os.path.basename(self.abspdb)
        if not os.path.exists(pdb):
            LinkFile(self.abspdb, pdb)
        if name is None: name = self.name
        write_leap(self.leapcmd, mol2=self.mol2, frcmod=self.frcmod, pdb=pdb, prefix=name, spath=self.spath, delcheck=delcheck)
        self.callamber("tleap -f %s_" % self.leapcmd)
        if read_prmtop:
            prmtop = PrmtopLoader('%s.prmtop' % name)
            self.AtomLists['Charge'] = prmtop.getCharges()

    def get_charges(self):
        self.leap(read_prmtop=True)
        return np.array(self.AtomLists['Charge'])

    def prepare(self, pbc=False, **kwargs):

        """ Called by __init__ ; prepare the temp directory and figure out the topology. """

        self.AtomLists = defaultdict(list)

        if hasattr(self,'FF'):
            if not (os.path.exists(self.FF.amber_frcmod) and os.path.exists(self.FF.amber_mol2)):
                # If the parameter files don't already exist, create them for the purpose of
                # preparing the engine, but then delete them afterward.
                prmtmp = True
                self.FF.make(np.zeros(self.FF.np))
            # Currently force field object only allows one mol2 and frcmod file although this can be lifted.
            self.mol2 = [self.FF.amber_mol2]
            self.frcmod = [self.FF.amber_frcmod]
            if 'mol2' in kwargs:
                logger.error("FF object is provided, which overrides mol2 keyword argument")
                raise RuntimeError
            if 'frcmod' in kwargs:
                logger.error("FF object is provided, which overrides frcmod keyword argument")
                raise RuntimeError
        else:
            prmtmp = False
            self.mol2 = listfiles(kwargs.get('mol2'), 'mol2', err=True)
            self.frcmod = listfiles(kwargs.get('frcmod'), 'frcmod', err=True)

        # Figure out the topology information.
        self.leap()
        
        # This is done by "rdparm" although I could actually use the PrmtopLoader now.
        # (Code written before I knew about the PrmtopLoader)
        o = self.callamber("rdparm %s.prmtop" % self.name, 
                           stdin="printAtoms\nprintBonds\nexit\n", 
                           persist=True, print_error=False)

        # Once we do this, we don't need the prmtop and inpcrd anymore
        os.unlink("%s.inpcrd" % self.name)
        os.unlink("%s.prmtop" % self.name)
        os.unlink("leap.log")

        mode = 'None'
        G = nx.Graph()
        for line in o:
            s = line.split()
            if 'Atom' in line:
                mode = 'Atom'
            elif 'Bond' in line:
                mode = 'Bond'
            elif 'RDPARM MENU' in line:
                continue
            elif 'EXITING' in line:
                break
            elif len(s) == 0:
                continue
            elif mode == 'Atom':
                # Based on parsing lines like these:
                """
   327:  HA   -0.01462  1.0 (  23:HIP )  H1    E   
   328:  CB   -0.33212 12.0 (  23:HIP )  CT    3   
   329:  HB2   0.10773  1.0 (  23:HIP )  HC    E   
   330:  HB3   0.10773  1.0 (  23:HIP )  HC    E   
   331:  CG    0.18240 12.0 (  23:HIP )  CC    B   
                """
                # Based on variable width fields.
                atom_number = int(line.split(':')[0])
                atom_name = line.split()[1]
                atom_charge = float(line.split()[2])
                atom_mass = float(line.split()[3])
                rnn = line.split('(')[1].split(')')[0].split(':')
                residue_number = int(rnn[0])
                residue_name = rnn[1]
                atom_type = line.split(')')[1].split()[0]
                self.AtomLists['Name'].append(atom_name)
                self.AtomLists['Charge'].append(atom_charge)
                self.AtomLists['Mass'].append(atom_mass)
                self.AtomLists['ResidueNumber'].append(residue_number)
                self.AtomLists['ResidueName'].append(residue_name)
                # Not sure if this works
                G.add_node(atom_number)
            elif mode == 'Bond':
                a, b = (int(i) for i in (line.split('(')[1].split(')')[0].split(',')))
                G.add_edge(a, b)

        self.AtomMask = [a == 'A' for a in self.AtomLists['ParticleType']]

        # Use networkx to figure out a list of molecule numbers.
        # gs = nx.connected_component_subgraphs(G)
        # tmols = [gs[i] for i in np.argsort(np.array([min(g.nodes()) for g in gs]))]
        # mnodes = [m.nodes() for m in tmols]
        # self.AtomLists['MoleculeNumber'] = [[i+1 in m for m in mnodes].index(1) for i in range(self.mol.na)]

        ## Write out the trajectory coordinates to a .mdcrd file.

        # I also need to write the trajectory
        if 'boxes' in self.mol.Data.keys():
            logger.info("\x1b[91mWriting %s-all.crd file with no periodic box information\x1b[0m\n" % self.name)
            del self.mol.Data['boxes']

        if hasattr(self, 'target') and hasattr(self.target,'shots'):
            self.qmatoms = target.qmatoms
            self.mol.write("%s-all.crd" % self.name, select=range(self.target.shots), ftype="mdcrd")
        else:
            self.qmatoms = self.mol.na
            self.mol.write("%s-all.crd" % self.name, ftype="mdcrd")

        if prmtmp:
            for f in self.FF.fnms: 
                os.unlink(f)

    def evaluate_(self, crdin, force=False):

        """ 
        Utility function for computing energy and forces using AMBER. 
        
        Inputs:
        crdin: AMBER .mdcrd file name.
        force: Switch for parsing the force. (Currently it always calculates the forces.)

        Outputs:
        Result: Dictionary containing energies (and optionally) forces.
        """

        force_mdin="""Loop over conformations and compute energy and force (use ioutfnm=1 for netcdf, ntb=0 for no box)
&cntrl
imin = 5, ntb = 0, cut={cut}, nstlim = 0, nsnb = 0
/
&debugf
do_debugf = 1, dumpfrc = 1
/
"""
        with wopen("%s-force.mdin" % self.name) as f:
            print >> f, force_mdin.format(cut="%i" % int(self.nbcut))

        ## This line actually runs AMBER.
        self.leap(delcheck=True)
        self.callamber("sander -i %s-force.mdin -o %s-force.mdout -p %s.prmtop -c %s.inpcrd -y %s -O" % 
                       (self.name, self.name, self.name, self.name, crdin))
        ParseMode = 0
        Result = {}
        Energies = []
        Forces = []
        Force = []
        for line in open('forcedump.dat'):
            line = line.strip()
            sline = line.split()
            if ParseMode == 1:
                if len(sline) == 1 and isfloat(sline[0]):
                    Energies.append(float(sline[0]) * 4.184)
                    ParseMode = 0
            if ParseMode == 2:
                if len(sline) == 3 and all(isfloat(sline[i]) for i in range(3)):
                    Force += [float(sline[i]) * 4.184 * 10 for i in range(3)]
                if len(Force) == 3*self.qmatoms:
                    Forces.append(np.array(Force))
                    Force = []
                    ParseMode = 0
            if line == '0 START of Energies':
                ParseMode = 1
            elif line == '1 Total Force':
                ParseMode = 2

        Result["Energy"] = np.array(Energies[1:])
        Result["Force"] = np.array(Forces[1:])
        return Result

    def energy_force_one(self, shot):
        
        """ Computes the energy and force using AMBER for one snapshot. """
        
        self.mol[shot].write("%s.mdcrd" % self.name, ftype="mdcrd")
        Result = self.evaluate_("%s.mdcrd" % self.name, force=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy(self):

        """ Computes the energy using AMBER over a trajectory. """

        if hasattr(self, 'md_trajectory'): 
            x = self.md_trajectory
        else:
            x = "%s-all.crd" % self.name
            self.mol.write(x, ftype="mdcrd")
        return self.evaluate_(x)["Energy"]

    def energy_force(self):

        """ Computes the energy and force using AMBER over a trajectory. """

        if hasattr(self, 'md_trajectory') : 
            x = self.md_trajectory
        else:
            x = "%s-all.crd" % self.name
        Result = self.evaluate_(x, force=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy_dipole(self):

        """ Computes the energy and dipole using TINKER over a trajectory. """

        logger.error('Dipole moments are not yet implemented in AMBER interface')
        raise NotImplementedError

        if hasattr(self, 'md_trajectory') : 
            x = self.md_trajectory
        else:
            x = "%s.xyz" % self.name
            self.mol.write(x, ftype="tinker")
        Result = self.evaluate_(x, dipole=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Dipole"]))

    def optimize(self, shot=0, method="newton", crit=1e-4):

        """ Optimize the geometry and align the optimized geometry to the starting geometry. """

        logger.error('Geometry optimizations are not yet implemented in AMBER interface')
        raise NotImplementedError
    
        # Code from tinkerio.py
        if os.path.exists('%s.xyz_2' % self.name):
            os.unlink('%s.xyz_2' % self.name)
        self.mol[shot].write('%s.xyz' % self.name, ftype="tinker")
        if method == "newton":
            if self.rigid: optprog = "optrigid"
            else: optprog = "optimize"
        elif method == "bfgs":
            if self.rigid: optprog = "minrigid"
            else: optprog = "minimize"
        o = self.calltinker("%s %s.xyz %f" % (optprog, self.name, crit))
        # Silently align the optimized geometry.
        M12 = Molecule("%s.xyz" % self.name, ftype="tinker") + Molecule("%s.xyz_2" % self.name, ftype="tinker")
        if not self.pbc:
            M12.align(center=False)
        M12[1].write("%s.xyz_2" % self.name, ftype="tinker")
        rmsd = M12.ref_rmsd(0)[1]
        cnvgd = 0
        mode = 0
        for line in o:
            s = line.split()
            if len(s) == 0: continue
            if "Optimally Conditioned Variable Metric Optimization" in line: mode = 1
            if "Limited Memory BFGS Quasi-Newton Optimization" in line: mode = 1
            if mode == 1 and isint(s[0]): mode = 2
            if mode == 2:
                if isint(s[0]): E = float(s[1])
                else: mode = 0
            if "Normal Termination" in line:
                cnvgd = 1
        if not cnvgd:
            for line in o:
                logger.info(str(line) + '\n')
            logger.info("The minimization did not converge in the geometry optimization - printout is above.\n")
        return E, rmsd

    def normal_modes(self, shot=0, optimize=True):
        self.leap(delcheck=True)
        if optimize:
            # Copied from AMBER tests folder.
            opt_temp = """  Newton-Raphson minimization
 &data
     ntrun = 4, nsave=20, ndiag=2, cut={cut}
     nprint=1, ioseen=0,
     drms = 0.000001, maxcyc=4000, bdwnhl=0.1, dfpred = 0.1,
     scnb=2.0, scee=1.2, idiel=1,
 /
"""
            with wopen("%s-nr.in" % self.name) as f: print >> f, opt_temp.format(cut="%i" % self.nbcut)
            self.callamber("nmode -O -i %s-nr.in -c %s.inpcrd -p %s.prmtop -r %s.rst -o %s-nr.out" % (self.name, self.name, self.name, self.name, self.name))
        nmode_temp = """  normal modes
 &data
     ntrun = 1, nsave=20, ndiag=2, cut={cut}
     nprint=1, ioseen=0,
     drms = 0.0001, maxcyc=1, bdwnhl=1.1, dfpred = 0.1,
     scnb=2.0, scee=1.2, idiel=1,
     nvect={nvect}, eta=0.9, ivform=2,
 /
"""
        with wopen("%s-nmode.in" % self.name) as f: print >> f, nmode_temp.format(cut="%i" % self.nbcut, nvect=3*self.mol.na)
        self.callamber("nmode -O -i %s-nmode.in -c %s.rst -p %s.prmtop -v %s-vecs.out -o %s-vibs.out" % (self.name, self.name, self.name, self.name, self.name))
        # My attempt at parsing AMBER frequency output.
        vmode = 0
        ln = 0
        freqs = []
        modeA = []
        modeB = []
        modeC = []
        vecs = []
        for line in open("%s-vecs.out" % self.name).readlines():
            if line.strip() == "$FREQ":
                vmode = 1
            elif line.strip().startswith("$"):
                vmode = 0
            elif vmode == 1: 
                # We are in the vibrational block now.
                if ln == 0: pass
                elif ln == 1:
                    freqs += [float(i) for i in line.split()]
                else:
                    modeA.append([float(i) for i in line.split()[0:3]])
                    modeB.append([float(i) for i in line.split()[3:6]])
                    modeC.append([float(i) for i in line.split()[6:9]])
                    if len(modeA) == self.mol.na:
                        vecs.append(modeA)
                        vecs.append(modeB)
                        vecs.append(modeC)
                        modeA = []
                        modeB = []
                        modeC = []
                        ln = -1
                ln += 1
        calc_eigvals = np.array(freqs)
        calc_eigvecs = np.array(vecs)
        # Sort by frequency absolute value and discard the six that are closest to zero
        calc_eigvecs = calc_eigvecs[np.argsort(np.abs(calc_eigvals))][6:]
        calc_eigvals = calc_eigvals[np.argsort(np.abs(calc_eigvals))][6:]
        # Sort again by frequency
        calc_eigvecs = calc_eigvecs[np.argsort(calc_eigvals)]
        calc_eigvals = calc_eigvals[np.argsort(calc_eigvals)]
        os.system("rm -rf *.xyz_* *.[0-9][0-9][0-9]")
        return calc_eigvals, calc_eigvecs

    def multipole_moments(self, shot=0, optimize=True, polarizability=False):

        logger.error('Multipole moments are not yet implemented in AMBER interface')
        raise NotImplementedError

        """ Return the multipole moments of the 1st snapshot in Debye and Buckingham units. """
        # This line actually runs TINKER
        if optimize:
            self.optimize(shot, crit=1e-6)
            o = self.calltinker("analyze %s.xyz_2 M" % (self.name))
        else:
            self.mol[shot].write('%s.xyz' % self.name, ftype="tinker")
            o = self.calltinker("analyze %s.xyz M" % (self.name))
        # Read the TINKER output.
        qn = -1
        ln = 0
        for line in o:
            s = line.split()
            if "Dipole X,Y,Z-Components" in line:
                dipole_dict = OrderedDict(zip(['x','y','z'], [float(i) for i in s[-3:]]))
            elif "Quadrupole Moment Tensor" in line:
                qn = ln
                quadrupole_dict = OrderedDict([('xx',float(s[-3]))])
            elif qn > 0 and ln == qn + 1:
                quadrupole_dict['xy'] = float(s[-3])
                quadrupole_dict['yy'] = float(s[-2])
            elif qn > 0 and ln == qn + 2:
                quadrupole_dict['xz'] = float(s[-3])
                quadrupole_dict['yz'] = float(s[-2])
                quadrupole_dict['zz'] = float(s[-1])
            ln += 1
        calc_moments = OrderedDict([('dipole', dipole_dict), ('quadrupole', quadrupole_dict)])
        if polarizability:
            if optimize:
                o = self.calltinker("polarize %s.xyz_2" % (self.name))
            else:
                o = self.calltinker("polarize %s.xyz" % (self.name))
            # Read the TINKER output.
            pn = -1
            ln = 0
            polarizability_dict = OrderedDict()
            for line in o:
                s = line.split()
                if "Total Polarizability Tensor" in line:
                    pn = ln
                elif pn > 0 and ln == pn + 2:
                    polarizability_dict['xx'] = float(s[-3])
                    polarizability_dict['yx'] = float(s[-2])
                    polarizability_dict['zx'] = float(s[-1])
                elif pn > 0 and ln == pn + 3:
                    polarizability_dict['xy'] = float(s[-3])
                    polarizability_dict['yy'] = float(s[-2])
                    polarizability_dict['zy'] = float(s[-1])
                elif pn > 0 and ln == pn + 4:
                    polarizability_dict['xz'] = float(s[-3])
                    polarizability_dict['yz'] = float(s[-2])
                    polarizability_dict['zz'] = float(s[-1])
                ln += 1
            calc_moments['polarizability'] = polarizability_dict
        os.system("rm -rf *.xyz_* *.[0-9][0-9][0-9]")
        return calc_moments

    def energy_rmsd(self, shot=0, optimize=True):

        """ Calculate energy of the selected structure (optionally minimize and return the minimized energy and RMSD). In kcal/mol. """

        logger.error('Geometry optimization is not yet implemented in AMBER interface')
        raise NotImplementedError

        rmsd = 0.0
        # This line actually runs TINKER
        # xyzfnm = sysname+".xyz"
        if optimize:
            E_, rmsd = self.optimize(shot)
            o = self.calltinker("analyze %s.xyz_2 E" % self.name)
            #----
            # Two equivalent ways to get the RMSD, here for reference.
            #----
            # M1 = Molecule("%s.xyz" % self.name, ftype="tinker")
            # M2 = Molecule("%s.xyz_2" % self.name, ftype="tinker")
            # M1 += M2
            # rmsd = M1.ref_rmsd(0)[1]
            #----
            # oo = self.calltinker("superpose %s.xyz %s.xyz_2 1 y u n 0" % (self.name, self.name))
            # for line in oo:
            #     if "Root Mean Square Distance" in line:
            #         rmsd = float(line.split()[-1])
            #----
            os.system("rm %s.xyz_2" % self.name)
        else:
            o = self.calltinker("analyze %s.xyz E" % self.name)
        # Read the TINKER output. 
        E = None
        for line in o:
            if "Total Potential Energy" in line:
                E = float(line.split()[-2].replace('D','e'))
        if E is None:
            logger.error("Total potential energy wasn't encountered when calling analyze!\n")
            raise RuntimeError
        if optimize and abs(E-E_) > 0.1:
            warn_press_key("Energy from optimize and analyze aren't the same (%.3f vs. %.3f)" % (E, E_))
        return E, rmsd

    def interaction_energy(self, fraga, fragb):
        
        """ Calculate the interaction energy for two fragments. """

        self.A = AMBER(name="A", mol=self.mol.atom_select(fraga), amberhome=self.amberhome, leapcmd=self.leapcmd, mol2=self.mol2, frcmod=self.frcmod, reqpdb=False)
        self.B = AMBER(name="B", mol=self.mol.atom_select(fragb), amberhome=self.amberhome, leapcmd=self.leapcmd, mol2=self.mol2, frcmod=self.frcmod, reqpdb=False)

        # Interaction energy needs to be in kcal/mol.
        return (self.energy() - self.A.energy() - self.B.energy()) / 4.184

    def molecular_dynamics(self, nsteps, timestep, temperature=None, pressure=None, nequil=0, nsave=1000, minimize=True, anisotropic=False, threads=1, verbose=False, **kwargs):
        
        """
        Method for running a molecular dynamics simulation.  

        Required arguments:
        nsteps      = (int)   Number of total time steps
        timestep    = (float) Time step in FEMTOSECONDS
        temperature = (float) Temperature control (Kelvin)
        pressure    = (float) Pressure control (atmospheres)
        nequil      = (int)   Number of additional time steps at the beginning for equilibration
        nsave       = (int)   Step interval for saving and printing data
        minimize    = (bool)  Perform an energy minimization prior to dynamics
        threads     = (int)   Specify how many OpenMP threads to use

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        """

        logger.error('Molecular dynamics not yet implemented in AMBER interface')
        raise NotImplementedError

        md_defs = OrderedDict()
        md_opts = OrderedDict()
        # Print out averages only at the end.
        md_opts["printout"] = nsave
        md_opts["openmp-threads"] = threads
        # Langevin dynamics for temperature control.
        if temperature is not None:
            md_defs["integrator"] = "stochastic"
        else:
            md_defs["integrator"] = "beeman"
            md_opts["thermostat"] = None
        # Periodic boundary conditions.
        if self.pbc:
            md_opts["vdw-correction"] = ''
            if temperature is not None and pressure is not None: 
                md_defs["integrator"] = "beeman"
                md_defs["thermostat"] = "bussi"
                md_defs["barostat"] = "montecarlo"
                if anisotropic:
                    md_opts["aniso-pressure"] = ''
            elif pressure is not None:
                warn_once("Pressure is ignored because temperature is turned off.")
        else:
            if pressure is not None:
                warn_once("Pressure is ignored because pbc is set to False.")
            # Use stochastic dynamics for the gas phase molecule.
            # If we use the regular integrators it may miss
            # six degrees of freedom in calculating the kinetic energy.
            md_opts["barostat"] = None

        eq_opts = deepcopy(md_opts)
        if self.pbc and temperature is not None and pressure is not None: 
            eq_opts["integrator"] = "beeman"
            eq_opts["thermostat"] = "bussi"
            eq_opts["barostat"] = "berendsen"

        if minimize:
            if verbose: logger.info("Minimizing the energy...")
            self.optimize(method="bfgs", crit=1)
            os.system("mv %s.xyz_2 %s.xyz" % (self.name, self.name))
            if verbose: logger.info("Done\n")

        # Run equilibration.
        if nequil > 0:
            write_key("%s-eq.key" % self.name, eq_opts, "%s.key" % self.name, md_defs)
            if verbose: printcool("Running equilibration dynamics", color=0)
            if self.pbc and pressure is not None:
                self.calltinker("dynamic %s -k %s-eq %i %f %f 4 %f %f" % (self.name, self.name, nequil, timestep, float(nsave*timestep)/1000, 
                                                                          temperature, pressure), print_to_screen=verbose)
            else:
                self.calltinker("dynamic %s -k %s-eq %i %f %f 2 %f" % (self.name, self.name, nequil, timestep, float(nsave*timestep)/1000,
                                                                       temperature), print_to_screen=verbose)
            os.system("rm -f %s.arc" % (self.name))

        # Run production.
        if verbose: printcool("Running production dynamics", color=0)
        write_key("%s-md.key" % self.name, md_opts, "%s.key" % self.name, md_defs)
        if self.pbc and pressure is not None:
            odyn = self.calltinker("dynamic %s -k %s-md %i %f %f 4 %f %f" % (self.name, self.name, nsteps, timestep, float(nsave*timestep/1000), 
                                                                             temperature, pressure), print_to_screen=verbose)
        else:
            odyn = self.calltinker("dynamic %s -k %s-md %i %f %f 2 %f" % (self.name, self.name, nsteps, timestep, float(nsave*timestep/1000), 
                                                                          temperature), print_to_screen=verbose)
            
        # Gather information.
        os.system("mv %s.arc %s-md.arc" % (self.name, self.name))
        self.md_trajectory = "%s-md.arc" % self.name
        edyn = []
        kdyn = []
        temps = []
        for line in odyn:
            s = line.split()
            if 'Current Potential' in line:
                edyn.append(float(s[2]))
            if 'Current Kinetic' in line:
                kdyn.append(float(s[2]))
            if len(s) > 0 and s[0] == 'Temperature' and s[2] == 'Kelvin':
                temps.append(float(s[1]))

        # Potential and kinetic energies converted to kJ/mol.
        edyn = np.array(edyn) * 4.184
        kdyn = np.array(kdyn) * 4.184
        temps = np.array(temps)
    
        if verbose: logger.info("Post-processing to get the dipole moments\n")
        oanl = self.calltinker("analyze %s-md.arc" % self.name, stdin="G,E,M", print_to_screen=False)

        # Read potential energy and dipole from file.
        eanl = []
        dip = []
        mass = 0.0
        ecomp = OrderedDict()
        havekeys = set()
        first_shot = True
        for ln, line in enumerate(oanl):
            strip = line.strip()
            s = line.split()
            if 'Total System Mass' in line:
                mass = float(s[-1])
            if 'Total Potential Energy : ' in line:
                eanl.append(float(s[4]))
            if 'Dipole X,Y,Z-Components :' in line:
                dip.append([float(s[i]) for i in range(-3,0)])
            if first_shot:
                for key in eckeys:
                    if strip.startswith(key):
                        if key in ecomp:
                            ecomp[key].append(float(s[-2])*4.184)
                        else:
                            ecomp[key] = [float(s[-2])*4.184]
                        if key in havekeys:
                            first_shot = False
                        havekeys.add(key)
            else:
                for key in havekeys:
                    if strip.startswith(key):
                        if key in ecomp:
                            ecomp[key].append(float(s[-2])*4.184)
                        else:
                            ecomp[key] = [float(s[-2])*4.184]
        for key in ecomp:
            ecomp[key] = np.array(ecomp[key])
        ecomp["Potential Energy"] = edyn
        ecomp["Kinetic Energy"] = kdyn
        ecomp["Temperature"] = temps
        ecomp["Total Energy"] = edyn+kdyn

        # Energies in kilojoules per mole
        eanl = np.array(eanl) * 4.184
        # Dipole moments in debye
        dip = np.array(dip)
        # Volume of simulation boxes in cubic nanometers
        # Conversion factor derived from the following:
        # In [22]: 1.0 * gram / mole / (1.0 * nanometer)**3 / AVOGADRO_CONSTANT_NA / (kilogram/meter**3)
        # Out[22]: 1.6605387831627252
        conv = 1.6605387831627252
        if self.pbc:
            vol = np.array([BuildLatticeFromLengthsAngles(*[float(j) for j in line.split()]).V \
                                for line in open("%s-md.arc" % self.name).readlines() \
                                if (len(line.split()) == 6 and isfloat(line.split()[1]) \
                                        and all([isfloat(i) for i in line.split()[:6]]))]) / 1000
            rho = conv * mass / vol
        else:
            vol = None
            rho = None
        prop_return = OrderedDict()
        prop_return.update({'Rhos': rho, 'Potentials': edyn, 'Kinetics': kdyn, 'Volumes': vol, 'Dips': dip, 'Ecomps': ecomp})
        return prop_return

class AbInitio_AMBER(AbInitio):

    """Subclass of Target for force and energy matching
    using AMBER."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Coordinate file.
        self.set_option(tgt_opts, 'coords')
        ## PDB file for topology (if different from coordinate file.)
        self.set_option(tgt_opts, 'pdb')
        ## AMBER home directory.
        self.set_option(options, 'amberhome')
        ## AMBER home directory.
        self.set_option(tgt_opts, 'amber_leapcmd', 'leapcmd')
        ## Nonbonded cutoff for AMBER (pacth).
        self.set_option(options, 'amber_nbcut', 'nbcut')
        ## Name of the engine.
        self.engine_ = AMBER
        ## Initialize base class.
        super(AbInitio_AMBER,self).__init__(options,tgt_opts,forcefield)

class Interaction_AMBER(Interaction):

    """Subclass of Target for calculating and matching ineraction energies
    using AMBER.  """

    def __init__(self,options,tgt_opts,forcefield):
        ## Coordinate file.
        self.set_option(tgt_opts, 'coords')
        ## PDB file for topology (if different from coordinate file.)
        self.set_option(tgt_opts, 'pdb')
        ## AMBER home directory.
        self.set_option(options, 'amberhome')
        ## AMBER home directory.
        self.set_option(tgt_opts, 'amber_leapcmd', 'leapcmd')
        ## Nonbonded cutoff for AMBER (pacth).
        self.set_option(options, 'amber_nbcut', 'nbcut')
        ## Name of the engine.
        self.engine_ = AMBER
        ## Initialize base class.
        super(Interaction_AMBER,self).__init__(options,tgt_opts,forcefield)

class Vibration_AMBER(Vibration):

    """Subclass of Target for calculating and matching vibrational modes using AMBER. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Coordinate file.
        self.set_option(tgt_opts, 'coords')
        ## PDB file for topology (if different from coordinate file.)
        self.set_option(tgt_opts, 'pdb')
        ## AMBER home directory.
        self.set_option(options, 'amberhome')
        ## AMBER home directory.
        self.set_option(tgt_opts, 'amber_leapcmd', 'leapcmd')
        ## Nonbonded cutoff for AMBER (pacth).
        self.set_option(options, 'amber_nbcut', 'nbcut')
        ## Name of the engine.
        self.engine_ = AMBER
        ## Initialize base class.
        super(Vibration_AMBER,self).__init__(options,tgt_opts,forcefield)
