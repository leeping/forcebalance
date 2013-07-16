#======================================================================#
#|                                                                    |#
#|              Chemical file format conversion module                |#
#|                                                                    |#
#|                Lee-Ping Wang (leeping@stanford.edu)                |#
#|                   Last updated July 10, 2013                       |#
#|                                                                    |#
#|               [ IN PROGRESS, USE AT YOUR OWN RISK ]                |#
#|                                                                    |#
#|   This is free software released under version 2 of the GNU GPL,   |#
#|   please use or redistribute as you see fit under the terms of     |#
#|   this license. (http://www.gnu.org/licenses/gpl-2.0.html)         |#
#|                                                                    |#
#|   This program is distributed in the hope that it will be useful,  |#
#|   but without any warranty; without even the implied warranty of   |#
#|   merchantability or fitness for a particular purpose.  See the    |#
#|   GNU General Public License for more details.                     |#
#|                                                                    |#
#|   Feedback and suggestions are encouraged.                         |#
#|                                                                    |#
#|   What this is for:                                                |#
#|   Converting a molecule between file formats                       |#
#|   Loading and processing of trajectories                           |#
#|   (list of geometries for the same set of atoms)                   |#
#|   Concatenating or slicing trajectories                            |#
#|   Combining molecule metadata (charge, Q-Chem rem variables)       |#
#|                                                                    |#
#|   Supported file formats:                                          |#
#|   See the __init__ method in the Molecule class.                   |#
#|                                                                    |#
#|   Note to self / developers:                                       |#
#|   Please make this file as standalone as possible                  |#
#|   (i.e. don't introduce dependencies).  If we load an external     |#
#|   library to parse a file, do so with 'try / except' so that       |#
#|   the module is still usable even if certain parts are missing.    |#
#|   It's better to be like a Millennium Falcon. :P                   |#
#|                                                                    |#
#|   Please make sure this file is up-to-date in                      |#
#|   both the 'leeping' and 'forcebalance' modules                    |#
#|                                                                    |#
#|   At present, when I perform operations like adding two objects,   |#
#|   the sum is created from deep copies of data members in the       |#
#|   originals. This is because copying by reference is confusing;    |#
#|   suppose if I do B += A and modify something in B; it should not  |#
#|   change in A.                                                     |#
#|                                                                    |#
#|   A consequence of this is that data members should not be too     |#
#|   complicated; they should be things like lists or dicts, and NOT  |#
#|   contain references to themselves.                                |#
#|                                                                    |#
#|   To-do list: Handling of comments is still not very good.         |#
#|   Comments from previous files should be 'passed on' better.       |#
#|                                                                    |#
#|              Contents of this file:                                |#
#|              0) Names of data variables                            |#
#|              1) Imports                                            |#
#|              2) Subroutines                                        |#
#|              3) Molecule class                                     |#
#|                a) Class customizations (add, getitem)              |#
#|                b) Instantiation                                    |#
#|                c) Core functionality (read, write)                 |#
#|                d) Reading functions                                |#
#|                e) Writing functions                                |#
#|                f) Extra stuff                                      |#
#|              4) "main" function (if executed)                      |#
#|                                                                    |#
#|                   Required: Python 2.7, Numpy 1.6                  |#
#|                   Optional: Mol2, PDB, DCD readers                 |#
#|                    (can be found in ForceBalance)                  |#
#|                    NetworkX package (for topologies)               |#
#|                                                                    |#
#|             Thanks: Todd Dolinsky, Yong Huang,                     |#
#|                     Kyle Beauchamp (PDB)                           |#
#|                     John Stone (DCD Plugin)                        |#
#|                     Pierre Tuffery (Mol2 Plugin)                   |#
#|                     #python IRC chat on FreeNode                   |#
#|                                                                    |#
#|             Instructions:                                          |#
#|                                                                    |#
#|               To import:                                           |#
#|                 from molecule import Molecule                      |#
#|               To create a Molecule object:                         |#
#|                 MyMol = Molecule(fnm)                              |#
#|               To convert to a new file format:                     |#
#|                 MyMol.write('newfnm.format')                       |#
#|               To concatenate geometries:                           |#
#|                 MyMol += MyMolB                                    |#
#|                                                                    |#
#======================================================================#

#=========================================#
#|     DECLARE VARIABLE NAMES HERE       |#
#|                                       |#
#|  Any member variable in the Molecule  |#
#| class must be declared here otherwise |#
#| the Molecule class won't recognize it |#
#=========================================#
#| Data attributes in FrameVariableNames |#
#| must be a list along the frame axis,  |#
#| and they must have the same length.   |#
#=========================================#
# xyzs       = List of numpy arrays of atomic xyz coordinates
# comms      = List of comment strings
# boxes      = List of 3-element or 9-element arrays for periodic boxes
# qm_forces  = List of numpy arrays of atomistic forces from QM calculations
# qm_espxyzs = List of numpy arrays of xyz coordinates for ESP evaluation
# qm_espvals = List of numpy arrays of ESP values
FrameVariableNames = set(['xyzs', 'comms', 'boxes', 'qm_forces', 'qm_energies', 'qm_interaction', 
                          'qm_espxyzs', 'qm_espvals', 'qm_extchgs', 'qm_mulliken_charges', 'qm_mulliken_spins'])
#=========================================#
#| Data attributes in AtomVariableNames  |#
#| must be a list along the atom axis,   |#
#| and they must have the same length.   |#
#=========================================#
# elem       = List of elements
# partial_charge = List of atomic partial charges 
# atomname   = List of atom names (can come from MM coordinate file)
# atomtype   = List of atom types (can come from MM force field)
# tinkersuf  = String that comes after the XYZ coordinates in TINKER .xyz or .arc files
# resid      = Residue IDs (can come from MM coordinate file)
# resname    = Residue names
AtomVariableNames = set(['elem', 'partial_charge', 'atomname', 'atomtype', 'tinkersuf', 'resid', 'resname', 'qcsuf', 'qm_ghost', 'chain', 'altloc', 'icode'])
#=========================================#
#| This can be any data attribute we     |#
#| want but it's usually some property   |#
#| of the molecule not along the frame   |#
#| atom axis.                            |#
#=========================================#
# bonds      = A list of 2-tuples representing bonds.  Carefully pruned when atom subselection is done.
# fnm        = The file name that the class was built from
# qcrems     = The Q-Chem 'rem' variables stored as a list of OrderedDicts
# qctemplate = The Q-Chem template file, not including the coordinates or rem variables
# charge     = The net charge of the molecule
# mult       = The spin multiplicity of the molecule
MetaVariableNames = set(['fnm', 'ftype', 'qcrems', 'qctemplate', 'charge', 'mult', 'bonds'])
# Variable names relevant to quantum calculations explicitly
QuantumVariableNames = set(['qcrems', 'qctemplate', 'charge', 'mult', 'qcsuf', 'qm_ghost'])
# Superset of all variable names.
AllVariableNames = QuantumVariableNames | AtomVariableNames | MetaVariableNames | FrameVariableNames

# OrderedDict requires Python 2.7 or higher
import os, sys, re, copy
import numpy as np
from numpy import sin, cos, arcsin, arccos
import imp
import itertools
from collections import OrderedDict, namedtuple
from ctypes import *
from warnings import warn

# Covalent radii from Cordero et al. 'Covalent radii revisited' Dalton Transactions 2008, 2832-2838.
Radii = [0.31, 0.28, # H and He
         1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, # First row elements
         1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, # Second row elements
         2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.50, 
         1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, # Third row elements, K through Kr
         2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 
         1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, # Fourth row elements, Rb through Xe
         2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 
         1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, # Fifth row elements, s and f blocks
         1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 
         1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, # Fifth row elements, d and p blocks
         2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69] # Sixth row elements

# A list that gives you the element if you give it the atomic number, hence the 'none' at the front.
Elements = ["None",'H','He',
            'Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
            'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
            'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
            'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
            'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt']

# Dictionary of atomic masses ; also serves as the list of elements (periodic table)
PeriodicTable = OrderedDict([('H' , 1.0079), ('He' , 4.0026), 
                             ('Li' , 6.941), ('Be' , 9.0122), ('B' , 10.811), ('C' , 12.0107), ('N' , 14.0067), ('O' , 15.9994), ('F' , 18.9984), ('Ne' , 20.1797),
                             ('Na' , 22.9897), ('Mg' , 24.305), ('Al' , 26.9815), ('Si' , 28.0855), ('P' , 30.9738), ('S' , 32.065), ('Cl' , 35.453), ('Ar' , 39.948), 
                             ('K' , 39.0983), ('Ca' , 40.078), ('Sc' , 44.9559), ('Ti' , 47.867), ('V' , 50.9415), ('Cr' , 51.9961), ('Mn' , 54.938), ('Fe' , 55.845), ('Co' , 58.9332), 
                             ('Ni' , 58.6934), ('Cu' , 63.546), ('Zn' , 65.39), ('Ga' , 69.723), ('Ge' , 72.64), ('As' , 74.9216), ('Se' , 78.96), ('Br' , 79.904), ('Kr' , 83.8), 
                             ('Rb' , 85.4678), ('Sr' , 87.62), ('Y' , 88.9059), ('Zr' , 91.224), ('Nb' , 92.9064), ('Mo' , 95.94), ('Tc' , 98), ('Ru' , 101.07), ('Rh' , 102.9055), 
                             ('Pd' , 106.42), ('Ag' , 107.8682), ('Cd' , 112.411), ('In' , 114.818), ('Sn' , 118.71), ('Sb' , 121.76), ('Te' , 127.6), ('I' , 126.9045), ('Xe' , 131.293), 
                             ('Cs' , 132.9055), ('Ba' , 137.327), ('La' , 138.9055), ('Ce' , 140.116), ('Pr' , 140.9077), ('Nd' , 144.24), ('Pm' , 145), ('Sm' , 150.36), 
                             ('Eu' , 151.964), ('Gd' , 157.25), ('Tb' , 158.9253), ('Dy' , 162.5), ('Ho' , 164.9303), ('Er' , 167.259), ('Tm' , 168.9342), ('Yb' , 173.04), 
                             ('Lu' , 174.967), ('Hf' , 178.49), ('Ta' , 180.9479), ('W' , 183.84), ('Re' , 186.207), ('Os' , 190.23), ('Ir' , 192.217), ('Pt' , 195.078), 
                             ('Au' , 196.9665), ('Hg' , 200.59), ('Tl' , 204.3833), ('Pb' , 207.2), ('Bi' , 208.9804), ('Po' , 209), ('At' , 210), ('Rn' , 222), 
                             ('Fr' , 223), ('Ra' , 226), ('Ac' , 227), ('Th' , 232.0381), ('Pa' , 231.0359), ('U' , 238.0289), ('Np' , 237), ('Pu' , 244), 
                             ('Am' , 243), ('Cm' , 247), ('Bk' , 247), ('Cf' , 251), ('Es' , 252), ('Fm' , 257), ('Md' , 258), ('No' , 259), 
                             ('Lr' , 262), ('Rf' , 261), ('Db' , 262), ('Sg' , 266), ('Bh' , 264), ('Hs' , 277), ('Mt' , 268)])

def getElement(mass):
    return PeriodicTable.keys()[np.argmin([np.abs(m-mass) for m in PeriodicTable.values()])]

#============================#
#| DCD read/write functions |#
#============================#
# Try to load _dcdlib.so either from a directory in the LD_LIBRARY_PATH
# or from the same directory as this module.
try: _dcdlib = CDLL("_dcdlib.so")
except:
    try: _dcdlib = CDLL(os.path.join(imp.find_module(__name__.split('.')[0])[1],"_dcdlib.so"))
    except: warn('The dcdlib module cannot be imported (Cannot read/write DCD files)')

#============================#
#| PDB read/write functions |#
#============================#
try: from PDB import *
except: warn('The pdb module cannot be miported (Cannot read/write PDB files)')

#=============================#
#| Mol2 read/write functions |#
#=============================#
try: import Mol2
except: warn('The Mol2 module cannot be imported (Cannot read/write Mol2 files)')

#==============================#
#| OpenMM interface functions |#
#==============================#
try: 
    from simtk.unit import *
    from simtk.openmm import *
    from simtk.openmm.app import *
except: warn('The OpenMM modules cannot be imported (Cannot interface with OpenMM)')
    
#===========================#
#| Convenience subroutines |#
#===========================#

## One bohr equals this many angstroms
bohrang = 0.529177249

def nodematch(node1,node2):
    # Matching two nodes of a graph.  Nodes are equivalent if the elements are the same
    return node1['e'] == node2['e']

def isint(word):
    """ONLY matches integers! If you have a decimal point? None shall pass!"""
    return re.match('^[-+]?[0-9]+$',word)

def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, integer, or what have you"""
    return re.match('^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$',word)

# Used to get the white spaces in a split line.
splitter = re.compile(r'(\s+|\S+)')

# Container for Bravais lattice vector.  Three cell lengths, three angles, three vectors, volume, and TINKER trig functions.
Box = namedtuple('Box',['a','b','c','alpha','beta','gamma','A','B','C','V'])
radian = 180. / np.pi
def BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma):
    """ This function takes in three lattice lengths and three lattice angles, and tries to return a complete box specification. """
    alph = alpha*np.pi/180
    bet  = beta*np.pi/180
    gamm = gamma*np.pi/180
    v = np.sqrt(1 - cos(alph)**2 - cos(bet)**2 - cos(gamm)**2 + 2*cos(alph)*cos(bet)*cos(gamm))
    Mat = np.mat([[a, b*cos(gamm), c*cos(bet)],
                  [0, b*sin(gamm), c*((cos(alph)-cos(bet)*cos(gamm))/sin(gamm))],
                  [0, 0, c*v/sin(gamm)]])
    L1 = Mat*np.mat([[1],[0],[0]])
    L2 = Mat*np.mat([[0],[1],[0]])
    L3 = Mat*np.mat([[0],[0],[1]])
    return Box(a,b,c,alpha,beta,gamma,np.array(L1).flatten(),np.array(L2).flatten(),np.array(L3).flatten(),v*a*b*c)

def BuildLatticeFromVectors(v1, v2, v3):
    """ This function takes in three lattice vectors and tries to return a complete box specification. """
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)
    alpha = arccos(np.linalg.dot(v2, v3) / np.linalg.norm(v2) / np.linalg.norm(v3)) * radian
    beta  = arccos(np.linalg.dot(v1, v3) / np.linalg.norm(v1) / np.linalg.norm(v3)) * radian
    gamma = arccos(np.linalg.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)) * radian
    alph = alpha*np.pi/180
    bet  = beta*np.pi/180
    gamm = gamma*np.pi/180
    v = np.sqrt(1 - cos(alph)**2 - cos(bet)**2 - cos(gamm)**2 + 2*cos(alph)*cos(bet)*cos(gamm))
    Mat = np.mat([[a, b*cos(gamm), c*cos(bet)],
                  [0, b*sin(gamm), c*((cos(alph)-cos(bet)*cos(gamm))/sin(gamm))],
                  [0, 0, c*v/sin(gamm)]])
    L1 = Mat*np.mat([[1],[0],[0]])
    L2 = Mat*np.mat([[0],[1],[0]])
    L3 = Mat*np.mat([[0],[0],[1]])
    return Box(a,b,c,alpha,beta,gamma,np.array(L1).flatten(),np.array(L2).flatten(),np.array(L3).flatten(),v*a*b*c)

#===========================#
#|   Connectivity graph    |#
#|  Good for doing simple  |#
#|     topology tricks     |#
#===========================#
try:
    import networkx as nx
    class MyG(nx.Graph):
        def __init__(self):
            super(MyG,self).__init__()
            self.Alive = True
        def __eq__(self, other):
            # This defines whether two MyG objects are "equal" to one another.
            if not self.Alive:
                return False
            if not other.Alive:
                return False
            return nx.is_isomorphic(self,other,node_match=nodematch)
        def __hash__(self):
            ''' The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash. '''
            return 1
        def L(self):
            ''' Return a list of the sorted atom numbers in this graph. '''
            return sorted(self.nodes())
        def AStr(self):
            ''' Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . '''
            return ','.join(['%i' % i for i in self.L()])
        def e(self):
            ''' Return an array of the elements.  For instance ['H' 'C' 'C' 'H']. '''
            elems = nx.get_node_attributes(self,'e')
            return [elems[i] for i in self.L()]
        def ef(self):
            ''' Create an Empirical Formula '''
            Formula = list(self.e())
            return ''.join([('%s%i' % (k, Formula.count(k)) if Formula.count(k) > 1 else '%s' % k) for k in sorted(set(Formula))])
        def x(self):
            ''' Get a list of the coordinates. '''
            coors = nx.get_node_attributes(self,'x')
            return np.array([coors[i] for i in self.L()])
    try:
        import contact
    except:
        warn("'contact' cannot be imported (topology tools will be slow.)")
except:
    warn("NetworkX cannot be imported (topology tools won't work).  Most functionality should still work though.")


def format_xyz_coord(element,xyz,tinker=False):
    """ Print a line consisting of (element, x, y, z) in accordance with .xyz file format

    @param[in] element A chemical element of a single atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    if tinker:
        return "%-3s % 11.6f % 11.6f % 11.6f" % (element,xyz[0],xyz[1],xyz[2])
    else:
        return "%-5s % 15.10f % 15.10f % 15.10f" % (element,xyz[0],xyz[1],xyz[2])

def format_gro_coord(resid, resname, aname, seqno, xyz):
    """ Print a line in accordance with .gro file format, with six decimal points of precision

    @param[in] resid The number of the residue that the atom belongs to
    @param[in] resname The name of the residue that the atom belongs to
    @param[in] aname The name of the atom
    @param[in] seqno The sequential number of the atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom
    
    """
    return "%5i%-5s%5s%5i % 10.6f % 10.6f % 10.6f" % (resid,resname,aname,seqno,xyz[0],xyz[1],xyz[2])

def format_xyzgen_coord(element,xyzgen):
    """ Print a line consisting of (element, p, q, r, s, t, ...) where
    (p, q, r) are arbitrary atom-wise data (this might happen, for
    instance, with atomic charges)

    @param[in] element A chemical element of a single atom
    @param[in] xyzgen A N-element array containing data for that atom

    """
    return "%-5s" + ' '.join(["% 15.10f" % i] for i in xyzgen)

def format_gro_box(box):
    """ Print a line corresponding to the box vector in accordance with .gro file format

    @param[in] box Box NamedTuple
    
    """
    if box.alpha == 90.0 and box.beta == 90.0 and box.gamma == 90.0:
        return ' '.join(["% 10.6f" % (i/10) for i in [box.a, box.b, box.c]])
    else:
        return ' '.join(["% 10.6f" % (i/10) for i in [box.A[0], box.B[1], box.C[2], box.A[1], box.A[2], box.B[0], box.B[2], box.C[0], box.C[1]]])

def is_gro_coord(line):
    """ Determines whether a line contains GROMACS data or not

    @param[in] line The line to be tested
    
    """
    sline = line.split()
    if len(sline) == 6:
        return all([isint(sline[2]),isfloat(sline[3]),isfloat(sline[4]),isfloat(sline[5])])
    elif len(sline) == 5:
        return all([isint(line[15:20]),isfloat(sline[2]),isfloat(sline[3]),isfloat(sline[4])])
    else:
        return 0

def is_charmm_coord(line):
    """ Determines whether a line contains CHARMM data or not

    @param[in] line The line to be tested
    
    """
    sline = line.split()
    if len(sline) >= 7:
        return all([isint(sline[0]), isint(sline[1]), isfloat(sline[4]), isfloat(sline[5]), isfloat(sline[6])])
    else:
        return 0

def is_gro_box(line):
    """ Determines whether a line contains a GROMACS box vector or not

    @param[in] line The line to be tested
    
    """
    sline = line.split()
    if len(sline) == 9 and all([isfloat(i) for i in sline]):
        return 1
    elif len(sline) == 3 and all([isfloat(i) for i in sline]):
        return 1
    else:
        return 0

def add_strip_to_mat(mat,strip):
    out = list(mat)
    if out == [] and strip != []: 
        out = list(strip)
    elif out != [] and strip != []:
        for (i,j) in zip(out,strip):
            i += list(j)
    return out

def pvec(vec):
    return ''.join([' % .10e' % i for i in list(vec.flatten())])

def grouper(n, iterable):
    """ Groups a big long iterable into groups of ten or what have you. """
    args = [iter(iterable)] * n
    return list([e for e in t if e != None] for t in itertools.izip_longest(*args))

def even_list(totlen, splitsize):
    """ Creates a list of number sequences divided as evenly as possible.  """
    joblens = np.zeros(splitsize,dtype=int)
    subsets = []
    for i in range(totlen):
        joblens[i%splitsize] += 1
    jobnow = 0
    for i in range(splitsize):
        subsets.append(range(jobnow, jobnow + joblens[i]))
        jobnow += joblens[i]
    return subsets

class MolfileTimestep(Structure):
    """ Wrapper for the timestep C structure used in molfile plugins. """
    _fields_ = [("coords",POINTER(c_float)), ("velocities",POINTER(c_float)),
                ("A",c_float), ("B",c_float), ("C",c_float), ("alpha",c_float), 
                ("beta",c_float), ("gamma",c_float), ("physical_time",c_double)]
    
def both(A, B, key):
    return key in A.Data and key in B.Data

def diff(A, B, key):
    if not (key in A.Data and key in B.Data) : return False
    else:
        if type(A.Data[key]) is np.ndarray:
            return (A.Data[key] != B.Data[key]).any()
        else:
            return A.Data[key] != B.Data[key]

def either(A, B, key):
    return key in A.Data or key in B.Data

#===========================#
#|  Alignment subroutines  |#
#| Moments added 08/03/12  |#
#===========================#
def EulerMatrix(T1,T2,T3):
    """ Constructs an Euler matrix from three Euler angles. """
    DMat = np.mat(np.zeros((3,3),dtype = float))
    DMat[0,0] = np.cos(T1)
    DMat[0,1] = np.sin(T1)
    DMat[1,0] = -np.sin(T1)
    DMat[1,1] = np.cos(T1)
    DMat[2,2] = 1
    CMat = np.mat(np.zeros((3,3),dtype = float))
    CMat[0,0] = 1
    CMat[1,1] = np.cos(T2)
    CMat[1,2] = np.sin(T2)
    CMat[2,1] = -np.sin(T2)
    CMat[2,2] = np.cos(T2)
    BMat = np.mat(np.zeros((3,3),dtype = float))
    BMat[0,0] = np.cos(T3)
    BMat[0,1] = np.sin(T3)
    BMat[1,0] = -np.sin(T3)
    BMat[1,1] = np.cos(T3)
    BMat[2,2] = 1
    EMat = BMat*CMat*DMat
    return np.mat(EMat)

def ComputeOverlap(theta,elem,xyz1,xyz2):
    """ 
    Computes an 'overlap' between two molecules based on some
    fictitious density.  Good for fine-tuning alignment but gets stuck
    in local minima.
    """
    xyz2R = np.array(EulerMatrix(theta[0],theta[1],theta[2])*np.mat(xyz2.T)).T
    Obj = 0.0
    elem = np.array(elem)
    for i in set(elem):
        for j in np.where(elem==i)[0]:
            for k in np.where(elem==i)[0]:
                dx = xyz1[j] - xyz2R[k]
                dx2 = np.dot(dx,dx)
                Obj -= np.exp(-0.5*dx2)
    return Obj

def AlignToDensity(elem,xyz1,xyz2,binary=False):
    """ 
    Computes a "overlap density" from two frames.
    This function can be called by AlignToMoments to get rid of inversion problems
    """
    grid = np.pi*np.array(list(itertools.product([0,1],[0,1],[0,1])))
    ovlp = np.array([ComputeOverlap(e, elem, xyz1, xyz2) for e in grid]) # Mao
    t1 = grid[np.argmin(ovlp)]
    xyz2R = (np.array(EulerMatrix(t1[0],t1[1],t1[2])*np.mat(xyz2.T)).T).copy()
    return xyz2R

def AlignToMoments(elem,xyz1,xyz2=None):
    """Pre-aligns molecules to 'moment of inertia'.
    If xyz2 is passed in, it will assume that xyz1 is already
    aligned to the moment of inertia, and it simply does 180-degree
    rotations to make sure nothing is inverted."""
    xyz = xyz1 if xyz2 == None else xyz2
    I = np.zeros((3,3))
    for i, xi in enumerate(xyz):
        I += (np.dot(xi,xi)*np.eye(3) - np.outer(xi,xi))
        # This is the original line from MSMBuilder, but we're choosing not to use masses
        # I += PeriodicTable[elem[i]]*(np.dot(xi,xi)*np.eye(3) - np.outer(xi,xi))
    A, B = np.linalg.eig(I)
    # Sort eigenvectors by eigenvalue
    BB   = B[:, np.argsort(A)]
    determ = np.linalg.det(BB)
    Thresh = 1e-3
    if np.abs(determ - 1.0) > Thresh:
        if np.abs(determ + 1.0) > Thresh:
            print "in AlignToMoments, determinant is % .3f" % determ
        BB[:,2] *= -1
    xyzr = np.array(np.mat(BB).T * np.mat(xyz).T).T.copy()
    if xyz2 != None:
        xyzrr = AlignToDensity(elem,xyz1,xyzr,binary=True)
        return xyzrr
    else:
        return xyzr

def get_rotate_translate(matrix1,matrix2):
    assert np.shape(matrix1) == np.shape(matrix2), 'Matrices not of same dimensions'
    
    # Store number of rows
    nrows = np.shape(matrix1)[0]
    
    # Getting centroid position for each selection
    avg_pos1 = matrix1.sum(axis=0)/nrows
    avg_pos2 = matrix2.sum(axis=0)/nrows

    # Translation of matrices
    avg_matrix1 = matrix1-avg_pos1
    avg_matrix2 = matrix2-avg_pos2

    # Covariance matrix
    covar = np.dot(avg_matrix1.T,avg_matrix2)
    
    # Do the SVD in order to get rotation matrix
    u,s,wt = np.linalg.svd(covar)
    
    # Rotation matrix
    # Transposition of u,wt
    rot_matrix = wt.T*u.T
    
    # Insure a right-handed coordinate system
    # need to fix this!!
    #if np.linalg.det(rot_matrix) > 0:
    #    wt[2] = -wt[2]
    rot_matrix = np.transpose(np.dot(np.transpose(wt),np.transpose(u)))

    trans_matrix = avg_pos2-np.dot(avg_pos1,rot_matrix)
    return trans_matrix, rot_matrix

class Molecule(object):
    """ Lee-Ping's general file format conversion class.

    The purpose of this class is to read and write chemical file formats in a
    way that is convenient for research.  There are highly general file format
    converters out there (e.g. catdcd, openbabel) but I find that writing 
    my own class can be very helpful for specific purposes.  Here are some things
    this class can do:
    
    - Convert a .gro file to a .xyz file, or a .pdb file to a .dcd file.
    Data is stored internally, so any readable file can be converted into
    any writable file as long as there is sufficient information to write 
    that file.
    
    - Accumulate information from different files.  For example, we may read
    A.gro to get a list of coordinates, add quantum settings from a B.in file,
    and write A.in (this gives us a file that we can use to run QM calculations)

    - Concatenate two trajectories together as long as they're compatible.  This
    is done by creating two Molecule objects and then simply adding them.  Addition
    means two things:  (1) Information fields missing from each class, but present 
    in the other, are added to the sum, and (2) Appendable or per-frame fields
    (i.e. coordinates) are concatenated together.

    - Slice trajectories using reasonable Python language.  That is to say,
    MyMolecule[1:10] returns a new Molecule object that contains frames 2 through 10.
    
    Next step: Read in Q-Chem output data using this too!

    Special variables:  These variables cannot be set manually because
    there is a special method associated with getting them.

    na = The number of atoms.  You'll get this if you use MyMol.na or MyMol['na'].
    na = The number of snapshots.  You'll get this if you use MyMol.ns or MyMol['ns'].

    Unit system:  Angstroms.

    """

    def __len__(self):
        """ Return the number of frames in the trajectory. """
        L = -1
        klast = None
        Defined = False
        for key in self.FrameKeys:
            Defined = True
            if L != -1 and len(self.Data[key]) != L:
                raise Exception('The keys %s and %s have different lengths - this isn\'t supposed to happen for two FrameKeys member variables.' % (key, klast))
            L = len(self.Data[key])
            klast = key
        if not Defined:
            return 0
        return L

    def __getattr__(self, key):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary. """
        if key == 'ns':
            return len(self)
        elif key == 'na': # The 'na' attribute is the number of atoms.
            L = -1
            klast = None
            Defined = False
            for key in self.AtomKeys:
                Defined = True
                if L != -1 and len(self.Data[key]) != L:
                    raise Exception('The keys %s and %s have different lengths (%i %i) - this isn\'t supposed to happen for two AtomKeys member variables.' % (key, klast, len(self.Data[key]), len(self.Data[klast])))
                L = len(self.Data[key])
                klast = key
            if Defined:
                return L
            elif 'xyzs' in self.Data:
                return len(self.xyzs[0])
            else:
                return 0
            #raise Exception('na is ill-defined if the molecule has no AtomKeys member variables.')
        ## These attributes return a list of attribute names defined in this class that belong in the chosen category.
        ## For example: self.FrameKeys should return set(['xyzs','boxes']) if xyzs and boxes exist in self.Data
        elif key == 'FrameKeys':
            return set(self.Data) & FrameVariableNames
        elif key == 'AtomKeys':
            return set(self.Data) & AtomVariableNames
        elif key == 'MetaKeys':
            return set(self.Data) & MetaVariableNames
        elif key == 'QuantumKeys':
            return set(self.Data) & QuantumVariableNames
        elif key in self.Data:
            return self.Data[key]
        return getattr(super(Molecule, self), key)

    def __setattr__(self, key, value):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary. """
        ## These attributes return a list of attribute names defined in this class, that belong in the chosen category.
        ## For example: self.FrameKeys should return set(['xyzs','boxes']) if xyzs and boxes exist in self.Data
        if key in AllVariableNames:
            self.Data[key] = value
        return super(Molecule,self).__setattr__(key, value)

    def __getitem__(self, key):
        """ 
        The Molecule class has list-like behavior, so we can get slices of it.
        If we say MyMolecule[0:10], then we'll return a copy of MyMolecule with frames 0 through 9.
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key,np.ndarray):
            if isinstance(key, int):
                key = [key]
            New = Molecule()
            for k in self.FrameKeys:
                if k == 'boxes':
                    New.Data[k] = [j for i, j in enumerate(self.Data[k]) if i in np.arange(len(self))[key]]
                else:
                    New.Data[k] = list(np.array(self.Data[k])[key])
            for k in self.AtomKeys | self.MetaKeys:
                New.Data[k] = copy.deepcopy(self.Data[k])
            return New
        else:
            raise Exception('getitem is not implemented for keys of type %s' % str(key))

    def __delitem__(self, key):
        """ 
        Similarly, in order to delete a frame, we simply perform item deletion on
        framewise variables.
        """
        for k in self.FrameKeys:
            del self.Data[k][key]

    def __iter__(self):
        """ List-like behavior for looping over trajectories. Note that these values are returned by reference. 
        Note that this is intended to be more efficient than __getitem__, so when we loop over a trajectory,
        it's best to go "for m in M" instead of "for i in range(len(M)): m = M[i]"
        """
        for frame in range(self.ns):
            New = Molecule()
            for k in self.FrameKeys:
                New.Data[k] = self.Data[k][frame]
            for k in self.AtomKeys | self.MetaKeys:
                New.Data[k] = self.Data[k]
            yield New

    def __add__(self,other):
        """ Add method for Molecule objects. """
        # Check type of other
        if not isinstance(other,Molecule):
            raise TypeError('A Molecule instance can only be added to another Molecule instance')
        # Create the sum of the two classes by copying the first class.
        Sum = Molecule()
        for key in AtomVariableNames | MetaVariableNames:
            # Because a molecule object can only have one 'file name' or 'file type' attribute,
            # we only keep the original one.  This isn't perfect, but that's okay.
            if key == 'fnm' or key == 'ftype' and key in self.Data:
                Sum.Data[key] = self.Data[key]
            elif diff(self, other, key):
                raise Exception('The data member called %s is not the same for these two objects' % key)
            elif key in self.Data:
                Sum.Data[key] = copy.deepcopy(self.Data[key])
            elif key in other.Data:
                Sum.Data[key] = copy.deepcopy(other.Data[key])
        for key in FrameVariableNames:
            if both(self, other, key):
                if type(self.Data[key]) is not list:
                    raise Exception('Key %s in self is a FrameKey, it must be a list' % key)
                if type(other.Data[key]) is not list:
                    raise Exception('Key %s in other is a FrameKey, it must be a list' % key)
                Sum.Data[key] = list(self.Data[key] + other.Data[key])
            elif either(self, other, key):
                raise Exception('Key %s is a FrameKey, must exist in both self and other for them to be added (for now).')
        return Sum
 
    def __iadd__(self,other):
        """ Add method for Molecule objects. """
        # Check type of other
        if not isinstance(other,Molecule):
            raise TypeError('A Molecule instance can only be added to another Molecule instance')
        # Create the sum of the two classes by copying the first class.
        for key in AtomVariableNames | MetaVariableNames:
            if key == 'fnm' or key == 'ftype': pass
            elif diff(self, other, key):
                raise Exception('The data member called %s is not the same for these two objects' % key)
            # Information from the other class is added to this class (if said info doesn't exist.)
            elif key in other.Data:
                self.Data[key] = copy.deepcopy(other.Data[key])
        # FrameKeys must be a list.
        for key in FrameVariableNames:
            if both(self, other, key):
                if type(self.Data[key]) is not list:
                    raise Exception('Key %s in self is a FrameKey, it must be a list' % key)
                if type(other.Data[key]) is not list:
                    raise Exception('Key %s in other is a FrameKey, it must be a list' % key)
                self.Data[key] += other.Data[key]
            elif either(self, other, key):
                raise Exception('Key %s is a FrameKey, must exist in both self and other for them to be added (for now).' % key)
        return self

    def append(self,other):
        self += other

    def __init__(self, fnm = None, ftype = None, positive_resid=True, build_topology = True):
        """ To create the Molecule object, we simply define the table of
        file reading/writing functions and read in a file if it is
        provided."""
        #=========================================#
        #|           File type tables            |#
        #|    Feel free to edit these as more    |#
        #|      readers / writers are added      |#
        #=========================================#
        ## The table of file readers
        self.Read_Tab = {'gaussian' : self.read_com,
                         'gromacs'  : self.read_gro,
                         'charmm'   : self.read_charmm,
                         'dcd'      : self.read_dcd,
                         'mdcrd'    : self.read_mdcrd,
                         'pdb'      : self.read_pdb,
                         'xyz'      : self.read_xyz,
                         'mol2'     : self.read_mol2,
                         'qcin'     : self.read_qcin,
                         'qcout'    : self.read_qcout,
                         'qcesp'    : self.read_qcesp,
                         'qdata'    : self.read_qdata,
                         'tinker'   : self.read_arc}
        ## The table of file writers
        self.Write_Tab = {'gromacs' : self.write_gro,
                          'xyz'     : self.write_xyz,
                          'molproq' : self.write_molproq,
                          'dcd'     : self.write_dcd,
                          'mdcrd'   : self.write_mdcrd,
                          'pdb'     : self.write_pdb,
                          'qcin'    : self.write_qcin,
                          'qdata'   : self.write_qdata,
                          'tinker'  : self.write_arc}
        ## A funnel dictionary that takes redundant file types
        ## and maps them down to a few.
        self.Funnel    = {'gromos'  : 'gromacs',
                          'gro'     : 'gromacs',
                          'g96'     : 'gromacs',
                          'gmx'     : 'gromacs',
                          'in'      : 'qcin',
                          'com'     : 'gaussian',
                          'out'     : 'qcout',
                          'esp'     : 'qcesp',
                          'txt'     : 'qdata',
                          'crd'     : 'charmm',
                          'cor'     : 'charmm',
                          'arc'     : 'tinker'}
        ## Creates entries like 'gromacs' : 'gromacs' and 'xyz' : 'xyz'
        ## in the Funnel
        self.positive_resid = positive_resid
        for i in set(self.Read_Tab.keys() + self.Write_Tab.keys()):
            self.Funnel[i] = i
        # Data container.  All of the data is stored in here.
        self.Data = {}
        ## Read in stuff if we passed in a file name, otherwise return an empty instance.
        if fnm != None:
            self.Data['fnm'] = fnm
            if ftype == None:
                ## Try to determine from the file name using the extension.
                ftype = os.path.splitext(fnm)[1][1:]
            if not os.path.exists(fnm):
                raise IOError('Tried to create Molecule object from a file that does not exist')
            self.Data['ftype'] = ftype
            ## Actually read the file.
            Parsed = self.Read_Tab[self.Funnel[ftype.lower()]](fnm)
            ## Set member variables.
            for key, val in Parsed.items():
                self.Data[key] = val
            ## Create a list of comment lines if we don't already have them from reading the file.
            if 'comms' not in self.Data:
                self.comms = ['Generated by ForceBalance from %s: Frame %i of %i' % (fnm, i+1, self.ns) for i in range(self.ns)]
            else:
                self.comms = [i.expandtabs() for i in self.comms]
            ## Make sure the comment line isn't too long
            # for i in range(len(self.comms)):
            #     self.comms[i] = self.comms[i][:100] if len(self.comms[i]) > 100 else self.comms[i]
            # Attempt to build the topology for small systems. :)
            if 'networkx' in sys.modules and build_topology:
                if self.na > 10000:
                    print "Warning: Large number of atoms (%i), topology building may take a long time" % self.na
                self.topology = self.build_topology()
                self.molecules = nx.connected_component_subgraphs(self.topology)
                if 'bonds' not in self.Data:
                    self.Data['bonds'] = self.topology.edges()

    #=====================================#
    #|     Core read/write functions     |#
    #| Hopefully we won't have to change |#
    #|         these very often!         |#
    #=====================================#

    def require(self, *args):
        for arg in args:
            if arg not in self.Data:
                raise Exception("%s is a required attribute for writing this type of file but it's not present" % arg)

    # def read(self, fnm, ftype = None):
    #     """ Read in a file. """
    #     if ftype == None:
    #         ## Try to determine from the file name using the extension.
    #         ftype = os.path.splitext(fnm)[1][1:]
    #     ## This calls the table of reader functions and prints out an error message if it fails.
    #     ## 'Answer' is a dictionary of data that is returned from the reader function.
    #     Answer = self.Read_Tab[self.Funnel[ftype.lower()]](fnm)
    #     return Answer

    def write(self,fnm=None,ftype=None,append=False,select=None):
        if fnm == None and ftype == None:
            raise Exception("Output file name and file type are not specified.")
        elif ftype == None:
            ftype = os.path.splitext(fnm)[1][1:]
        ## Fill in comments.
        if len(self.comms) < len(self.xyzs):
            for i in range(len(self.comms), len(self.xyzs)):
                self.comms.append("Frame %i: generated by ForceBalance" % i)
        ## I needed to add in this line because the DCD writer requires the file name,
        ## but the other methods don't.
        self.fout = fnm
        if type(select) in [int, np.int64, np.int32]:
            select = [select]
        if select == None:
            select = range(len(self))
        Answer = self.Write_Tab[self.Funnel[ftype.lower()]](select)
        ## Any method that returns text will give us a list of lines, which we then write to the file.
        if Answer != None:
            if fnm == None or fnm == sys.stdout:
                outfile = sys.stdout
            elif append:
                outfile = open(fnm,'a')
            else:
                outfile = open(fnm,'w')
            for line in Answer:
                print >> outfile,line
            outfile.close()

    #=====================================#
    #|         Useful functions          |#
    #|     For doing useful things       |#
    #=====================================#

    def center_of_mass(self):
        M = sum([PeriodicTable[self.elem[i]] for i in range(self.na)])
        return np.array([np.sum([xyz[i,:] * PeriodicTable[self.elem[i]] / M for i in range(xyz.shape[0])],axis=0) for xyz in self.xyzs])

    def radius_of_gyration(self):
        M = sum([PeriodicTable[self.elem[i]] for i in range(self.na)])
        coms = self.center_of_mass()
        rgs = []
        for i, xyz in enumerate(self.xyzs):
            xyz1 = xyz.copy()
            xyz1 -= coms[i]
            rgs.append(np.sum([PeriodicTable[self.elem[i]]*np.dot(x,x) for i, x in enumerate(xyz1)])/M)
        return np.array(rgs)
            
#        if center:
#            xyz1 -= xyz1.mean(0)
#        for index2, xyz2 in enumerate(self.xyzs):
#            if index2 == 0: continue
#            xyz2 -= xyz2.mean(0)
#            if smooth:
#                ref = index2-1
#            else:
#                ref = 0
#            tr, rt = get_rotate_translate(xyz2,self.xyzs[ref])
#            xyz2 = np.dot(xyz2, rt) + tr
#            self.xyzs[index2] = xyz2

    def load_frames(self, fnm):
        NewMol = Molecule(fnm)
        if NewMol.na != self.na:
            raise Exception('When loading frames, don\'t change the number of atoms.')
        for key in NewMol.FrameKeys:
            self.Data[key] = NewMol.Data[key]

    def edit_qcrems(self, in_dict, subcalc = None):
        if subcalc == None:
            for qcrem in self.qcrems:
                for key, val in in_dict.items():
                    qcrem[key] = val
        else:
            for key, val in in_dict.items():
                self.qcrems[subcalc][key] = val

    def add_quantum(self, other):
        if type(other) is Molecule:
            OtherMol = other
        elif type(other) is str:
            OtherMol = Molecule(other)
        for key in OtherMol.QuantumKeys:
            if key in AtomVariableNames and len(OtherMol.Data[key]) != self.na:
                raise Exception('The quantum-key %s is AtomData, but it doesn\'t have the same number of atoms as the Molecule object we\'re adding it to.')
            self.Data[key] = copy.deepcopy(OtherMol.Data[key])

    def add_virtual_site(self, idx, **kwargs):
        """ Add a virtual site to the system.  This does NOT set the position of the virtual site; it sits at the origin. """
        for key in self.AtomKeys:
            if key in kwargs:
                self.Data[key].insert(idx,kwargs[key])
            else:
                raise Exception('You need to specify %s when adding a virtual site to this molecule.' % key)
        if 'xyzs' in self.Data:
            for i, xyz in enumerate(self.xyzs):
                if 'pos' in kwargs:
                    self.xyzs[i] = np.insert(xyz, idx, xyz[kwargs['pos']], axis=0)
                else:
                    self.xyzs[i] = np.insert(xyz, idx, 0.0, axis=0)
        else:
            raise Exception('You need to have xyzs in this molecule to add a virtual site.')

    def replace_peratom(self, key, orig, want):
        """ Replace all of the data for a certain attribute in the system from orig to want. """
        if key in self.Data:
            for i in range(self.na):
                if self.Data[key][i] == orig:
                    self.Data[key][i] = want
        else:
            raise Exception('The key that we want to replace (%s) doesn\'t exist.' % key)

    def replace_peratom_conditional(self, key1, cond, key2, orig, want):
        """ Replace all of the data for a attribute key2 from orig to want, contingent on key1 being equal to cond. 
        For instance: replace H1 with H2 if resname is SOL."""
        if key2 in self.Data and key1 in self.Data:
            for i in range(self.na):
                if self.Data[key2][i] == orig and self.Data[key1][i] == cond:
                    self.Data[key2][i] = want
        else:
            raise Exception('Either the comparison or replacement key (%s, %s) doesn\'t exist.' % (key1, key2))

    def atom_select(self,atomslice):
        """ Return a copy of the object with certain atoms selected.  Takes an integer, list or array as argument. """
        if isinstance(atomslice, int):
            atomslice = [atomslice]
        if isinstance(atomslice, list):
            atomslice = np.array(atomslice)
        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])
        for key in self.AtomKeys:
            if key == 'tinkersuf': # Tinker suffix is a bit tricky
                Map = dict([(a+1, i+1) for i, a in enumerate(atomslice)])
                CopySuf = list(np.array(self.Data[key])[atomslice])
                NewSuf = []
                for line in CopySuf:
                    whites      = re.split('[^ ]+',line)
                    s           = line.split()
                    if len(s) > 1:
                        for i in range(1,len(s)):
                            s[i] = str(Map[int(s[i])])
                    NewSuf.append(''.join([whites[j]+s[j] for j in range(len(s))]))
                New.Data['tinkersuf'] = NewSuf[:]
            else:
                New.Data[key] = list(np.array(self.Data[key])[atomslice])
        if 'xyzs' in self.Data:
            for i in range(self.ns):
                New.xyzs[i] = self.xyzs[i][atomslice]
        if 'bonds' in self.Data:
            # print self.bonds
            # print atomslice
            # print [(b[0] in atomslice and b[1] in atomslice) for b in self.bonds]
            New.Data['bonds'] = [(list(atomslice).index(b[0]), list(atomslice).index(b[1])) for b in self.bonds if (b[0] in atomslice and b[1] in atomslice)]
        return New

    def atom_stack(self, other):
        """ Return a copy of the object with another molecule object appended.  WARNING: This function may invalidate stuff like QM energies. """
        if len(other) != len(self):
            raise Exception('The number of frames of the Molecule objects being stacked are not equal.')

        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])
            
        # This is how we're going to stack things like Cartesian coordinates and QM forces.
        def FrameStack(k):
            if k in self.Data and k in other.Data:
                New.Data[k] = [np.vstack((s, o)) for s, o in zip(self.Data[k], other.Data[k])]
        for i in ['xyzs', 'qm_forces', 'qm_espxyzs', 'qm_espvals', 'qm_extchgs', 'qm_mulliken_charges', 'qm_mulliken_spins']:
            FrameStack(i)

        # Now build the new atom keys.
        for key in self.AtomKeys:
            if key not in other.Data:
                raise Exception('Trying to stack two Molecule objects - the first object contains %s and the other does not' % (key))
            if key == 'tinkersuf': # Tinker suffix is a bit tricky
                NewSuf = []
                for line in other.Data[key]:
                    whites      = re.split('[^ ]+',line)
                    s           = line.split()
                    if len(s) > 1:
                        for i in range(1,len(s)):
                            s[i] = str(int(s[i]) + self.na)
                    NewSuf.append(''.join([whites[j]+s[j] for j in range(len(s))]))
                New.Data[key] = deepcopy(self.Data[key]) + NewSuf
            else:
                if type(self.Data[key]) is np.ndarray:
                    New.Data[key] = np.concatenate((self.Data[key], other.Data[key]))
                elif type(self.Data[key]) is list:
                    New.Data[key] = self.Data[key] + other.Data[key]
                else:
                    raise Exception('Cannot stack %s because it is of type %s' % (key, str(type(New.Data[key]))))
        if 'bonds' in self.Data and 'bonds' in other.Data:
            New.Data['bonds'] = self.bonds + [(b[0]+self.na, b[1]+self.na) for b in other.bonds]
        return New

    def align_by_moments(self):
        """ Align molecules using the "moment of inertia."
        Note that we're following the MSMBuilder convention 
        of using all ones for the masses. """
        xyz1  = self.xyzs[0]
        xyz1 -= xyz1.mean(0)
        xyz1  = AlignToMoments(self.elem,xyz1)
        for index2, xyz2 in enumerate(self.xyzs):
            xyz2 -= xyz2.mean(0)
            xyz2 = AlignToMoments(self.elem,xyz1,xyz2)
            self.xyzs[index2] = xyz2

    def align(self, smooth = True, center = True):
        """ Align molecules. 
        
        Has the option to create smooth trajectories 
        (align each frame to the previous one)
        or to align each frame to the first one.

        Also has the option to remove the center of mass.

        """
        xyz1 = self.xyzs[0]
        if center:
            xyz1 -= xyz1.mean(0)
        for index2, xyz2 in enumerate(self.xyzs):
            if index2 == 0: continue
            xyz2 -= xyz2.mean(0)
            if smooth:
                ref = index2-1
            else:
                ref = 0
            tr, rt = get_rotate_translate(xyz2,self.xyzs[ref])
            xyz2 = np.dot(xyz2, rt) + tr
            self.xyzs[index2] = xyz2

    def build_topology(self, sn=None, Fac=1.2):
        ''' A bare-bones implementation of the bond graph capability in the nanoreactor code.  
        Returns a NetworkX graph that depicts the molecular topology, which might be useful for stuff. 
        Provide, optionally, the frame number used to compute the topology. '''
        if sn == None:
            sn = 0
        mindist = 1.0 # Any two atoms that are closer than this distance are bonded.
        # Create an atom-wise list of covalent radii.
        R = np.array([(Radii[Elements.index(i)-1] if i in Elements else 0.0) for i in self.elem])
        # Create a list of 2-tuples corresponding to combinations of atomic indices.
        # This is optimized and much faster than using itertools.combinations.
        AtomIterator = np.vstack((np.fromiter(itertools.chain(*[[i]*(self.na-i-1) for i in range(self.na)]),dtype=int), np.fromiter(itertools.chain(*[range(i+1,self.na) for i in range(self.na)]),dtype=int))).T
        # Create a list of thresholds for determining whether a certain interatomic distance is considered to be a bond.
        # BondThresh = np.fromiter([max(mindist,(R[i[0]] + R[i[1]])*Fac) for i in AtomIterator], dtype=float)
        BT0 = R[AtomIterator[:,0]]
        BT1 = R[AtomIterator[:,1]]
        BondThresh = (BT0+BT1) * Fac
        BondThresh = (BondThresh > mindist) * BondThresh + (BondThresh < mindist) * mindist
        try:
            dxij = contact.atom_distances(np.array([self.xyzs[sn]]),AtomIterator)
        except:
            # Extremely inefficient implementation if importing contact doesn't work.
            dxij = [np.array(list(itertools.chain(*[[np.linalg.norm(self.xyzs[sn][i]-self.xyzs[sn][j]) for i in range(j+1,self.na)] for j in range(self.na)])))]
        G = MyG()
        bonds = [[] for i in range(self.na)]
        lengths = []
        for i, a in enumerate(self.elem):
            G.add_node(i)
            nx.set_node_attributes(G,'e',{i:a})
            nx.set_node_attributes(G,'x',{i:self.xyzs[sn][i]})
        bond_bool = dxij[0] < BondThresh
        for i, a in enumerate(bond_bool):
            if not a: continue
            (ii, jj) = AtomIterator[i]
            bonds[ii].append(jj)
            bonds[jj].append(ii)
            G.add_edge(ii, jj)
            lengths.append(dxij[0][i])
        return G

    def measure_dihedrals(self, i, j, k, l):
        """ Return a series of dihedral angles, given four atom indices numbered from zero. """
        phis = []
        if 'bonds' in self.Data:
            if any(p not in self.bonds for p in [(min(i,j),max(i,j)),(min(j,k),max(j,k)),(min(k,l),max(k,l))]):
                print [(min(i,j),max(i,j)),(min(j,k),max(j,k)),(min(k,l),max(k,l))]
                warn("Measuring dihedral angle for four atoms that aren't bonded.  Hope you know what you're doing!")
        else:
            warn("This molecule object doesn't have bonds defined, sanity-checking is off.")
        for s in range(self.ns):
            x4 = self.xyzs[s][l]
            x3 = self.xyzs[s][k]
            x2 = self.xyzs[s][j]
            x1 = self.xyzs[s][i]
            v1 = x2-x1
            v2 = x3-x2
            v3 = x4-x3
            t1 = np.linalg.norm(v2)*np.dot(v1,np.cross(v2,v3))
            t2 = np.dot(np.cross(v1,v2),np.cross(v2,v3))
            phi = np.arctan2(t1,t2)
            phis.append(phi * 180 / np.pi)
            #phimod = phi*180/pi % 360
            #phis.append(phimod)
            #print phimod
        return phis

    def all_pairwise_rmsd(self):
        """ Find pairwise RMSD (super slow, not like the one in MSMBuilder.) """
        N = len(self)
        Mat = np.zeros((N,N),dtype=float)
        for i in range(N):
            xyzi = self.xyzs[i].copy()
            xyzi -= xyzi.mean(0)
            for j in range(i):
                xyzj = self.xyzs[j].copy()
                xyzj -= xyzj.mean(0)
                tr, rt = get_rotate_translate(xyzj, xyzi)
                xyzj = np.dot(xyzj, rt) + tr
                rmsd = np.sqrt(np.mean((xyzj - xyzi) ** 2))
                Mat[i,j] = rmsd
                Mat[j,i] = rmsd
        return Mat

    def align_center(self):
        self.align()

    def openmm_positions(self):
        """ Returns the Cartesian coordinates in the Molecule object in
        a list of OpenMM-compatible positions, so it is possible to type
        simulation.context.setPositions(Mol.openmm_positions()[0])
        or something like that.
        """

        Positions = []
        self.require('xyzs')
        for xyz in self.xyzs:
            Pos = []
            for xyzi in xyz:
                Pos.append(Vec3(xyzi[0]/10,xyzi[1]/10,xyzi[2]/10))
            Positions.append(Pos*nanometer)
        return Positions

    def openmm_boxes(self):
        """ Returns the periodic box vectors in the Molecule object in
        a list of OpenMM-compatible boxes, so it is possible to type
        simulation.context.setPeriodicBoxVectors(Mol.openmm_boxes()[0])
        or something like that.
        """
        
        self.require('boxes')
        return [(Vec3(box.A)/10.0, Vec3(box.B)/10.0, Vec3(box.C)/10.0) * nanometer for box in self.boxes]

    def split(self, fnm=None, ftype=None, method="chunks", num=None):

        """ Split the molecule object into a number of separate files
        (chunks), either by specifying the number of frames per chunk
        or the number of chunks.  Only relevant for "trajectories".
        The type of file may be specified; if they aren't specified
        then the original file type is used.

        The output file names are [name].[numbers].[extension] where
        [name] can be specified by passing 'fnm' or taken from the
        object's 'fnm' attribute by default.  [numbers] are integers
        ranging from the lowest to the highest chunk number, prepended
        by zeros.

        If the number of chunks / frames is not specified, then one file
        is written for each frame.
        
        @return fnms A list of the file names that were written.
        """
        
        

    #=====================================#
    #|         Reading functions         |#
    #=====================================#

    def read_xyz(self, fnm):
        """ Parse a .xyz file which contains several xyz coordinates, and return their elements.

        @param[in] fnm The input file name
        @return elem  A list of chemical elements in the XYZ file
        @return comms A list of comments.
        @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)

        """
        xyz   = []
        xyzs  = []
        comms = []
        elem  = []
        an    = 0
        na    = 0
        ln    = 0
        absln = 0
        for line in open(fnm):
            if ln == 0:
                # Skip blank lines.
                if len(line.strip()) > 0:
                    na = int(line.strip())
            elif ln == 1:
                comms.append(line.strip())
            else:
                sline = line.split()
                xyz.append([float(i) for i in sline[1:]])
                if len(elem) < na:
                    elem.append(sline[0])
                an += 1
                if an == na:
                    xyzs.append(np.array(xyz))
                    xyz = []
                    an  = 0
            if ln == na+1:
                # Reset the line number counter when we hit the last line in a block.
                ln = -1
            ln += 1
            absln += 1
        Answer = {'elem' : elem,
                  'xyzs' : xyzs,
                  'comms': comms}
        return Answer

    def read_mdcrd(self, fnm):
        """ Parse an AMBER .mdcrd file.  This requires at least the number of atoms.
        This will FAIL for monatomic trajectories (but who the heck makes those?)

        @param[in] fnm The input file name
        @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)
        @return boxes Boxes (if present.)

        """
        self.require('na')
        xyz    = []
        xyzs   = []
        boxes  = []
        ln     = 0
        for line in open(fnm):
            sline = line.split()
            if ln == 0:
                pass
            else:
                if xyz == [] and len(sline) == 3:
                    a, b, c = (float(i) for i in line.split())
                    boxes.append(BuildLatticeFromLengthsAngles(a, b, c, 90.0, 90.0, 90.0))
                else:
                    xyz += [float(i) for i in line.split()]
                    if len(xyz) == self.na * 3:
                        xyzs.append(np.array(xyz).reshape(-1,3))
                        xyz = []
            ln += 1
        Answer = {'xyzs' : xyzs}
        if len(boxes) > 0:
            Answer['boxes'] = boxes
        return Answer

    def read_qdata(self, fnm):
        xyzs     = []
        energies = []
        forces   = []
        espxyzs  = []
        espvals  = []
        interaction = []
        for line in open(fnm):
            if 'COORDS' in line:
                xyzs.append(np.array([float(i) for i in line.split()[1:]]).reshape(-1,3))
            elif 'FORCES' in line:
                forces.append(np.array([float(i) for i in line.split()[1:]]).reshape(-1,3))
            elif 'ESPXYZ' in line:
                espxyzs.append(np.array([float(i) for i in line.split()[1:]]).reshape(-1,3))
            elif 'ESPVAL' in line:
                espvals.append(np.array([float(i) for i in line.split()[1:]]))
            elif 'ENERGY' in line:
                energies.append(float(line.split()[1]))
            elif 'INTERACTION' in line:
                interaction.append(float(line.split()[1]))
        Answer = {}
        if len(xyzs) > 0:
            Answer['xyzs'] = xyzs
        if len(energies) > 0:
            Answer['qm_energies'] = energies
        if len(interaction) > 0:
            Answer['qm_interaction'] = interaction
        if len(forces) > 0:
            Answer['qm_forces'] = forces
        if len(espxyzs) > 0:
            Answer['qm_espxyzs'] = espxyzs
        if len(espvals) > 0:
            Answer['qm_espvals'] = espvals
        return Answer

    def read_mol2(self, fnm):
        xyz      = []
        charge   = []
        atomname = []
        atomtype = []
        elem     = []
        data = Mol2.mol2_set(fnm)
        if len(data.compounds) > 1:
            sys.stderr.write("Not sure what to do if the MOL2 file contains multiple compounds\n")
        for i, atom in enumerate(data.compounds.items()[0][1].atoms):
            xyz.append([atom.x, atom.y, atom.z])
            charge.append(atom.charge)
            atomname.append(atom.atom_name)
            atomtype.append(atom.atom_type)
            thiselem = atom.atom_name
            if len(thiselem) > 1:
                thiselem = thiselem[0] + re.sub('[A-Z0-9]','',thiselem[1:])
            elem.append(thiselem)

        resname = [data.compounds.items()[0][0] for i in range(len(elem))]
        resid = [1 for i in range(len(elem))]
        
        # Deprecated 'abonds' format.
        # bonds    = [[] for i in range(len(elem))]
        # for bond in data.compounds.items()[0][1].bonds:
        #     a1 = bond.origin_atom_id - 1
        #     a2 = bond.target_atom_id - 1
        #     aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
        #     bonds[aL].append(aH)

        bonds = []
        for bond in data.compounds.items()[0][1].bonds:
            a1 = bond.origin_atom_id - 1
            a2 = bond.target_atom_id - 1
            aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
            bonds.append((aL,aH))

        Answer = {'xyzs' : [np.array(xyz)],
                  'partial_charge' : charge,
                  'atomname' : atomname,
                  'atomtype' : atomtype,
                  'elem'     : elem,
                  'resname'  : resname,
                  'resid'    : resid,
                  'bonds'    : bonds
                  }

        return Answer

    def read_dcd(self, fnm):
        xyzs = []
        boxes = []
        if _dcdlib.vmdplugin_init() != 0:
            raise IOError("Unable to init DCD plugin")
        natoms = c_int(-1)
        frame  = 0
        dcd       = _dcdlib.open_dcd_read(fnm, "dcd", byref(natoms))
        ts        = MolfileTimestep()
        _xyz      = c_float * (natoms.value * 3)
        xyzvec    = _xyz()
        ts.coords = xyzvec
        while True:
            result = _dcdlib.read_next_timestep(dcd, natoms, byref(ts))
            if result == 0:
                frame += 1
            elif result == -1:
                break
            npa    = np.array(xyzvec)
            xyz    = np.asfarray(npa)
            xyzs.append(xyz.reshape(-1, 3))
            boxes.append(BuildLatticeFromLengthsAngles(ts.A, ts.B, ts.C, 90.0, 90.0, 90.0))
        _dcdlib.close_file_read(dcd)
        dcd = None
        Answer = {'xyzs' : xyzs,
                  'boxes' : boxes}
        return Answer

    def read_com(self, fnm):
        """ Parse a Gaussian .com file and return a SINGLE-ELEMENT list of xyz coordinates (no multiple file support)

        @param[in] fnm The input file name
        @return elem   A list of chemical elements in the XYZ file
        @return comms  A single-element list for the comment.
        @return xyzs   A single-element list for the  XYZ coordinates.
        @return charge The total charge of the system.
        @return mult   The spin multiplicity of the system.

        """
        elem    = []
        xyz     = []
        ln      = 0
        absln   = 0
        comfile = open(fnm).readlines()
        inxyz = 0
        for line in comfile:
            # Everything after exclamation point is a comment
            sline = line.split('!')[0].split()
            if len(sline) == 2:
                if isint(sline[0]) and isint(sline[1]):
                    charge = int(sline[0])
                    mult = int(sline[1])
                    title_ln = ln - 2
            elif len(sline) == 4:
                inxyz = 1
                if sline[0].capitalize() in PeriodicTable and isfloat(sline[1]) and isfloat(sline[2]) and isfloat(sline[3]):
                    elem.append(sline[0])
                    xyz.append(np.array([float(sline[1]),float(sline[2]),float(sline[3])]))
            elif inxyz:
                break
            ln += 1
            absln += 1

        Answer = {'xyzs'   : [np.array(xyz)],
                  'elem'   : elem,
                  'comms'  : [comfile[title_ln].strip()],
                  'charge' : charge,
                  'mult'   : mult}
        return Answer

    def read_arc(self, fnm):
        """ Read a TINKER .arc file.

        @param[in] fnm  The input file name
        @return xyzs    A list for the  XYZ coordinates.
        @return resid   The residue ID numbers.  These are not easy to get!
        @return elem    A list of chemical elements in the XYZ file
        @return comms   A single-element list for the comment.
        @return tinkersuf  The suffix that comes after lines in the XYZ coordinates; this is usually topology info

        """
        tinkersuf   = []
        xyzs  = []
        xyz   = []
        resid = []
        elem  = []
        comms = []
        thisres = set([])
        forwardres = set([])
        title = True
        nframes = 0
        thisresid   = 1
        ln = 0
        thisatom = 0
        for line in open(fnm):
            sline = line.split()
            if len(sline) == 0: continue
            # The first line always contains the number of atoms
            # The words after the first line are comments
            if title:
                na = int(sline[0])
                comms.append(' '.join(sline[1:]))
                title = False
            elif len(sline) >= 5:
                if isint(sline[0]) and isfloat(sline[2]) and isfloat(sline[3]) and isfloat(sline[4]): # A line of data better look like this
                    if nframes == 0:
                        elem.append(sline[1])
                        resid.append(thisresid)
                        whites      = re.split('[^ ]+',line)
                        if len(sline) > 5:
                            tinkersuf.append(''.join([whites[j]+sline[j] for j in range(5,len(sline))]))
                        else:
                            tinkersuf.append('')
                    # LPW Make sure ..
                    thisatom += 1
                    #thisatom = int(sline[0])
                    thisres.add(thisatom)
                    forwardres.add(thisatom)
                    if len(sline) >= 6:
                        forwardres.update([int(j) for j in sline[6:]])
                    if thisres == forwardres:
                        thisres = set([])
                        forwardres = set([])
                        thisresid += 1
                    xyz.append([float(sline[2]),float(sline[3]),float(sline[4])])
                    if thisatom == na:
                        thisatom = 0
                        nframes += 1
                        title = True
                        xyzs.append(np.array(xyz))
                        xyz = []
            ln += 1
        Answer = {'xyzs'   : xyzs,
                  'resid'  : resid,
                  'elem'   : elem,
                  'comms'  : comms,
                  'tinkersuf' : tinkersuf}
        return Answer

    def read_gro(self, fnm):
        """ Read a GROMACS .gro file.

        """
        xyzs     = []
        elem     = [] # The element, most useful for quantum chemistry calculations
        atomname = [] # The atom name, for instance 'HW1'
        comms    = []
        resid    = []
        resname  = []
        boxes    = []
        xyz      = []
        ln       = 0
        frame    = 0
        absln    = 0
        for line in open(fnm):
            sline = line.split()
            if ln == 0:
                comms.append(line.strip())
            elif ln == 1:
                na = int(line.strip())
            elif is_gro_coord(line):
                if frame == 0: # Create the list of residues, atom names etc. only if it's the first frame.
                    # Name of the residue, for instance '153SOL1 -> SOL1' ; strips leading numbers
                    thisresname = re.sub('^[0-9]*','',sline[0])
                    resname.append(thisresname)
                    resid.append(int(sline[0].replace(thisresname,'')))
                    atomname.append(sline[1])
                    thiselem = sline[1]
                    if len(thiselem) > 1:
                        thiselem = thiselem[0] + re.sub('[A-Z0-9]','',thiselem[1:])
                    elem.append(thiselem)
                xyz.append([float(i) for i in sline[-3:]])
            elif is_gro_box(line) and ln == na + 2:
                box = [float(i)*10 for i in sline]
                if len(box) == 3:
                    a = box[0]
                    b = box[1]
                    c = box[2]
                    alpha = 90.0
                    beta = 90.0
                    gamma = 90.0
                    boxes.append(BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma))
                elif len(box) == 9:
                    v1 = np.array([box[0], box[3], box[4]])
                    v2 = np.array([box[5], box[1], box[6]])
                    v3 = np.array([box[7], box[8], box[2]])
                    boxes.append(BuildLatticeFromVectors(v1, v2, v3))
                xyzs.append(np.array(xyz)*10)
                xyz = []
                ln = -1
                frame += 1
            ln += 1
            absln += 1
        Answer = {'xyzs'     : xyzs,
                  'elem'     : elem,
                  'atomname' : atomname,
                  'resid'    : resid,
                  'resname'  : resname,
                  'boxes'    : boxes,
                  'comms'    : comms}
        return Answer

    def read_charmm(self, fnm):
        """ Read a CHARMM .cor (or .crd) file.

        """
        xyzs     = []
        elem     = [] # The element, most useful for quantum chemistry calculations
        atomname = [] # The atom name, for instance 'HW1'
        comms    = []
        resid    = []
        resname  = []
        xyz      = []
        thiscomm = []
        ln       = 0
        frame    = 0
        an       = 0
        for line in open(fnm):
            sline = line.split()
            if re.match('^\*',line):
                if len(sline) == 1:
                    comms.append(';'.join(list(thiscomm)))
                    thiscomm = []
                else:
                    thiscomm.append(' '.join(sline[1:]))
            elif re.match('^ *[0-9]+ +(EXT)?$',line):
                na = int(sline[0])
            elif is_charmm_coord(line):
                if frame == 0: # Create the list of residues, atom names etc. only if it's the first frame.
                    resid.append(sline[1])
                    resname.append(sline[2])
                    atomname.append(sline[3])
                    thiselem = sline[3]
                    if len(thiselem) > 1:
                        thiselem = thiselem[0] + re.sub('[A-Z0-9]','',thiselem[1:])
                    elem.append(thiselem)
                xyz.append([float(i) for i in sline[4:7]])
                an += 1
                if an == na:
                    xyzs.append(np.array(xyz))
                    xyz = []
                    an = 0
                    frame += 1
            ln += 1
        Answer = {'xyzs'     : xyzs,
                  'elem'     : elem,
                  'atomname' : atomname,
                  'resid'    : resid,
                  'resname'  : resname,
                  'comms'    : comms}
        return Answer

    def read_qcin(self, fnm):
        """ Read a Q-Chem input file.

        These files can be very complicated, and I can't write a completely
        general parser for them.  It is important to keep our goal in
        mind:

        1) The main goal is to convert a trajectory to Q-Chem input
        files with identical calculation settings.

        2) When we print the Q-Chem file, we should preserve the line
        ordering of the 'rem' section, but also be able to add 'rem'
        options at the end.

        3) We should accommodate the use case that the Q-Chem file may have
        follow-up calculations delimited by '@@@@'.

        4) We can read in all of the xyz's as a trajectory, but only the
        Q-Chem settings belonging to the first xyz will be saved.

        """

        qcrem                = OrderedDict()
        qcrems               = []
        xyz                  = []
        xyzs                 = []
        elem                 = []
        section              = None
        # The Z-matrix printing in new versions throws me off.
        zmatrix              = False
        template             = []
        fff = False
        inside_section       = False
        reading_template     = True
        charge               = 0
        mult                 = 0
        Answer               = {}
        SectionData          = []
        template_cut         = 0
        readsuf              = True
        suffix               = [] # The suffix, which comes after every atom line in the $molecule section, is for determining the MM atom type and topology.
        ghost                = [] # If the element in the $molecule section is preceded by an '@' sign, it's a ghost atom for counterpoise calculations.

        for line in open(fnm).readlines():
            line = line.strip().expandtabs()
            sline = line.split()
            dline = line.split('!')[0].split()
            if "Z-matrix Print" in line:
                zmatrix = True
            if re.match('^\$',line):
                wrd = re.sub('\$','',line)
                if wrd == 'end':
                    zmatrix = False
                    inside_section = False
                    if section == 'molecule':
                        if len(xyz) > 0:
                            xyzs.append(np.array(xyz))
                        xyz = []
                        fff = True
                        if suffix != []:
                            readsuf = False
                    elif section == 'rem':
                        if reading_template:
                            qcrems.append(qcrem)
                            qcrem = OrderedDict()
                    if reading_template:
                        if section != 'external_charges': # Ignore the external charges section because it varies from frame to frame.
                            template.append((section,SectionData))
                    SectionData = []
                else:
                    section = wrd
                    inside_section = True
            elif inside_section:
                if section == 'molecule' and not zmatrix:
                    if len(dline) >= 4 and all([isfloat(dline[i]) for i in range(1,4)]):
                        if fff:
                            reading_template = False
                            template_cut = list(i for i, dat in enumerate(template) if dat[0] == '@@@@')[-1]
                        else:
                            if re.match('^@', sline[0]): # This is a ghost atom
                                ghost.append(True)
                            else:
                                ghost.append(False)
                            elem.append(re.sub('@','',sline[0]))
                        xyz.append([float(i) for i in sline[1:4]])
                        if readsuf and len(sline) > 4:
                            whites      = re.split('[^ ]+',line)
                            suffix.append(''.join([whites[j]+sline[j] for j in range(4,len(sline))]))
                    elif re.match("[+-]?[0-9]+ +[0-9]+$",line.split('!')[0].strip()):
                        if not fff:
                            charge = int(sline[0])
                            mult = int(sline[1])
                    else:
                        SectionData.append(line)
                elif reading_template and not zmatrix:
                    if section == 'basis':
                        SectionData.append(line.split('!')[0])
                    elif section == 'rem':
                        S = splitter.findall(line)
                        if S[0] == '!':
                            qcrem[''.join(S[0:3]).lower()] = ''.join(S[4:])
                        else:
                            qcrem[S[0].lower()] = ''.join(S[2:])
                    else:
                        SectionData.append(line)
            elif re.match('^@+$', line) and reading_template:
                template.append(('@@@@', []))
            elif re.match('Welcome to Q-Chem', line) and reading_template and fff:
                template.append(('@@@@', []))

        if template_cut != 0:
            template = template[:template_cut]

        Answer = {'qctemplate'  : template,
                  'qcrems'      : qcrems,
                  'charge'      : charge,
                  'mult'        : mult,
                  }
        if suffix != []:
            Answer['qcsuf'] = suffix

        if len(xyzs) > 0:
            Answer['xyzs'] = xyzs
        if len(elem) > 0:
            Answer['elem'] = elem
        if len(ghost) > 0:
            Answer['qm_ghost'] = ghost
        return Answer


    def read_pdb(self, fnm):
        """ Loads a PDB and returns a dictionary containing its data. """

        F1=file(fnm,'r')
        ParsedPDB=readPDB(F1)

        Box = None
        #Separate into distinct lists for each model.
        PDBLines=[[]]
        for x in ParsedPDB[0]:
            if x.__class__ in [END, ENDMDL]:
                PDBLines.append([])
            if x.__class__ in [ATOM, HETATM]:
                PDBLines[-1].append(x)
            if x.__class__==CRYST1:
                Box = BuildLatticeFromLengthsAngles(x.a, x.b, x.c, x.alpha, x.beta, x.gamma)

        X=PDBLines[0]

        XYZ=np.array([[x.x,x.y,x.z] for x in X])/10.0#Convert to nanometers
        AltLoc=np.array([x.altLoc for x in X],'str') # Alternate location
        ICode=np.array([x.iCode for x in X],'str') # Insertion code
        ChainID=np.array([x.chainID for x in X],'str')
        AtomNames=np.array([x.name for x in X],'str')
        ResidueNames=np.array([x.resName for x in X],'str')
        ResidueID=np.array([x.resSeq for x in X],'int')
        # Try not to number Residue IDs starting from 1...
        if self.positive_resid:
            ResidueID=ResidueID-ResidueID[0]+1

        XYZList=[]
        for Model in PDBLines:
            XYZList.append([])
            for x in Model:
                XYZList[-1].append([x.x,x.y,x.z])

        if len(XYZList[-1])==0:#If PDB contains trailing END / ENDMDL, remove empty list
            XYZList.pop()

        # Build a list of chemical elements
        elem = []
        for i in AtomNames:
            thiselem = i
            if len(thiselem) > 1:
                thiselem = thiselem[0] + re.sub('[A-Z0-9]','',thiselem[1:])
            elem.append(thiselem)

        XYZList=list(np.array(XYZList).reshape((-1,len(ChainID),3)))

        bonds = []
        # Read in CONECT records.
        F2=open(fnm,'r')
        for line in F2:
            s = line.split()
            if s[0].upper() == "CONECT":
                if len(s) > 2:
                    for i in range(2, len(s)):
                        bonds.append((int(s[1])-1, int(s[i])-1))

        Answer={"xyzs":XYZList, "chain":ChainID, "altloc":AltLoc, "icode":ICode, "atomname":AtomNames,
                "resid":ResidueID, "resname":ResidueNames, "elem":elem,
                "comms":['' for i in range(len(XYZList))]}

        if len(bonds) > 0:
            Answer["bonds"] = bonds
        if Box != None:
            Answer["boxes"] = [Box for i in range(len(XYZList))]

        return Answer

    def read_qcesp(self, fnm):
        espxyz = []
        espval = []
        for line in open(fnm):
            sline = line.split()
            if len(sline) == 4 and all([isfloat(sline[i]) for i in range(4)]):
                espxyz.append([float(sline[i]) for i in range(3)])
                espval.append(float(sline[3]))
        Answer = {'qm_espxyzs' : [np.array(espxyz) * bohrang],
                  'qm_espvals'  : [np.array(espval)]
                  }
        return Answer
    
    def read_qcout(self, fnm):
        """ Q-Chem output file reader, adapted for our parser. 
    
        Q-Chem output files are very flexible and there's no way I can account for all of them.  Here's what
        I am able to account for:
        
        A list of:
        - Coordinates
        - Energies
        - Forces
        
        Note that each step in a geometry optimization counts as a frame.
    
        As with all Q-Chem output files, note that successive calculations can have different numbers of atoms.
    
        """
    
        xyzs     = []
        xyz      = []
        elem     = []
        elemThis = []
        mkchg    = []
        mkspn    = []
        mkchgThis= []
        mkspnThis= []
        XMode    = 0
        MMode    = 0
        conv     = []
        convThis = 0
        readChargeMult = 0
        energy_scf = []
        float_match  = {'energy_scfThis'   : ("^[1-9][0-9]* +[-+]?([0-9]*\.)?[0-9]+ +[-+]?([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)[A-Za-z0 ]*$", 1),
                        'energy_opt'       : ("^Final energy is +[-+]?([0-9]*\.)?[0-9]+$", -1),
                        'charge'           : ("Sum of atomic charges", -1),
                        'mult'             : ("Sum of spin +charges", -1),
                        'energy_mp2'       : ("^(ri)*(-)*mp2 +total energy += +[-+]?([0-9]*\.)?[0-9]+ +au$",-2),
                        'energy_ccsd'      : ("^CCSD Total Energy += +[-+]?([0-9]*\.)?[0-9]+$",-1),
                        'energy_ccsdt'     : ("^CCSD\(T\) Total Energy += +[-+]?([0-9]*\.)?[0-9]+$",-1)
                        }
        matrix_match = {'analytical_grad'  :'Full Analytical Gradient',
                        'gradient_scf'     :'Gradient of SCF Energy',
                        'gradient_mp2'     :'Gradient of MP2 Energy',
                        'gradient_dualbas' :'Gradient of the Dual-Basis Energy'
                        }
        qcrem    = OrderedDict()

        matblank   = {'match' : '', 'All' : [], 'This' : [], 'Strip' : [], 'Mode' : 0}
        Mats      = {}
        Floats    = {}
        for key, val in matrix_match.items():
            Mats[key] = matblank.copy()
        for key, val in float_match.items():
            Floats[key] = []
    
        for line in open(fnm):
            line = line.strip().expandtabs()
            if 'fatal error' in line:
                raise Exception('Calculation encountered a fatal error!')
            if XMode >= 1:
                # Perfectionist here; matches integer, element, and three floating points
                if re.match("^[0-9]+ +[A-Z][a-z]?( +[-+]?([0-9]*\.)?[0-9]+){3}$", line):
                    XMode = 2
                    sline = line.split()
                    elemThis.append(sline[1])
                    xyz.append([float(i) for i in sline[2:]])
                elif XMode == 2: # Break out of the loop if we encounter anything other than atomic data
                    if elem == []:
                        elem = elemThis
                    elif elem != elemThis:
                        raise Exception('Q-Chem output parser will not work if successive calculations have different numbers of atoms!')
                    elemThis = []
                    xyzs.append(np.array(xyz))
                    xyz  = []
                    XMode = 0
            if MMode >= 1:
                # Perfectionist here; matches integer, element, and two floating points
                if re.match("^[0-9]+ +[A-Z][a-z]?( +[-+]?([0-9]*\.)?[0-9]+){2}$", line):
                    MMode = 2
                    sline = line.split()
                    mkchgThis.append(float(sline[2]))
                    mkspnThis.append(float(sline[3]))
                elif MMode == 2: # Break out of the loop if we encounter anything other than Mulliken charges
                    mkchg.append(mkchgThis[:])
                    mkspn.append(mkspnThis[:])
                    mkchgThis = []
                    mkspnThis = []
                    MMode = 0
            elif re.match("Standard Nuclear Orientation".lower(), line.lower()):
                XMode = 1
            elif re.match("Ground-State Mulliken Net Atomic Charges".lower(), line.lower()):
                MMode = 1
            for key, val in float_match.items():
                if re.match(val[0].lower(), line.lower()):
                    Floats[key].append(float(line.split()[val[1]]))
            if re.match(".*Convergence criterion met$".lower(), line.lower()):
                conv.append(1)
                energy_scf.append(Floats['energy_scfThis'][-1])
                Floats['energy_scfThis'] = []
            elif re.match(".*Convergence failure$".lower(), line.lower()):
                conv.append(0)
                Floats['energy_scfThis'] = []
            for key, val in matrix_match.items():
                if Mats[key]["Mode"] >= 1:
                    if re.match("^[0-9]+( +[0-9]+)+$",line):
                        # print "A", line
                        Mats[key]["This"] = add_strip_to_mat(Mats[key]["This"],Mats[key]["Strip"])
                        Mats[key]["Strip"] = []
                    elif re.match("^[0-9]+( +[-+]?([0-9]*\.)?[0-9]+)+$",line):
                        # print "B", line
                        Mats[key]["Strip"].append([float(i) for i in line.split()[1:]])
                    else:
                        # print "C", line
                        Mats[key]["This"] = add_strip_to_mat(Mats[key]["This"],Mats[key]["Strip"])
                        Mats[key]["Strip"] = []
                        Mats[key]["All"].append(np.array(Mats[key]["This"]))
                        Mats[key]["This"] = []
                        Mats[key]["Mode"] = 0
                elif re.match(val.lower(), line.lower()):
                    # print "D", line
                    Mats[key]["Mode"] = 1

        if len(Floats['mult']) == 0:
            Floats['mult'] = [0]

        # Copy out the element and coordinate lists
        Answer = {'elem':elem, 'xyzs':xyzs}
        # Read the output file as an input file to get a Q-Chem template.
        Aux = self.read_qcin(fnm)
        Answer['qctemplate'] = Aux['qctemplate']
        Answer['qcrems'] = Aux['qcrems']
        if 'qm_ghost' in Aux:
            Answer['qm_ghost'] = Aux['qm_ghost']
        # Copy out the charge and multiplicity
        Answer['charge'] = int(Floats['charge'][0])
        Answer['mult']   = int(Floats['mult'][0]) + 1
        # Copy out the energies and forces
        # Q-Chem can print out gradients with several different headings.
        # We start with the most reliable heading and work our way down.
        if len(Mats['analytical_grad']['All']) > 0:
            Answer['qm_forces'] = Mats['analytical_grad']['All']
        if len(Mats['gradient_mp2']['All']) > 0:
            Answer['qm_forces'] = Mats['gradient_mp2']['All']
        elif len(Mats['gradient_dualbas']['All']) > 0:
            Answer['qm_forces'] = Mats['gradient_dualbas']['All']
        elif len(Mats['gradient_scf']['All']) > 0:
            Answer['qm_forces'] = Mats['gradient_scf']['All']
        #else:
        #    raise Exception('There are no forces in %s' % fnm)
        # Also work our way down with the energies.
        if len(Floats['energy_ccsdt']) > 0:
            Answer['qm_energies'] = Floats['energy_ccsdt']
        elif len(Floats['energy_ccsd']) > 0:
            Answer['qm_energies'] = Floats['energy_ccsd']
        elif len(Floats['energy_mp2']) > 0:
            Answer['qm_energies'] = Floats['energy_mp2']
        elif len(energy_scf) > 0:
            if 'correlation' in Answer['qcrems'][0] and Answer['qcrems'][0]['correlation'].lower() in ['mp2', 'rimp2', 'ccsd', 'ccsd(t)']:
                raise Exception("Q-Chem was called with a post-HF theory but we only got the SCF energy")
            Answer['qm_energies'] = energy_scf
        else:
            raise Exception('There are no energies in %s' % fnm)
    
        #### Sanity checks
        # We currently don't have a graceful way of dealing with SCF convergence failures in the output file.
        # For instance, a failed calculation will have elem / xyz but no forces. :/
        if 0 in conv:
            raise Exception('SCF convergence failure encountered in parsing %s' % fnm)
        # The molecule should have only one charge and one multiplicity
        if len(set(Floats['charge'])) != 1 or len(set(Floats['mult'])) != 1:
            raise Exception('Unexpected number of charges or multiplicities in parsing %s' % fnm)
        lens = [len(i) for i in Answer['qm_energies'], Answer['xyzs']]
        if len(set(lens)) != 1:
            raise Exception('The number of energies and coordinates in %s are not the same : %s' % (fnm, str(lens)))
        # The number of atoms should all be the same
        if len(set([len(i) for i in Answer['xyzs']])) != 1:
            raise Exception('The numbers of atoms across frames in %s are not all the same' % (fnm))

        if 'qm_forces' in Answer:
            for i, frc in enumerate(Answer['qm_forces']):
                Answer['qm_forces'][i] = frc.T
        # A strange peculiarity; Q-Chem sometimes prints out the final Mulliken charges a second time, after the geometry optimization.
        if mkchg != []:
            Answer['qm_mulliken_charges'] = np.array(mkchg[:len(Answer['qm_energies'])])
        if mkspn != []:
            Answer['qm_mulliken_spins'] = np.array(mkspn[:len(Answer['qm_energies'])])

        return Answer
    
    #=====================================#
    #|         Writing functions         |#
    #=====================================#

    def write_qcin(self, select):
        self.require('qctemplate','qcrems','charge','mult','xyzs','elem')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            remidx = 0
            molecule_printed = False
            # Each 'extchg' has number_of_atoms * 4 elements corresponding to x, y, z, q.
            if 'qm_extchgs' in self.Data:
                extchg = self.qm_extchgs[I]
                out.append('$external_charges')
                print extchg.shape
                for i in range(len(extchg)):
                    out.append("% 15.10f % 15.10f % 15.10f %15.10f" % (extchg[i,0],extchg[i,1],extchg[i,2],extchg[i,3]))
                out.append('$end')
            for SectName, SectData in self.qctemplate:
                if SectName != '@@@@':
                    out.append('$%s' % SectName)
                    for line in SectData:
                        out.append(line)
                    if SectName == 'molecule':
                        if molecule_printed == False:
                            molecule_printed = True
                            out.append("%i %i" % (self.charge, self.mult))
                            an = 0
                            for e, x in zip(self.elem, xyz):
                                pre = '@' if ('qm_ghost' in self.Data and self.Data['qm_ghost'][an]) else ''
                                suf =  self.Data['qcsuf'][an] if 'qcsuf' in self.Data else ''
                                out.append(pre + format_xyz_coord(e, x) + suf)
                                an += 1
                    if SectName == 'rem':
                        for key, val in self.qcrems[remidx].items():
                            out.append("%-21s %-s" % (key, str(val)))
                        remidx += 1
                    if SectName == 'comments' and 'comms' in self.Data:
                        out.append(self.comms[I])
                    out.append('$end')
                else:
                    out.append('@@@@')
                out.append('')
            #if I < (len(self) - 1):
            if I != select[-1]:
                out.append('@@@@')
                out.append('')
        return out

    def write_xyz(self, select):
        self.require('elem','xyzs')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            out.append("%-5i" % self.na)
            out.append(self.comms[I])
            for i in range(self.na):
                out.append(format_xyz_coord(self.elem[i],xyz[i]))
        return out

    def write_molproq(self, select):
        self.require('xyzs','partial_charge')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            # Comment comes first, then number of atoms.
            out.append(self.comms[I])
            out.append("%-5i" % self.na)
            for i in range(self.na):
                out.append("% 15.10f % 15.10f % 15.10f % 15.10f   0" % (xyz[i,0],xyz[i,1],xyz[i,2],self.partial_charge[i]))
        return out

    def write_mdcrd(self, select):
        self.require('xyzs')
        # In mdcrd files, there is only one comment line
        out = ['mdcrd file generated using ForceBalance'] 
        for I in select:
            xyz = self.xyzs[I]
            out += [''.join(["%8.3f" % i for i in g]) for g in grouper(10, list(xyz.flatten()))]
            if 'boxes' in self.Data:
                out.append(''.join(["%8.3f" % i for i in [self.boxes[I].a, self.boxes[I].b, self.boxes[I].c]]))
        return out

    def write_arc(self, select):
        self.require('elem','xyzs')
        out = []
        if 'tinkersuf' not in self.Data:
            sys.stderr.write("Beware, this .arc file contains no atom type or topology info\n")
        for I in select:
            xyz = self.xyzs[I]
            out.append("%6i  %s" % (self.na, self.comms[I]))
            for i in range(self.na):
                out.append("%6i  %s%s" % (i+1,format_xyz_coord(self.elem[i],xyz[i],tinker=True),self.tinkersuf[i] if 'tinkersuf' in self.Data else ''))
        return out

    def write_gro(self, select):
        self.require('elem','xyzs')
        out = []
        self.require_resname()
        self.require_resid()
        self.require_boxes()

        if 'atomname' not in self.Data:
            atomname = ["%s%i" % (self.elem[i], i+1) for i in range(self.na)]
        else:
            atomname = self.atomname

        for I in select:
            xyz = self.xyzs[I]
            xyzwrite = xyz.copy()
            xyzwrite /= 10.0 # GROMACS uses nanometers
            out.append(self.comms[I])
            #out.append("Generated by ForceBalance from %s" % self.fnm)
            out.append("%5i" % self.na)
            for an, line in enumerate(xyzwrite):
                out.append(format_gro_coord(self.resid[an],self.resname[an],atomname[an],an+1,xyzwrite[an]))
            out.append(format_gro_box(self.boxes[I]))
        return out

    def write_dcd(self, select):
        if _dcdlib.vmdplugin_init() != 0:
            raise IOError("Unable to init DCD plugin")
        natoms    = c_int(self.na)
        dcd       = _dcdlib.open_dcd_write(self.fout, "dcd", natoms)
        ts        = MolfileTimestep()
        _xyz      = c_float * (natoms.value * 3)
        for I in select:
            xyz = self.xyzs[I]
            ts.coords = _xyz(*list(xyz.flatten()))
            ts.A      = self.boxes[I].a if 'boxes' in self.Data else 1.0
            ts.B      = self.boxes[I].b if 'boxes' in self.Data else 1.0
            ts.C      = self.boxes[I].c if 'boxes' in self.Data else 1.0
            result    = _dcdlib.write_timestep(dcd, byref(ts))
            if result != 0:
                raise IOError("Error encountered when writing DCD")
        ## Close the DCD file
        _dcdlib.close_file_write(dcd)
        dcd = None

    def write_pdb(self, select):
        """Save to a PDB. Copied wholesale from MSMBuilder.
        COLUMNS  TYPE   FIELD  DEFINITION
        ---------------------------------------------
        7-11      int   serial        Atom serial number.
        13-16     string name          Atom name.
        17        string altLoc        Alternate location indicator.
        18-20 (17-21 KAB)    string resName       Residue name.
        22        string chainID       Chain identifier.
        23-26     int    resSeq        Residue sequence number.
        27        string iCode         Code for insertion of residues.
        31-38     float  x             Orthogonal coordinates for X in
        Angstroms.
        39-46     float  y             Orthogonal coordinates for Y in
        Angstroms.
        47-54     float  z             Orthogonal coordinates for Z in
        Angstroms.
        55-60     float  occupancy     Occupancy.
        61-66     float  tempFactor    Temperature factor.
        73-76     string segID         Segment identifier, left-justified.
        77-78     string element       Element symbol, right-justified.
        79-80     string charge        Charge on the atom.

        CRYST1 line, added by Lee-Ping
        COLUMNS  TYPE   FIELD  DEFINITION
        ---------------------------------------
         7-15    float  a      a (Angstroms).
        16-24    float  b      b (Angstroms).
        25-33    float  c      c (Angstroms).
        34-40    float  alpha  alpha (degrees).
        41-47    float  beta   beta (degrees).
        48-54    float  gamma  gamma (degrees).
        56-66    string sGroup Space group.
        67-70    int    z      Z value.

        """
        self.require('xyzs')
        self.require_resname()
        self.require_resid()
        ATOMS = self.atomname if 'atomname' in self.Data else ["%s%i" % (self.elem[i], i+1) for i in range(self.na)]
        CHAIN = self.chain if 'chain' in self.Data else [1 for i in range(self.na)]
        RESNAMES = self.resname
        RESNUMS = self.resid

        out = []
        if min(RESNUMS) == 0:
            RESNUMS = [i+1 for i in RESNUMS]

        if 'boxes' in self.Data:
            a = self.boxes[0].a
            b = self.boxes[0].b
            c = self.boxes[0].c
            alpha = self.boxes[0].alpha
            beta = self.boxes[0].beta
            gamma = self.boxes[0].gamma
            line=np.chararray(80)
            line[:] = ' '
            line[0:6]=np.array(list("CRYST1"))
            line=np.array(line,'str')
            line[6:15] =np.array(list(("%9.3f"%(a))))
            line[15:24]=np.array(list(("%9.3f"%(b))))
            line[24:33]=np.array(list(("%9.3f"%(c))))
            line[33:40]=np.array(list(("%7.2f"%(alpha))))
            line[40:47]=np.array(list(("%7.2f"%(beta))))
            line[47:54]=np.array(list(("%7.2f"%(gamma))))
            # LPW: Put in a dummy space group, we never use it.
            line[55:66]=np.array(list(str("P 21 21 21").rjust(11)))
            line[66:70]=np.array(list(str(4).rjust(4)))
            out.append(line.tostring())
            
        for I in select:
            XYZ = self.xyzs[I]
            for i in range(self.na):
                ATOMNUM = i + 1
                line=np.chararray(80)
                line[:]=' '
                line[0:4]=np.array(list("ATOM"))
                line=np.array(line,'str')
                line[6:11]=np.array(list(str(ATOMNUM%100000).rjust(5)))
                # if ATOMNUM < 100000:
                #     line[6:11]=np.array(list(str(ATOMNUM%100000).rjust(5)))
                # else:
                #     line[6:11]=np.array(list(hex(ATOMNUM)[2:].rjust(5)))
                #Molprobity is picky about atom name centering
                if len(str(ATOMS[i]))==3:
                    line[12:16]=np.array(list(str(ATOMS[i]).rjust(4)))
                elif len(str(ATOMS[i]))==2:
                    line[12:16]=np.array(list(" "+str(ATOMS[i])+" "))
                elif len(str(ATOMS[i]))==1:
                    line[12:16]=np.array(list(" "+str(ATOMS[i])+"  "))
                else:
                    line[12:16]=np.array(list(str(ATOMS[i]).center(4)))
                if len(str(RESNAMES[i]))==3:
                    line[17:20]=np.array(list(str(RESNAMES[i])))
                else:
                    line[17:21]=np.array(list(str(RESNAMES[i]).ljust(4)))

                line[21]=str(CHAIN[i]).rjust(1)
                line[22:26]=np.array(list(str(RESNUMS[i]%10000).rjust(4)))
                # if RESNUMS[i] < 100000:
                #     line[22:26]=np.array(list(str(RESNUMS[i]).rjust(4)))
                # else:
                #     line[22:26]=np.array(list(hex(RESNUMS[i])[2:].rjust(4)))

                x=XYZ[i][0]
                y=XYZ[i][1]
                z=XYZ[i][2]
                sx=np.sign(x)
                sy=np.sign(y)
                sz=np.sign(z)

                line[30:38]=np.array(list(("%8.3f"%(x))))
                line[38:46]=np.array(list(("%8.3f"%(y))))
                line[46:54]=np.array(list(("%8.3f"%(z))))

                if ATOMNUM!=-1:
                    out.append(line.tostring())
            out.append('ENDMDL')
        if 'bonds' in self.Data:
            connects = ["CONECT%5i" % (b0+1) + "".join(["%5i" % (b[1]+1) for b in self.bonds if b[0] == b0]) for b0 in sorted(list(set(b[0] for b in self.bonds)))]
            out += connects
        return out
        
    def write_qdata(self, select):
        """ Text quantum data format. """
        #self.require('xyzs','qm_energies','qm_forces')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            out.append("JOB %i" % I)
            out.append("COORDS"+pvec(xyz))
            if 'qm_energies' in self.Data:
                out.append("ENERGY % .12e" % self.qm_energies[I])
            if 'mm_energies' in self.Data:
                out.append("EMD0   % .12e" % self.mm_energies[I])
            if 'qm_forces' in self.Data:
                out.append("FORCES"+pvec(self.qm_forces[I]))
            if 'qm_espxyzs' in self.Data and 'qm_espvals' in self.Data:
                out.append("ESPXYZ"+pvec(self.qm_espxyzs[I]))
                out.append("ESPVAL"+pvec(self.qm_espvals[I]))
            if 'qm_interaction' in self.Data: 
                out.append("INTERACTION % .12e" % self.qm_interaction[I])
            out.append('')
        return out

    def require_resid(self):
        if 'resid' not in self.Data:
            na_res = int(raw_input("Enter how many atoms are in a residue, or zero as a single residue -> "))
            if na_res == 0:
                self.resid = [1 for i in range(self.na)]
            else:
                self.resid = [1 + i/na_res for i in range(self.na)]
            
    def require_resname(self):
        if 'resname' not in self.Data:
            resname = raw_input("Enter a residue name (3-letter like 'SOL') -> ")
            self.resname = [resname for i in range(self.na)]
            
    def require_boxes(self):
        def buildbox(line):
            s = [float(i) for i in line.split()]
            if len(s) == 1:
                a = s[0]
                b = s[0]
                c = s[0]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 3:
                a = s[0]
                b = s[1]
                c = s[2]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 6:
                a = s[0]
                b = s[1]
                c = s[2]
                alpha = s[3]
                beta = s[4]
                gamma = s[5]
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 9:
                v1 = np.array([s[0], s[3], s[4]])
                v2 = np.array([s[5], s[1], s[6]])
                v3 = np.array([s[7], s[8], s[2]])
                return BuildLatticeFromVectors(v1, v2, v3)
            else:
                raise Exception("Not sure what to do since you gave me %i numbers" % len(s))
            
        if 'boxes' not in self.Data or len(self.boxes) != self.ns:
            sys.stderr.write("Please specify the periodic box using:\n")
            sys.stderr.write("1 float (cubic lattice length in Angstrom)\n")
            sys.stderr.write("3 floats (orthogonal lattice lengths in Angstrom)\n")
            sys.stderr.write("6 floats (triclinic lattice lengths and angles in degrees)\n")
            sys.stderr.write("9 floats (triclinic lattice vectors v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y) in Angstrom)\n")
            sys.stderr.write("Or: Name of a file containing one of these lines for each frame in the trajectory\n")
            boxstr = raw_input("Box Vector Input: -> ")
            if os.path.exists(boxstr):
                boxfile = open(boxstr).readlines()
                if len(boxfile) != len(self):
                    raise Exception('Tried to read in the box file, but it has a different length from the number of frames.')
                else:
                    self.boxes = [buildbox(line) for line in boxfile]
            else:
                mybox = buildbox(boxstr)
                self.boxes = [mybox for i in range(self.ns)]

def main():
    print "Basic usage as an executable: molecule.py input.format1 output.format2"
    print "where format stands for xyz, pdb, gro, etc."
    Mao = Molecule(sys.argv[1])
    Mao.write(sys.argv[2])

if __name__ == "__main__":
    main()
