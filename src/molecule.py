#======================================================================#
#|                                                                    |#
#|              Chemical file format conversion module                |#
#|                                                                    |#
#|                Lee-Ping Wang (leeping@stanford.edu)                |#
#|                  Last updated October 29, 2012                     |#
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
FrameVariableNames = set(['xyzs', 'comms', 'boxes', 'qm_forces', 'qm_energies', 'qm_interaction', 'qm_espxyzs', 'qm_espvals', 'qm_extchgs'])
#=========================================#
#| Data attributes in AtomVariableNames  |#
#| must be a list along the atom axis,   |#
#| and they must have the same length.   |#
#=========================================#
# elem       = List of elements
# partial_charge = List of atomic partial charges 
# atomname   = List of atom names (can come from MM coordinate file)
# atomtype   = List of atom types (can come from MM force field)
# bonds      = For each atom, the list of higher-numbered atoms it's bonded to
# suffix     = String that comes after the XYZ coordinates in TINKER .xyz or .arc files
# resid      = Residue IDs (can come from MM coordinate file)
# resname    = Residue names
AtomVariableNames = set(['elem', 'partial_charge', 'atomname', 'atomtype', 'bonds', 'suffix', 'resid', 'resname', 'qcsuf', 'qm_ghost'])
#=========================================#
#| This can be any data attribute we     |#
#| want but it's usually some property   |#
#| of the molecule not along the frame   |#
#| atom axis.                            |#
#=========================================#
# fnm        = The file name that the class was built from
# qcrems     = The Q-Chem 'rem' variables stored as a list of OrderedDicts
# qctemplate = The Q-Chem template file, not including the coordinates or rem variables
# charge     = The net charge of the molecule
# mult       = The spin multiplicity of the molecule
MetaVariableNames = set(['fnm', 'ftype', 'qcrems', 'qctemplate', 'charge', 'mult'])
# Variable names relevant to quantum calculations explicitly
QuantumVariableNames = set(['qcrems', 'qctemplate', 'charge', 'mult', 'qcsuf', 'qm_ghost'])
# Superset of all variable names.
AllVariableNames = QuantumVariableNames | AtomVariableNames | MetaVariableNames | FrameVariableNames

# OrderedDict requires Python 2.7 or higher
import os, sys, re, copy
import numpy as np
import imp
import itertools
from chemistry import PeriodicTable as PT
from collections import OrderedDict
from ctypes import *
from warnings import warn

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

def isint(word):
    """ONLY matches integers! If you have a decimal point? None shall pass!"""
    return re.match('^[-+]?[0-9]+$',word)

def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, integer, or what have you"""
    return re.match('^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$',word)

# Used to get the white spaces in a split line.
splitter = re.compile(r'(\s+|\S+)')

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

def format_gro_box(xyz):
    """ Print a line corresponding to the box vector in accordance with .gro file format

    @param[in] xyz A 3-element or 9-element array containing the box vectors
    
    """
    return ' '.join(["% 10.6f" % (i/10) for i in xyz])

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
    return key in A.Data and key in B.Data and A.Data[key] != B.Data[key]

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
        # I += PT[elem[i]]*(np.dot(xi,xi)*np.eye(3) - np.outer(xi,xi))
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
    # if linalg.det(rot_matrix) > 0:
    # wt[2] = -wt[2]
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
                    raise Exception('The keys %s and %s have different lengths - this isn\'t supposed to happen for two AtomKeys member variables.' % (key, klast))
                L = len(self.Data[key])
                klast = key
            if Defined:
                return L
            elif 'xyzs' in self.Data:
                return len(self.xyzs[0])
            else:
                raise Exception('na is ill-defined if the molecule has no AtomKeys member variables.')
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

    def __init__(self, fnm = None, ftype = None):
        """ To instantiate the class we simply define the table of
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
                self.comms = [(("Generated by ForceBalance from %s: " % fnm) if not i.startswith('Generated by ForceBalance') else "") + i.expandtabs() for i in self.comms]
            ## Make sure the comment line isn't too long
            # for i in range(len(self.comms)):
            #     self.comms[i] = self.comms[i][:100] if len(self.comms[i]) > 100 else self.comms[i]

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
            self.Data[key] = OtherMol.Data[key]

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
        """ Return a copy of the object with certain atoms selected. """
        if isinstance(atomslice, int) or isinstance(atomslice, slice) or isinstance(atomslice,np.ndarray):
            if isinstance(atomslice, int):
                atomslice = [atomslice]
        if isinstance(atomslice, list):
            atomslice = np.array(atomslice)
        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])
        for key in self.AtomKeys:
            New.Data[key] = list(np.array(self.Data[key])[atomslice])
        if 'xyzs' in self.Data:
            for i in range(self.ns):
                New.xyzs[i] = self.xyzs[i][atomslice]
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
        return [(Vec3(box[0]/10,0.0,0.0), Vec3(0.0,box[1]/10,0.0), Vec3(0.0,0.0,box[2]/10)) * nanometer for box in self.boxes]

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
                    boxes.append([float(i) for i in line.split()])
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

        bonds    = [[] for i in range(len(elem))]
        for bond in data.compounds.items()[0][1].bonds:
            a1 = bond.origin_atom_id - 1
            a2 = bond.target_atom_id - 1
            aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
            bonds[aL].append(aH)

        Answer = {'xyzs' : [np.array(xyz)],
                  'partial_charge' : charge,
                  'atomname' : atomname,
                  'atomtype' : atomtype,
                  'elem'     : elem,
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
            boxes.append([ts.A, ts.B, ts.C])
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
                if sline[0].capitalize() in PT and isfloat(sline[1]) and isfloat(sline[2]) and isfloat(sline[3]):
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
        @return suffix  The suffix that comes after lines in the XYZ coordinates; this is usually topology info

        """
        suffix   = []
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
        for line in open(fnm):
            sline = line.split()
            # The first line always contains the number of atoms
            # The words after the first line are comments
            if title:
                na = int(sline[0])
                comms.append(' '.join(sline[1:]))
                title = False
            elif len(sline) >= 6:
                if isint(sline[0]) and isfloat(sline[2]) and isfloat(sline[3]) and isfloat(sline[4]): # A line of data better look like this
                    if nframes == 0:
                        elem.append(sline[1])
                        resid.append(thisresid)
                        whites      = re.split('[^ ]+',line)
                        if len(sline) > 5:
                            suffix.append(''.join([whites[j]+sline[j] for j in range(5,len(sline))]))
                        else:
                            suffix.append('')
                    thisatom = int(sline[0])
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
                        nframes += 1
                        title = True
                        xyzs.append(np.array(xyz))
                        xyz = []
            ln += 1
        Answer = {'xyzs'   : xyzs,
                  'resid'  : resid,
                  'elem'   : elem,
                  'comms'  : comms,
                  'suffix' : suffix}
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
                boxes.append([float(i)*10 for i in sline])
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
            if re.match('^\$',line):
                wrd = re.sub('\$','',line)
                if wrd == 'end':
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
                if section == 'molecule':
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
                elif reading_template:
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

        Box = np.array([10.0, 10.0, 10.0])
        #Separate into distinct lists for each model.
        PDBLines=[[]]
        for x in ParsedPDB[0]:
            if x.__class__ in [END, ENDMDL]:
                PDBLines.append([])
            if x.__class__==ATOM:
                PDBLines[-1].append(x)
            if x.__class__==CRYST1:
                Box = np.array([x.a, x.b, x.c])

        X=PDBLines[0]

        XYZ=np.array([[x.x,x.y,x.z] for x in X])/10.0#Convert to nanometers
        ChainID=np.array([x.chainID for x in X],'str')
        AtomNames=np.array([x.name for x in X],'str')
        ResidueNames=np.array([x.resName for x in X],'str')
        ResidueID=np.array([x.resSeq for x in X],'int')
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

        Answer={"xyzs":XYZList, "chain":ChainID, "atomname":AtomNames,
                "resid":ResidueID, "resname":ResidueNames, "elem":elem,
                "comms":['' for i in range(len(XYZList))], "boxes":[Box for i in range(len(XYZList))]}

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
        XMode    = 0
        ffd      = 0
        conv     = []
        convThis = 0
        readChargeMult = 0
        energy_scf = []
        float_match  = {'energy_scfThis'   : ("^[1-9][0-9]* +[-+]?([0-9]*\.)?[0-9]+ +[-+]?([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)[A-Za-z0 ]*$", 1),
                        'energy_opt'       : ("^Final energy is +[-+]?([0-9]*\.)?[0-9]+$", -1),
                        'charge'           : ("Sum of atomic charges", -1),
                        'mult'             : ("Sum of spin +charges", -1),
                        'energy_mp2'       : ("^(ri-)*mp2 total energy += +[-+]?([0-9]*\.)?[0-9]+ +au$",-2),
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
                    ffd  = 1
            elif re.match("Standard Nuclear Orientation".lower(), line.lower()):
                XMode = 1
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
                        Mats[key]["This"] = add_strip_to_mat(Mats[key]["This"],Mats[key]["Strip"])
                        Mats[key]["Strip"] = []
                    elif re.match("^[0-9]+( +[-+]?([0-9]*\.)?[0-9]+)+$",line):
                        Mats[key]["Strip"].append([float(i) for i in line.split()[1:]])
                    else:
                        Mats[key]["This"] = add_strip_to_mat(Mats[key]["This"],Mats[key]["Strip"])
                        Mats[key]["Strip"] = []
                        Mats[key]["All"].append(np.array(Mats[key]["This"]))
                        Mats[key]["This"] = []
                        Mats[key]["Mode"] = 0
                elif re.match(val.lower(), line.lower()):
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
        elif len(Mats['gradient_mp2']['All']) > 0:
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
                out.append(''.join(["%8.3f" % i for i in self.boxes[I][:3]]))
        return out

    def write_arc(self, select):
        self.require('elem','xyzs')
        out = []
        if 'suffix' not in self.Data:
            sys.stderr.write("Beware, this .arc file contains no atom type or topology info\n")
        for I in select:
            xyz = self.xyzs[I]
            out.append("%6i  %s" % (self.na, self.comms[I]))
            for i in range(self.na):
                out.append("%6i  %s%s" % (i+1,format_xyz_coord(self.elem[i],xyz[i],tinker=True),self.suffix[i] if 'suffix' in self.Data else ''))
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
            out.append(format_gro_box(xyz = self.boxes[I]))
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
            ts.A      = self.boxes[I][0] if 'boxes' in self.Data else 1.0
            ts.B      = self.boxes[I][1] if 'boxes' in self.Data else 1.0
            ts.C      = self.boxes[I][2] if 'boxes' in self.Data else 1.0
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
        """
        self.require('xyzs')
        self.require_resname()
        self.require_resid()
        ATOMS = self.atomname if 'atomname' in self.Data else ["%s%i" % (self.elem[i], i+1) for i in range(self.na)]
        CHAIN = self.chain if 'chain' in self.Data else [1 for i in range(self.na)]
        RESNAMES = self.resname
        RESNUMS = self.resid

        out = []

        for I in select:
            XYZ = self.xyzs[I]
            for i in range(self.na):
                ATOMNUM = i + 1
                line=np.chararray(80)
                line[:]=' '
                line[0:4]=np.array(list("ATOM"))
                line=np.array(line,'str')
                line[6:11]=np.array(list(str(ATOMNUM%100000).rjust(5)))
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
                line[22:26]=np.array(list(str(RESNUMS[i]).rjust(4)))

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
        if 'boxes' not in self.Data or len(self.boxes) != self.ns:
            sys.stderr.write("We're writing a file with boxes, but the file either contains no boxes or too few\n")
            boxstr = raw_input("Enter 1 / 3 / 9 numbers for box in Angstrom or name of timeseries file -> ")
            if os.path.exists(boxstr):
                self.boxes = [[float(i.strip()),float(i.strip()),float(i.strip())] for i in open(boxstr).readlines()]
            else:
                box    = [float(i) for i in boxstr.split()]
                if len(box) == 3 or len(box) == 9:
                    self.boxes = [box for i in range(self.ns)]
                elif len(box) == 1:
                    self.boxes = [[box[0],box[0],box[0]] for i in range(self.ns)]
                else:
                    raise Exception("Not sure what to do since you gave me %i numbers" % len(box))

def main():
    print "Basic usage as an executable: molecule.py input.format1 output.format2"
    print "where format stands for xyz, pdb, gro, etc."
    Mao = Molecule(sys.argv[1])
    Mao.write(sys.argv[2])

if __name__ == "__main__":
    main()
