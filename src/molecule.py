#!/usr/bin/env python

#========================================================#
#|                                                      |#
#|       Chemical file format conversion module         |#
#|                                                      |#
#|         Lee-Ping Wang (leeping@stanford.edu)         |#
#|              Last updated May 29, 2012               |#
#|                                                      |#
#|        [ IN PROGRESS, USE AT YOUR OWN RISK ]         |#
#|                                                      |#
#|    This is free software released under version 2    |#
#|    of the GNU GPL, please use or redistribute as     |#
#|    you see fit under the terms of this license.      |#
#|    (http://www.gnu.org/licenses/gpl-2.0.html)        |#
#|    Feedback and suggestions are encouraged.          |#
#|                                                      |#
#|    What this is for:                                 |#
#|    Converting a molecule between file formats        |#
#|    Loading and processing of trajectories            |#
#|    (list of geometries for the same set of atoms)    |#
#|    Concatenating or slicing trajectories             |#
#|    Combining molecule metadata (charge, bonds,       |#
#|    Q-Chem rem variables)                             |#
#|                                                      |#
#|    What this isn't for (yet):                        |#
#|    Adding or removing atoms in a molecule            |#
#|                                                      |#
#|    Supported file formats:                           |#
#|    See the __init__ method in the Molecule class.    |#
#|                                                      |#
#|    Note to self / developers:                        |#
#|    Please make this file as standalone as possible   |#
#|    (i.e. don't introduce too many dependencies)      |#
#|    Please make sure this file is up-to-date in       |#
#|    both the 'leeping' and 'forcebalance' modules     |#
#|                                                      |#
#|       Contents of this file:                         |#
#|       1) Imports                                     |#
#|       2) Subroutines                                 |#
#|       3) Molecule class                              |#
#|         a) Class customizations (add, getitem)       |#
#|         b) Instantiation                             |#
#|         c) Core functionality (read, write)          |#
#|         d) Reading functions                         |#
#|         e) Writing functions                         |#
#|         f) Extra stuff                               |#
#|       4) "main" function (if executed)               |#
#|                                                      |#
#|            Required: Python 2.7, Numpy 1.6           |#
#|            Optional: Mol2, PDB, DCD readers          |#
#|             (can be found in ForceBalance)           |#
#|                                                      |#
#|      Thanks: Todd Dolinsky, Yong Huang,              |#
#|              Kyle Beauchamp (PDB)                    |#
#|              John Stone (DCD Plugin)                 |#
#|              Pierre Tuffery (Mol2 Plugin)            |#
#|                                                      |#
#|      Instructions:                                   |#
#|                                                      |#
#|        To import:                                    |#
#|          from molecule import Molecule               |#
#|        To create a Molecule object:                  |#
#|          MyMol = Molecule(fnm)                       |#
#|        To convert to a new file format:              |#
#|          MyMol.write('newfnm.format')                |#
#|        To concatenate geometries:                    |#
#|          MyMol += MyMolB                             |#
#|                                                      |#
#========================================================#

import os, sys, re, copy
import numpy as np
import imp
import itertools
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

class MolfileTimestep(Structure):
    """ Wrapper for the timestep C structure used in molfile plugins. """
    _fields_ = [("coords",POINTER(c_float)), ("velocities",POINTER(c_float)),
                ("A",c_float), ("B",c_float), ("C",c_float), ("alpha",c_float), 
                ("beta",c_float), ("gamma",c_float), ("physical_time",c_double)]
    
class Molecule(dict):
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
        L = 0
        klast = None
        for k in self.keys():
            if k in self.PerFrameData:
                if L != 0 and len(self[k]) != L:
                    raise Exception('The keys %s and %s have different lengths, this isn\'t supposed to happen.' % (k, klast))
                L = len(self[k])
                klast = k
            else: continue
        return L

    def __getattr__(self, name):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the dictionary. """
        if name == 'ns':
            return len(self)
        elif name == 'na':
            if 'elem' in self:
                return len(self.elem)
            else:
                return len(self.xyzs[0])
        elif name in self:
            return super(Molecule, self).__getitem__(name)
        else:
            raise AttributeError()

    def __setattr__(self, name, value):
        """ Whenever we set a class attribute, it's actually stored in the dictionary. """
        if name in ['Read_Tab', 'Write_Tab', 'Funnel', 'Immutable', 'PerFrameData', 'QuantumData']:
            return object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __getitem__(self, key):
        """ 
        If we say MyMolecule[0:10], then we'll return a copy of MyMolecule with frames 0 through 9.
        If we say MyMolecule['xyzs'], then we'll return MyMolecule.xyzs
        
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key,np.ndarray):
            if isinstance(key, int):
                key = [key]
            New = Molecule()
            for k in self.keys():
                if k in self.PerFrameData:
                    setattr(New, k, list(np.array(self[k])[key]))
                else:
                    setattr(New, k, copy.deepcopy(self[k]))
            return New
        elif key == 'ns' or key == 'na':
            return self.__getattr__(key)
        return super(Molecule, self).__getitem__(key)

    def __add__(self,other):
        """ Add method for Molecule objects. """
        # Check type of other
        if not isinstance(other,Molecule):
            raise TypeError('A Molecule instance can only be added to another Molecule instance')
        # Create the sum of the two classes by copying the first class.
        Sum = copy.deepcopy(self)
        self.check_immutable(other)
        # Information from the other class is added to this class (if said info doesn't exist.)
        for key in list(k for k in other if k not in self):
            Sum[key] = other[key]
        # If the two objects have 'PerFrameData' attributes, then we simply string the attributes together.
        # An example is the list of XYZ coordinates.
        for key in self.PerFrameData:
            if key in self and key in other:
                Sum[key] = self[key] + other[key]
        return Sum
 
    def __iadd__(self,other):
        return self + other

    def append(self,other):
        self += other

    def __init__(self, fnm = None, ftype = None):
        """ To instantiate the class we simply define the table of
        file reading/writing functions and read in a file if it is
        provided."""
        #=====================================#
        #|         File type tables          |#
        #|  Feel free to edit these as more  |#
        #|    readers / writers are added    |#
        #=====================================#
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
                          'out'     : 'qcout',
                          'esp'     : 'qcesp',
                          'txt'     : 'qdata',
                          'crd'     : 'charmm',
                          'cor'     : 'charmm',
                          'arc'     : 'tinker'}
        ## Creates entries like 'gromacs' : 'gromacs' and 'xyz' : 'xyz'
        for i in set(self.Read_Tab.keys() + self.Write_Tab.keys()):
            self.Funnel[i] = i

        self.PerFrameData = ['xyzs', 'comms', 'boxes', 'qm_forces', 'qm_energies', 'qm_espxyzs', 'qm_espvals']
        self.QuantumData = ['qcrems', 'qc_template', 'charge', 'mult']
        self.Immutable = ['elem', 'na']

        ## Read in stuff, if we passed in a file name, otherwise return an empty instance.
        if fnm != None:
            Parsed = self.read(fnm, ftype)
            ## Set attributes.
            for key, val in Parsed.items():
                self[key] = val
            ## Create a list of comment lines if we don't already have them from reading the file.
            if hasattr(self, 'comms'):
                for i in range(len(self.comms)):
                    self.comms[i] += ' (Converted using molecule.py from %s)' % fnm
            else:
                self.comms = ['Frame %i of %i : Converted using molecule.py from %s' % (i+1, self.ns, fnm) for i in range(self.ns)]

    #=====================================#
    #|     Core read/write functions     |#
    #| Hopefully we won't have to change |#
    #|         these very often!         |#
    #=====================================#

    def check_immutable(self, other):
        # Sanity checks.  Crash if the two objects have any 'Immutable' attributes that are different.
        # An example is the list of chemical elements; we can't add a benzene to a formaldehyde
        for key in self.Immutable:
            if key in self and key in other and self[key] != other[key]:
                print self[key]
                print other[key]
                raise Exception("When adding two Molecule objects, the values for key %s must be identical." % key)

    def require(self, *args):
        for arg in args:
            if arg not in self:
                raise AttributeError("%s is a required attribute for writing this type of file but it's not present" % arg)

    def read(self, fnm, ftype = None):
        """ Read in a file. """
        if ftype == None:
            ## Try to determine from the file name using the extension.
            ftype = os.path.splitext(fnm)[1][1:]
        ## This calls the table of reader functions and prints out an error message if it fails.
        ## 'Answer' is a dictionary of data that is returned from the reader function.
        Answer = self.Read_Tab[self.Funnel[ftype.lower()]](fnm)
        return Answer

    def write(self,fnm=None,ftype=None,append=False,select=None):
        if fnm == None and ftype == None:
            raise Exception("Output file name and file type are not specified.")
        elif ftype == None:
            ftype = os.path.splitext(fnm)[1][1:]
        ## I needed to add in this line because the DCD writer requires the file name,
        ## but the other methods don't.
        self.fout = fnm
        if type(select) is int:
            select = [select]
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
    #|  For doing useful things like     |#
    #|    readers / writers are added    |#
    #=====================================#

    def read_frames(self, fnm):
        Parsed = self.read(fnm)
        for key in self.PerFrameData:
            self[key] = OtherMol[key]

    def replace_frames(self, other):
        if type(other) is Molecule:
            self.check_immutable(other)
            for key in self.PerFrameData:
                self[key] = other[key]
        elif type(other) is str:
            OtherMol = self.__class__(other)
            self.check_immutable(OtherMol)
            for key in self.PerFrameData:
                self[key] = OtherMol[key]
    
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
            for key in other.QuantumData:
                self[key] = other[key]
        elif type(other) is str:
            OtherMol = self.__class__(other)
            for key in OtherMol.QuantumData:
                self[key] = OtherMol[key]

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
            elif 'ENERGY' in line:
                energies.append(float(line.split()[1]))
        return {'xyzs' : xyzs, 'qm_energies' : energies, 'qm_forces' : forces, 'qm_espxyzs' : espxyzs, 'qm_espvals' : espvals}

    def read_mol2(self, fnm):
        xyz      = []
        charge   = []
        atomname = []
        atomtype = []
        elem     = []
        data = Mol2.mol2_set(fnm)
        if len(data.compounds) > 1:
            warn("Not sure what to do if the MOL2 file contains multiple compounds")
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
                  'atomic_charges' : charge,
                  'atomname' : atomname,
                  'atomtype' : atomtype,
                  'elem'     : elem,
                  'bonds'    : bonds
                  }

        return Answer
        # print data.compounds
        # print data
        # print dir(data)

    def read_dcd(self, fnm):
        xyzs = []
        boxes = []
        
        if _dcdlib.vmdplugin_init() != 0:
            raise IOError("Unable to init DCD plugin")
        natoms = c_int(-1)
        frame  = 0
        ## Open the DCD file
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
        ## Close the DCD file
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
        @return rawarcs The entire .arc file in list form (because some extra info is required for re-printing the .arc)
        @return suffix  The suffix that comes after lines in the XYZ coordinates; this is usually topology info

        """
        rawarcs  = []
        rawarc   = []
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
            rawarc.append(line.strip())
            sline = line.split()
            # The first line always contains the number of atoms
            # The words after the first line are comments
            if title:
                na = int(sline[0])
                comms.append(' '.join(sline[1:]))
                title = False
            elif len(sline) >= 6:
                #print sline
                if isint(sline[0]) and isfloat(sline[2]) and isfloat(sline[3]) and isfloat(sline[4]): # A line of data better look like this
                    if nframes == 0:
                        elem.append(sline[1])
                        resid.append(thisresid)
                        whites      = split('[^ ]+',line)
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
                        rawarcs.append(list(rawarc))
                        rawarc = []
            ln += 1
        Answer = {'xyzs'   : xyzs,
                  'resid'  : resid,
                  'elem'   : elem,
                  'comms'  : comms,
                  'rawarcs': rawarcs,
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

        These files can be complicated, and I can't write a completely
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
                    elif section == 'rem':
                        if reading_template:
                            qcrems.append(qcrem)
                            qcrem = OrderedDict()
                    if reading_template:
                        template.append((section,SectionData))
                    SectionData = []
                else:
                    section = wrd
                    inside_section = True
            elif inside_section:
                if section == 'molecule':
                    if len(dline) == 4 and all([isfloat(dline[i]) for i in range(1,4)]):
                        if fff:
                            reading_template = False
                            template_cut = list(i for i, dat in enumerate(template) if dat[0] == '@@@@')[-1]
                        else:
                            elem.append(sline[0])
                        xyz.append([float(i) for i in sline[1:4]])
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

        Answer = {'qc_template' : template,
                  'qcrems'      : qcrems,
                  'charge'      : charge,
                  'mult'        : mult,
                  }

        if len(xyzs) > 0:
            Answer['xyzs'] = xyzs
        if len(elem) > 0:
            Answer['elem'] = elem
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
        Answer['qc_template'] = Aux['qc_template']
        Answer['qcrems'] = Aux['qcrems']
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
        else:
            raise Exception('There are no forces in %s' % fnm)
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
        lens = [len(i) for i in Answer['qm_energies'], Answer['xyzs'], Answer['qm_forces']]
        if len(set(lens)) != 1:
            raise Exception('The number of energies, forces, and coordinates in %s are not the same : %s' % (fnm, str(lens)))
        # The number of atoms should all be the same
        if len(set([len(i) for i in Answer['xyzs']])) != 1:
            raise Exception('The numbers of atoms across frames in %s are not all the same' % (fnm))

        for i, frc in enumerate(Answer['qm_forces']):
            Answer['qm_forces'][i] = frc.T

        return Answer
    
    #=====================================#
    #|         Writing functions         |#
    #=====================================#

    def write_qcin(self, select):
        self.require('qc_template','qcrems','charge','mult','xyzs','elem')
        out = []
        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            remidx = 0
            molecule_printed = False
            for SectName, SectData in self.qc_template:
                if SectName != '@@@@':
                    out.append('$%s' % SectName)
                    for line in SectData:
                        out.append(line)
                    if SectName == 'molecule':
                        if molecule_printed == False:
                            molecule_printed = True
                            out.append("%i %i" % (self.charge, self.mult))
                            for e, x in zip(self.elem, xyz):
                                out.append(format_xyz_coord(e, x))
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
        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            out.append("%-5i" % self.na)
            out.append(self.comms[I])
            for i in range(self.na):
                out.append(format_xyz_coord(self.elem[i],xyz[i]))
        return out

    def write_mdcrd(self, select):
        self.require('xyzs')
        # Groups a big long list into groups of ten.
        def grouper(n, iterable):
            args = [iter(iterable)] * n
            return list([e for e in t if e != None] for t in itertools.izip_longest(*args))
        # In mdcrd files, there is only one comment line
        out = ['mdcrd file generated using ForceBalance'] 
        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            out += [''.join(["%8.3f" % i for i in g]) for g in grouper(10, list(xyz.flatten()))]
            if 'boxes' in self:
                out.append(''.join(["%8.3f" % i for i in self.boxes[I][:3]]))
        return out

    def write_arc(self, select):
        self.require('elem','xyzs')
        out = []
        if 'suffix' not in self:
            sys.stderr.write("Beware, this .arc file contains no atom type or topology info\n")
        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            out.append("%6i  %s" % (self.na, self.comms[I]))
            for i in range(self.na):
                out.append("%6i  %s%s" % (i+1,format_xyz_coord(self.elem[i],xyz[i],tinker=True),self.suffix[i] if 'suffix' in self else ''))
        return out

    def write_gro(self, select):
        self.require('elem','xyzs')
        out = []
        self.require_resname()
        self.require_resid()
        self.require_boxes()

        if 'atomname' not in self:
            atomname = ["%s%i" % (self.elem[i], i+1) for i in range(self.na)]
        else:
            atomname = self.atomname

        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            xyzwrite = xyz.copy()
            xyzwrite /= 10.0 # GROMACS uses nanometers
            out.append("Generated by filecnv.py : " + self.comms[I])
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
        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            ts.coords = _xyz(*list(xyz.flatten()))
            ts.A      = self.boxes[I][0] if 'boxes' in self else 1.0
            ts.B      = self.boxes[I][1] if 'boxes' in self else 1.0
            ts.C      = self.boxes[I][2] if 'boxes' in self else 1.0
            result    = _dcdlib.write_timestep(dcd, byref(ts))
            if result != 0:
                raise IOError("Error encountered when writing DCD")
        ## Close the DCD file
        _dcdlib.close_file_write(dcd)
        dcd = None

    def write_pdb(self, select):
        """Save Conformation to a PDB. 
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
        ATOMS = self.atomname if 'atomname' in self else ["%s%i" % (self.elem[i], i+1) for i in range(self.na)]
        CHAIN = self.chain if 'chain' in self else [1 for i in range(self.na)]
        RESNAMES = self.resname
        RESNUMS = self.resid

        out = []

        for I in (select if select != None else range(len(self))):
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
        self.require('xyzs','qm_energies','qm_forces')
        out = []
        for I in (select if select != None else range(len(self))):
            xyz = self.xyzs[I]
            out.append("JOB %i" % I)
            out.append("COORDS"+pvec(xyz))
            out.append("ENERGY % .12e" % self.qm_energies[I])
            if 'mm_energies' in self:
                out.append("EMD0   % .12e" % self.mm_energies[I])
            out.append("FORCES"+pvec(self.qm_forces[I]))
            if 'qm_espxyzs' in self and 'qm_espvals' in self:
                out.append("ESPXYZ"+pvec(self.qm_espxyzs[I]))
                out.append("ESPVAL"+pvec(self.qm_espvals[I]))
            out.append('')
        return out

    def require_resid(self):
        if 'resid' not in self:
            na_res = int(raw_input("Enter how many atoms are in a residue -> "))
            self.resid = [1 + i/na_res for i in range(self.na)]
            
    def require_resname(self):
        if 'resname' not in self:
            resname = raw_input("Enter a residue name (3-letter like 'SOL') -> ")
            self.resname = [resname for i in range(self.na)]
            
    def require_boxes(self):
        if 'boxes' not in self or len(self.boxes) != self.ns:
            warn("We're writing a file with boxes, but the file either contains no boxes or too few")
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

    #=====================================#
    #|      Stuff I'm not sure about     |#
    #=====================================#

    def read_ndx(self, fnm):
        """ Read an index.ndx file and add an entry to the dictionary
        {'index_name': [ num1, num2, num3 .. ]}
        """

        section = None
        for line in open(fnm):
            line = line.strip().expandtabs()
            s = line.split()
            if re.match('^\[.*\]',line):
                section = re.sub('[\[\] \n]','',line)
            elif all([isint(i) for i in s]):
                for j in s:
                    self.index.setdefault(section,[]).append(int(j)-1)

    def reorder(self, idx_name=None):
        """ Reorders an xyz file using data provided from an .ndx file. """
        
        if idx_name not in self.index:
            raise Exception("ARGHH, %s doesn't exist in the index" % idx_name)

        self.require_resid()
        self.require_resname()

        newatomname = list(self.atomname)
        newelem = list(self.elem)
        newresid = list(self.resid)
        newresname = list(self.resname)
        for an, assign in enumerate(self.index[idx_name]):
            newatomname[an] = self.atomname[assign]
            newelem[an] = self.elem[assign]
            newresid[an] = self.resid[assign]
            newresname[an] = self.resname[assign]
        self.atomname = list(newatomname)
        self.elem = list(newelem)
        self.resid = list(newresid)
        self.resname = list(newresname)

        for sn, xyz in enumerate(self.xyzs):
            newxyz = xyz.copy()
            for an, assign in enumerate(self.index[idx_name]):
                newxyz[an] = xyz[assign]
            self.xyzs[sn] = newxyz.copy()

def main():
    print "Basic usage as an executable: molecule.py input.format1 output.format2"
    print "where format stands for xyz, pdb, gro, etc."
    Mao = Molecule(sys.argv[1])
    Mao.write(sys.argv[2])

if __name__ == "__main__":
    main()
