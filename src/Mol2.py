"""
Author: P. Tuffery 2008
Ressource Parisienne en Bioinformatique Structurale
http://bioserv.rpbs.univ-paris-diderot.fr

This is free software. You can use it, modify it, distribute it.
However, thanks for the feedback for any improvement you  bring to it!
Changed by Lee-Ping from dict to OrderedDict.

mol2_set:
  A class to manage multipe mol2 files
mol2: 
  A class to manage simple mol2 data
"""

import sys
import types
from collections import OrderedDict

#=====================================================================
#
# The mol2 atom line class
#
#=====================================================================

class mol2_atom:
    """
    This is to manage mol2 atomic lines on the form:
      1 C1          5.4790   42.2880   49.5910 C.ar    1  <1>         0.0424
    """
    def __init__(self, data=None):
        """
        if data is passed, it will be installed
        """
        self.atom_id    = None
        self.atom_name  = None
        self.x          = None
        self.y          = None
        self.z          = None
        self.atom_type  = None
        self.subst_id   = None
        self.subst_name = None
        self.charge     = None
        self.status_bit = None
        
        if data is not None:
            self.parse(data)

    def parse(self, data):
        """
        split the text line into a series of properties
        """
        it = data.split()
        self.set_atom_id(it[0])
        self.set_atom_name(it[1])
        self.set_crds(it[2],it[3],it[4])
        self.set_atom_type(it[5])
        self.set_subst_id(it[6])
        self.set_subst_name(it[7])
        try:
            self.set_charge(it[8])
        except:
            self.set_charge(0.)
        try:
            self.set_status_bit(it[9])
        except:
            self.status_bit = None
        # self.__repr__()
    
    def __repr__(self):
        """
        assemble the properties as a text line, and return it
        """
        # print "mol2_atom.__repr__()"
        # print self.atom_id, self.atom_name, self.x, self.y, self.z, self.atom_type, self.subst_id, self.subst_name, self.charge
        rs = "%7d %-5s    %9.4f %9.4f %9.4f %-7s %2d %4s  %12.6f" % (self.atom_id, self.atom_name, self.x, self.y, self.z, self.atom_type, self.subst_id, self.subst_name, self.charge)
        if self.status_bit is not None:
            rs = rs + " %s" % self.status_bit
        return rs

    def set_atom_id(self, atom_id=None):
        """
        atom identifier (integer, starting from 1)
        """
        if atom_id is not None:
            self.atom_id = int(atom_id)
        return self.atom_id

    def set_atom_name(self, atom_name=None):
        """
        The name of the atom (string)
        """
        if atom_name is not None:
            self.atom_name = atom_name
        return self.atom_name
        
    def set_crds(self, x = None, y = None, z = None):
        """
        the coordinates of the atom
        """
        if (x is not None) and (y is not None) and (z is not None):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        return self.x, self.y, self.z

    def set_atom_type(self, atom_type=None):
        """
        The mol2 type of the atom
        """
        if atom_type is not None:
            self.atom_type = atom_type
        return self.atom_type
        
    def set_subst_id(self, subst_id=None):
        """
        substructure identifier
        """
        if subst_id is not None:
            self.subst_id = int(subst_id)
        return self.subst_id
        
    def set_subst_name(self, subst_name=None):
        """
        substructure name
        """
        if subst_name is not None:
            self.subst_name = subst_name
        return self.subst_name
        
    def set_charge(self, charge=None):
        """
        atomic charge
        """
        if charge is not None:
            self.charge = float(charge)
        return self.charge
        
    def set_status_bit(self, status_bit=None):
        """
        Never to use (in theory)
        """
        if status_bit is not None:
            self.status_bit = status_bit
        return self.status_bit
        

#=====================================================================
#
# The mol2 bond line class
#
#=====================================================================

class mol2_bond:
    """
    This is to manage mol2 bond lines on the form:
     1     1     2   ar
    """
    def __init__(self, data=None):
        """
        if data is passed, it will be installed
        """
        self.bond_id         = None
        self.origin_atom_id  = None
        self.target_atom_id  = None
        self.bond_type       = None
        
        if data is not None:
            self.parse(data)

    def __repr__(self):
        # print "mol2_bond.__repr__()", self.bond_id, self.origin_atom_id, self.target_atom_id, self.bond_type
        rs = "%6d %5d %5d %4s" % (self.bond_id, self.origin_atom_id, self.target_atom_id, self.bond_type)
        if self.status_bit is not None:
            rs = rs + " %s" % self.status_bit
        return rs        

    def parse(self, data):
        """
        split the text line into a series of properties
        """
        it = data.split()
        self.bond_id        = int(it[0])
        self.origin_atom_id = int(it[1])
        self.target_atom_id = int(it[2])
        self.bond_type      = it[3]
        try:
            self.set_status_bit(it[4])
        except:
            self.status_bit = None
        # print "mol2_bond.__repr__():", self.__repr__()

 
    def set_bond_id(self, bond_id=None):
        """
        bond identifier (integer, starting from 1)
        """
        if bond_id is not None:
            self.bond_id = bond_id
        return self.bond_id

    def set_origin_atom_id(self, origin_atom_id=None):
        """
        the origin atom identifier (integer)
        """
        if origin_atom_id is not None:
            self.origin_atom_id = origin_atom_id
        return self.origin_atom_id

    def set_target_atom_id(self, target_atom_id=None):
        """
        the target atom identifier (integer)
        """
        if target_atom_id is not None:
            self.target_atom_id = target_atom_id
        return self.target_atom_id

    def set_bond_type(self, bond_type=None):
        """
        bond type (string) 
        one of: 
        1 = single
        2 = double
        3 = triple
        am = amide
        ar = aromatic
        du = dummy
        un = unknown
        nc = not connected
        """
        if bond_type is not None:
            self.bond_type = bond_type
        return self.bond_type

    def set_status_bit(self, status_bit=None):
        """
        Never to use (in theory)
        """
        if status_bit is not None:
            self.status_bit = status_bit
        return self.status_bit

#=====================================================================
#
# The one mol2 class
#
#=====================================================================

class mol2:
    """
    This is to manage one mol2 series of lines on the form: @verbatim
@<TRIPOS>MOLECULE
CDK2.xray.inh1.1E9H
 34 37 0 0 0
SMALL
GASTEIGER
Energy = 0

@<TRIPOS>ATOM
      1 C1          5.4790   42.2880   49.5910 C.ar    1  <1>         0.0424
      2 C2          4.4740   42.6430   50.5070 C.ar    1  <1>         0.0447
@<TRIPOS>BOND
     1     1     2   ar
     2     1     6   ar

    @endverbatim"""
    def __init__(self, data):
        self.mol_name  = None
        self.num_atoms = 0
        self.num_bonds = 0
        self.num_subst = 0
        self.num_feat  = 0
        self.num_sets  = 0
        self.mol_type  = None
        self.charge_type = None
        self.comments  = ""
        
        self.atoms = []
        self.bonds = []

        self.parse(data)

    def __repr__(self):
        # print "mol2.__repr__()", self.mol_name, self.num_atoms, self.num_bonds, self.num_subst, self.num_feat, self.num_sets
        rs = ""
        rs = rs + "%s\n" % "@<TRIPOS>MOLECULE"
        rs = rs + "%s\n" % self.mol_name
        rs = rs + "%d %d %d %d %d\n" % (self.num_atoms, self.num_bonds, self.num_subst, self.num_feat, self.num_sets)
        rs = rs + "%s\n" % self.mol_type
        rs = rs + "%s\n" % self.charge_type
        rs = rs + "%s" % self.comments
        rs = rs + "%s\n" % "@<TRIPOS>ATOM"
        
        for atom in self.atoms:
            rs = rs + "%s\n" % atom.__repr__()
        rs = rs + "%s\n" % "@<TRIPOS>BOND"
        for bond in self.bonds:
            rs = rs + "%s\n" % bond.__repr__()
        rs = rs + "\n"
        
        return rs

    def out(self, f = sys.stdout):
        f.write(self.__repr__())

    def set_mol_name(self, mol_name=None):
        """
        bond identifier (integer, starting from 1)
        """
        if mol_name is not None:
            self.mol_name = mol_name
        return self.mol_name

    def set_num_atoms(self, num_atoms=None):
        """
        number of atoms (integer)
        """
        if num_atoms is not None:
            self.num_atoms = int(num_atoms)
        return self.num_atoms

    def set_num_bonds(self, num_bonds=None):
        """
        number of bonds (integer)
        """
        if num_bonds is not None:
            self.num_bonds = int(num_bonds)
        return self.num_bonds

    def set_num_subst(self, num_subst=None):
        """
        number of substructures (integer)
        """
        if num_subst is not None:
            self.num_subst = int(num_subst)
        return self.num_subst

    def set_num_feat(self, num_feat=None):
        """
        number of features (integer)
        """
        if num_feat is not None:
            self.num_feat = int(num_feat)
        return self.num_feat

    def set_num_sets(self, num_sets=None):
        """
        number of sets (integer)
        """
        if num_sets is not None:
            self.num_sets = int(num_sets)
        return self.num_sets

    def set_mol_type(self, mol_type=None):
        """
        bond identifier (integer, starting from 1)
        """
        if mol_type is not None:
            self.mol_type = mol_type
        return self.mol_type

    def set_charge_type(self, charge_type=None):
        """
        bond identifier (integer, starting from 1)
        """
        if charge_type is not None:
            self.charge_type = charge_type
        return self.charge_type

    def parse(self, data):
        """
        Parse a series of text lines, 
        and setup compound information
        """
        for l in data:
            if l.count("@<TRIPOS>MOLECULE"):
                status = 1
                continue
            if status == 1:
                self.set_mol_name(l.split()[0])
                status = 2
                continue
            if status == 2:
                it = l.split()
                self.set_num_atoms(it[0])
                self.set_num_bonds(it[1])
                self.set_num_subst(it[2])
                self.set_num_feat(it[3])
                self.set_num_sets(it[4])
                status = 3
                continue
            if status == 3:
                self.set_mol_type(l.split()[0])
                status = 4
                continue
            if status == 4:
                self.set_charge_type(l.split()[0])
                status = 5
                continue
            if status == 5:
                if l.count("@<TRIPOS>ATOM"):
                    status = 6
                    if self.comments == "":
                        self.comments = "\n"
                    continue
                self.comments = self.comments + l
                continue
            if status == 6:
                if l.count("@<TRIPOS>BOND"):
                    status = 7
                    continue
                self.atoms.append(mol2_atom(l))
#                 if len(self.atoms) == self.num_atoms:
#                     status = 7
#                     continue
            if status == 7:
                if l.count("@<TRIPOS>"):
                    status = 8
                    continue
                self.bonds.append(mol2_bond(l))
                if len(self.bonds) == self.num_bonds:
                    status = 8
                    continue
        # print self.__repr__()

    def get_atom(self, id):
        """
        return the atom instance given its atom identifier
        """
        if self.atoms[id-1].set_atom_id() == id:
            return self.atoms[id-1]
        else:
            for i in range(0,len(self.atoms)):
                if self.atoms[i].set_atom_id() == id:
                    return self.atoms[i]
            else:
                return None
            
    def get_bonded_atoms(self, id):
        """
        return a dictionnary of atom instances bonded to the atom, and their types
        """
        rs = []
        for i in range(0,len(self.bonds)):
            if self.bonds[i].set_origin_atom_id() == id:
                # print id, "connected to",self.bonds[i].set_target_atom_id(),"(target)"
                rs.append(self.get_atom(self.bonds[i].set_target_atom_id()))
            if self.bonds[i].set_target_atom_id() == id:
                # print id, "connected to",self.bonds[i].set_origin_atom_id(),"(origin)"
                rs.append(self.get_atom(self.bonds[i].set_origin_atom_id()))
        return rs
            
    def set_donnor_acceptor_atoms(self, verbose = 0):
        """
        modify atom types to specify donnor, acceptor, or both
        """
        # print "set_donnor_acceptor_atoms", len(self.atoms)
        for i in range(0,len(self.atoms)):
        # for i in range(8,12):
            # print self.atoms[i]
            bonds = self.get_bonded_atoms(self.atoms[i].set_atom_id())
            atmType = self.atoms[i].set_atom_type()
            # print atmType
            # print bonds

            # sulfur
            if atmType in ["S.2", "S.3"]:
                isDonnor = 0
                for j in bonds:
                    if j.set_atom_type()[0] == "H":
                        isDonnor = 1
                        break
                if isDonnor:
                    self.atoms[i].set_atom_type("S.don")
                else:
                    self.atoms[i].set_atom_type("S.acc")
                continue
            
            # oxygen
            if atmType in ["O.3","O.2","O.co2"]:
                isAcceptor = 1
                isDonnor = 0
                if atmType in ["O.3"]:
                    for j in bonds:
                        if j.set_atom_type()[0] == "H":
                            isDonnor = 1
                            break
                if isDonnor:
                    self.atoms[i].set_atom_type("O.da") # donnor acceptor
                else:
                    self.atoms[i].set_atom_type("O.acc")
                continue
            
            # nitrogen
            if atmType in ["N.1","N.2","N.ar","N.3","N.4"]:
                hasH = 0
                for j in bonds:
                    if j.set_atom_type()[0] == "H":
                        hasH = 1
                        break
                isDonnor = 0
                if (atmType in ["N.2","N.ar","N.3","N.4"]) and hasH:
                    isDonnor = 1
                isAcceptor = 0
                if (atmType in ["N.1","N.2","N.ar"]) and (not hasH):
                    isAcceptor = 1
                if isDonnor and isAcceptor:
                    self.atoms[i].set_atom_type("N.da") # donnor acceptor
                elif isDonnor:
                    self.atoms[i].set_atom_type("N.don")
                elif isAcceptor:
                    self.atoms[i].set_atom_type("N.acc")
                continue

            # hydrophobic
            if atmType in ["C.2","C.3","N.pl3"]:
                hasH = 0
                for j in bonds:
                    if j.set_atom_type()[0] == "H":
                        hasH = 1
                        break
                hasONS = 0
                for j in bonds:
                    if j.set_atom_type()[0] in "ONS":
                        hasONS = 1
                        break
                if (atmType in ["N.pl3"]) and (hasH == 0):
                    self.atoms[i].set_atom_type("N.phb")
                    continue
                if (atmType in ["C.2","C.3"]) and hasH and (not hasONS):
                    self.atoms[i].set_atom_type("C.phb")
                    continue
                
class mol2_set:
    def __init__(self, data = None, subset = None):
        """
        A collection is organized as a dictionnary of compounds
        self.num_compounds : the number of compounds
        self.compounds     : the dictionnary of compounds
        data  : the data to setup the set
        subset: it is possible to specify a subset of the compounds to load, based on their mol_name identifiers.
        """
        self.num_compounds = 0
        self.comments = ""
        self.compounds = OrderedDict()
        
        # subset management
        if subset is not None:
            if isinstance(subset,types.ListType):
                pass
            elif isinstance(subset,types.StringType):
                try:
                    f = open(subset)
                    lines = f.readlines()
                    f.close()
                    for i in range(0, len(lines)):
                        lines[i] = lines[i].replace("\n","")
                    subset = lines
                except:
                    subset = None

        # data management
        if data is not None:
            if isinstance(data,mol2_set):
                self.num_compounds = data.num_compounds
                self.compounds     = data.compounds
                self.comments      = data.comments
            elif isinstance(data,types.StringType):
                try:
                    f = open(data)
                    lines = f.readlines()
                    f.close()
                    # print "Parsing %d lines" % len(lines)
                    self.parse(lines, subset)
                except:
                    pass
            elif isinstance(data,types.ListType):
                self.parse(data, subset)
        # return self

    def parse(self, data, subset = None):
        """
        parse a list of lines, detect compounds, load them
        only load the subset if specified.
        """
        status = 0
        cmpnds = OrderedDict()
        for l in range(0,len(data)):
            if (not status) and (data[l][0] == "#"):
                self.comments = self.comments + l
            if data[l].count("@<TRIPOS>MOLECULE"):
                status = 1
                if len(cmpnds):
                    if (subset is None) or (cmpnd in subset):
                        cmpnds[cmpnd]["to"] = l
                ffrom = l
                cmpnd = data[l+1].split()[0]
                if (subset is None) or (cmpnd in subset):
                    cmpnds[cmpnd] = OrderedDict([("from", l)])
            if (subset is None) or (cmpnd in subset):
                cmpnds[cmpnd]["to"] = len(data) 
        
        for cmpnd in cmpnds.keys():
            # print cmpnd, cmpnds[cmpnd]["from"],cmpnds[cmpnd]["to"]
            # print data[cmpnds[cmpnd]["from"]:cmpnds[cmpnd]["to"]]
            self.compounds[cmpnd] = mol2(data[cmpnds[cmpnd]["from"]:cmpnds[cmpnd]["to"]])
            self.num_compounds += 1
            # break

if __name__ == "__main__":

    import sys

    # data = mol2_set(sys.argv[1], subset=["CDK2.xray.inh1.1E9H", 'RNAse.xray.inh4.1O0F']) # , 'RNAse.xray.inh4.1O0F'
    data = mol2_set(sys.argv[1], subset=["RNAse.xray.inh8.1QHC"]) # No subset

    sys.stderr.write("Loaded %d compounds\n" % data.num_compounds)

    for cmpnd in data.compounds.keys():
        # print data.compounds[cmpnd],
        data.compounds[cmpnd].set_donnor_acceptor_atoms()
        print data.compounds[cmpnd],
        break
