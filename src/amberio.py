""" @package amberio AMBER force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os
from re import match, sub, split, findall
from nifty import isint, isfloat, _exec
import numpy as np
from basereader import BaseReader
from abinitio import AbInitio

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
        elif line.strip().lower() == '@<tripos>molecule':
            self.itype = 'None'
            self.section = 'Molecule'
        elif self.section == 'Molecule' and self.mol == None:
            self.mol = '_'.join(s)
        elif not is_mol2_atom(line):
            self.itype = 'None'

        if is_mol2_atom(line) and self.itype == 'COUL':
            #self.atomnames.append(s[self.pdict[self.itype]['Atom'][0]])
            #self.adict.setdefault(self.mol,[]).append(s[self.pdict[self.itype]['Atom'][0]])
            self.atomnames.append(s[0])
            self.adict.setdefault(self.mol,[]).append(s[0])
            print self.adict
            print self.atomnames

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
                print self.molatom

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
        return split(' +(?!-(?![0-9.]))', line.replace('\n',''))

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

class AbInitio_AMBER(AbInitio):

    """Subclass of FittingSimulation for force and energy matching
    using AMBER.  Implements the prepare and energy_force_driver
    methods.  The get method is in the base class.  """

    def __init__(self,options,sim_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.trajfnm = "all.gro"
        super(AbInitio_AMBER,self).__init__(options,sim_opts,forcefield)
        ## all_at_once is not implemented.
        self.all_at_once = True

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        os.symlink(os.path.join(self.root,self.simdir,"settings","force.mdin"),os.path.join(abstempdir,"force.mdin"))
        os.symlink(os.path.join(self.root,self.simdir,"settings","stage.leap"),os.path.join(abstempdir,"stage.leap"))
        # I also need to write the trajectory
        if 'boxes' in self.traj.Data.keys():
            del self.traj.Data['boxes']
        self.traj.write(os.path.join(abstempdir,"all.mdcrd"))

    def energy_force_driver_all_external_(self):
        ## Create the run input files (inpcrd, prmtop) from the force field file.  
        ## Note that the frcmod and mol2 files are required.
        ## This is like 'grompp' in GROMACS.
        _exec("tleap -f stage.leap", print_to_screen=False, print_command=False)
        ## This line actually runs AMBER.
        _exec("sander -i force.mdin -o force.mdout -p prmtop -c inpcrd -y all.mdcrd -O", print_to_screen=False, print_command=False)
        ## Simple parser for 
        ParseMode = 0
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
                if len(Force) == 3*self.natoms:
                    Forces.append(np.array(Force))
                    Force = []
                    ParseMode = 0
            if line == '0 START of Energies':
                ParseMode = 1
            elif line == '1 Total Force':
                ParseMode = 2

        Energies = np.array(Energies[1:])
        Forces = np.array(Forces[1:])
        
        M = np.hstack((Energies.reshape(-1,1), Forces))

        return M

    def energy_force_driver_all(self):
        return self.energy_force_driver_all_external_()
