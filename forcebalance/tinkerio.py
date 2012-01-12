""" @package tinkerio TINKER input/output.

@author Lee-Ping Wang
@date 01/2012
"""

import os
from re import match, sub
from nifty import isint, isfloat
from numpy import array
from basereader import BaseReader
from subprocess import Popen, PIPE
from forceenergymatch import ForceEnergyMatch

pdict = {'VDW'          : {'Atom':[1], 2:'S',3:'T',4:'D'}, # Van der Waals distance, well depth, damping
         'BOND'         : {'Atom':[1,2], 3:'K',4:'B'},     # Bond force constant and equilibrium distance (Angstrom)
         'ANGLE'        : {'Atom':[1,2,3], 4:'K',5:'B'},   # Angle force constant and equilibrium angle
         'UREYBRAD'     : {'Atom':[1,2,3], 4:'K',5:'B'},   # Urey-Bradley force constant and equilibrium distance (Angstrom)
         'CHARGE'       : {'Atom':[1,2,3], 4:''},          # Atomic charge
         'DIPOLE'       : {0:'X',1:'Y',2:'Z'},             # Dipole moment in local frame
         'QUADX'        : {0:'X'},                         # Quadrupole moment, X component
         'QUADY'        : {0:'X',1:'Y'},                   # Quadrupole moment, Y component
         'QUADZ'        : {0:'X',1:'Y',2:'Z'},             # Quadrupole moment, Y component
         'POLARIZE'     : {'Atom':[1], 2:'A',3:'T'},       # Atomic dipole polarizability
         'BOND-CUBIC'   : {'Atom':[], 0:''},    # Below are global parameters.
         'BOND-QUARTIC' : {'Atom':[], 0:''},
         'ANGLE-CUBIC'  : {'Atom':[], 0:''},
         'ANGLE-QUARTIC': {'Atom':[], 0:''},
         'ANGLE-PENTIC' : {'Atom':[], 0:''},
         'ANGLE-SEXTIC' : {'Atom':[], 0:''},
         'DIELECTRIC'   : {'Atom':[], 0:''},
         'POLAR-SOR'    : {'Atom':[], 0:''}
                                                # Ignored for now: stretch/bend coupling, out-of-plane bending,
                                                # torsional parameters, pi-torsion, torsion-torsion
         }



class Tinker_Reader(BaseReader):
    """Finite state machine for parsing TINKER force field files.

    This class is instantiated when we begin to read in a file.  The
    feed(line) method updates the state of the machine, informing it
    of the current interaction type.  Using this information we can
    look up the interaction type and parameter type for building the
    parameter ID.
    
    """
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(Tinker_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict
        ## The atom numbers in the interaction (stored in the TINKER parser)
        self.atom   = []

    def feed(self, line):
        """ Given a line, determine the interaction type and the atoms involved (the suffix).

        TINKER generally has stuff like this:

        bond-cubic              -2.55
        bond-quartic            3.793125

        vdw           1               3.4050     0.1100
        vdw           2               2.6550     0.0135      0.910

        multipole     2    1    2               0.25983
                                               -0.03859    0.00000   -0.05818
                                               -0.03673
                                                0.00000   -0.10739
                                               -0.00203    0.00000    0.14412


        Unit of force is kcal / mole / angstrom squared
        
        """
        s          = line.split()
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^#',line): return None, None
        # From the line, figure out the interaction type.  If this line doesn't correspond
        # to an interaction, then do nothing.
        if s[0].upper() in pdict:
            self.itype = s[0].upper()
        # This is kind of a silly hack that allows us to take care of the 'multipole' keyword,
        # because of the syntax that the TINKER .prm file uses.
        elif s[0].upper() == 'MULTIPOLE':
            self.itype = 'CHARGE'
        elif self.itype == 'CHARGE':
            self.itype = 'DIPOLE'
        elif self.itype == 'DIPOLE':
            self.itype = 'QUADX'
        elif self.itype == 'QUADX':
            self.itype = 'QUADY'
        elif self.itype == 'QUADY':
            self.itype = 'QUADZ'
        else:
            self.itype = None

        if self.itype in pdict:
            if 'Atom' in pdict[self.itype]:
                self.atom = [s[i] for i in pdict[self.itype]['Atom']]
            self.suffix = '.'.join(self.atom)

class ForceEnergyMatch_TINKER(ForceEnergyMatch):
    """ Subclass of FittingSimulation for force and energy matching using TINKER.
    Implements the prepare and energy_force_driver methods."""

    def __init__(self,options,sim_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.trajfnm = "all.arc"
        ## Initialize the SuperClass!
        super(ForceEnergyMatch_TINKER,self).__init__(options,sim_opts,forcefield)

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        os.symlink(os.path.join(options['tinkerpath'],"testgrad"),os.path.join(abstempdir,"testgrad"))
        # Link the run parameter file
        os.symlink(os.path.join(self.root,self.simdir,"shot.key"),os.path.join(abstempdir,"shot.key"))

    def energy_force_driver(self):
        o, e = Popen(["./testgrad","shot.arc",self.FF.fnms[0],"y","n"],stdout=PIPE,stderr=PIPE).communicate()
        F = []
        for line in o.split('\n'):
            s = line.split()
            if "Total Potential Energy" in line:
                E = [float(s[4]) * 4.18]
            elif len(s) == 6 and all([s[0] == 'Anlyt',isint(s[1]),isfloat(s[2]),isfloat(s[3]),isfloat(s[4]),isfloat(s[5])]):
                F += [-1 * float(i) * 4.18 * 10 for i in s[2:5]]
        M = array(E + F)
        return M
