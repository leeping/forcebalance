""" @package psi4io PSI4 force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os
from re import match, sub, split, findall
from nifty import isint, isfloat, _exec
import numpy as np
from leastsq import LeastSquares
from basereader import BaseReader
from string import capitalize

##Interaction type -> Parameter Dictionary.
#pdict = {'Exponent':{0:'A', 1:'C'},
#         'BASSP' :{0:'A', 1:'B', 2:'C'}
#         }

class GBS_Reader(BaseReader):
    """Finite state machine for parsing basis set files.
    
    """
    
    def __init__(self,fnm):
        super(GBS_Reader,self).__init__(fnm)
        self.element = ''
        self.amom  = ''
        self.last_amom = ''
        self.basis_number  = 0
        self.contraction_number = -1
        self.adict={None:None}
    
    def build_pid(self, pfld):
        if pfld == 0:
            ptype = 'Exponent'
        elif pfld == 1:
            ptype = 'Contraction'
        return ptype+":"+"Elem=%s,AMom=%s,Bas=%i,Con=%i" % (self.element, self.amom, self.basis_number, self.contraction_number)
        
    def feed(self, line):
        """ Feed in a line.

        @param[in] line     The line of data

        """
        line       = line.split('!')[0].strip()
        s          = line.split()
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^!',line): return None, None
        # Now go through all the cases.
        if match('^[A-Za-z][a-z]? +[0-9]$',line):
            # This is supposed to match the element line. For example 'Li 0'
            self.element = capitalize(s[0])
        elif len(s) == 3 and match('[SPDFGH]+',s[0]) and isint(s[1]) and isfloat(s[2]):
            self.amom = s[0]
            if self.amom == self.last_amom:
                self.basis_number += 1
            else:
                self.basis_number = 0
                self.last_amom = self.amom
            self.contraction_number = -1
            # This is supposed to match a line like 'P   1   1.00'
        elif len(s) == 2 and isfloat(s[0]) and isfloat(s[1]):
            self.contraction_number += 1

class THCDF_Psi4(LeastSquares):

    def __init__(self,options,sim_opts,forcefield):
        super(THCDF_Psi4,self).__init__(options,sim_opts,forcefield)
        # Parse the input.dat file to figure out the elements and molecules
        MolSection = False
        ElemList = []
        self.Molecules = []
        for line in open(os.path.join(self.root,self.simdir,"input.dat")).readlines():
            line = line.strip()
            s = line.split()
            if len(s) >= 3 and s[0].lower() == 'molecule' and s[2] == '{':
                MolSection = True
                self.Molecules.append(s[1])
            elif len(s) >= 1 and s[0] == '}':
                MolSection = False
            elif MolSection and len(s) >= 4 and match("^[A-Za-z]+$",s[0]) and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
                ElemList.append(capitalize(s[0]))
        self.Elements = set(ElemList)
        for p in range(self.FF.np):
            Pelem = []
            for pid in self.FF.plist[p].split():
                # Extract the chemical element.
                Pelem.append(pid.split(':')[1].split(',')[0].split('=')[1])
            Pelem = set(Pelem)
            if len(self.Elements.intersection(Pelem)) == 0:
                self.call_derivatives[p] = False

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        os.symlink(os.path.join(self.root,self.simdir,"input.dat"),os.path.join(abstempdir,"input.dat"))

    def indicate(self):
        print "\rSim: %-15s" % self.name, 
        print "Molecules =", self.Molecules,
        print "Objective = %.5e" % self.objective
        return

    def driver(self):
        ## Delete objective.dat (because PSI4 appends it).
        #if os.path.exists("objective.dat"):
        ## Actually run PSI4.
        _exec("psi4", print_command=False)
        Ans = np.array([[float(i) for i in line.split()] for line in open("objective.dat").readlines()])
        os.unlink("objective.dat")
        return Ans
