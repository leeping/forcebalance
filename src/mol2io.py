""" @package forcebalance.mol2io Mol2 I/O.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 05/2012
"""

import os
from re import match, sub
from nifty import isint, isfloat
from numpy import array
from forcebalance import BaseReader
from subprocess import Popen, PIPE

mol2_pdict = {'COUL':{'Atom':[1], 6:''}}

class Mol2_Reader(BaseReader):
    """Finite state machine for parsing Mol2 force field file. (just for parameterizing the charges)"""
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(Tinker_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = mol2_pdict
        ## The atom numbers in the interaction (stored in the parser)
        self.atom   = []

    def feed(self, line):
        s          = line.split()
        self.ln   += 1
        # In mol2 files, the only defined interaction type is the Coulomb interaction.
        self.itype = 'COUL'

        if self.itype in self.pdict:
            if 'Atom' in self.pdict[self.itype]:
                # List the atoms in the interaction.
                self.atom = [s[i] for i in self.pdict[self.itype]['Atom']]
            # The suffix of the parameter ID is built from the atom
            # types/classes involved in the interaction.
            self.suffix = '.'.join(self.atom)

