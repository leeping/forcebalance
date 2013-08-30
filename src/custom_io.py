""" @package forcebalance.custom_io Custom force field parser.

We take advantage of the sections in GROMACS and the 'interaction
type' concept, but these interactions are not supported in GROMACS;
rather, they are computed within our program.

@author Lee-Ping Wang
@date 12/2011
"""

from re import match, sub
from forcebalance import BaseReader

## Types of counterpoise correction
cptypes = [None, 'CPGAUSS', 'CPEXPG', 'CPGEXP']
## Types of NDDO correction
ndtypes = [None]

##Section -> Interaction type dictionary.
fdict = {
    'counterpoise'  : cptypes    }

##Interaction type -> Parameter Dictionary.
pdict = {'CPGAUSS':{3:'A', 4:'B', 5:'C'},
         'CPGEXP' :{3:'A', 4:'B', 5:'G', 6:'X'},
         'CPEXPG' :{3:'A1', 4:'B', 5:'X0', 6:'A2'}
         }

class Gen_Reader(BaseReader):
    """Finite state machine for parsing custom GROMACS force field files.

    This class is instantiated when we begin to read in a file.  The
    feed(line) method updates the state of the machine, giving it
    information like the residue we're currently on, the nonbonded
    interaction type, and the section that we're in.  Using this
    information we can look up the interaction type and parameter type
    for building the parameter ID.
    
    """
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(Gen_Reader,self).__init__(fnm)
        ## The current section that we're in
        self.sec = None
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict

    def feed(self, line):
        """ Feed in a line.

        @param[in] line     The line of data

        """
        s          = line.split()
        atom       = []
        self.itype = None
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^;',line): return None, None
        # Now go through all the cases.
        if match('^\[.*\]',line):
            # Makes a word like "atoms", "bonds" etc.
            self.sec = sub('[\[\] \n]','',line)
        elif self.sec == 'counterpoise':
            self.itype = cptypes[int(s[2])]
            atom = [s[0],s[1]]
        elif self.sec == 'NDDO':
            # NDDO hasn't been tested since the refactoring.
            self.itype = '_'.join(['NDDO', s[0], s[1]])
        else:
            return [],"Confused"
        if len(atom) > 1 and atom[0] > atom[-1]:
            # Enforce a canonical ordering of the atom labels in a parameter ID
            atom = atom[::-1]
        self.suffix = ''.join(atom)
