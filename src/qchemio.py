""" @package qchemio Q-Chem input file parser. """

from re import match, sub
from basereader import BaseReader
from nifty import isfloat

## Types of counterpoise correction
#cptypes = [None, 'BASS', 'BASSP']
## Types of NDDO correction
ndtypes = [None]

##Section -> Interaction type dictionary.
#fdict = {
#    'basis'  : bastypes    }

##Interaction type -> Parameter Dictionary.
pdict = {'BASS':{0:'A', 1:'C'},
         'BASSP' :{0:'A', 1:'B', 2:'C'}
         }

class QCIn_Reader(BaseReader):
    """Finite state machine for parsing Q-Chem input files.
    
    """
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(QCIn_Reader,self).__init__(fnm)
        self.atom  = ''
        self.snum  = -1
        self.cnum  = -1
        self.shell = None
        self.pdict = pdict
        
    def feed(self, line):
        """ Feed in a line.

        @param[in] line     The line of data

        """
        line       = line.split('!')[0].strip()
        s          = line.split()
        #self.itype = None
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^!',line): return None, None
        # Now go through all the cases.
        if match('^\$',line):
            # Makes a word like "atoms", "bonds" etc.
            self.sec = sub('^\$','',line)
        elif self.sec == 'basis':
            if match('^[A-Za-z][a-z]* +0$',line):
                self.atom = s[0]
                self.snum = -1
            elif match('^[SPDFGH]P? +[0-9]+ +1\.0+$',line):
                self.snum += 1
                self.cnum  = -1
                self.shell = s[0]
                self.itype = 'BAS'+self.shell
                print line
            elif all([isfloat(i) for i in s]):
                self.cnum += 1
                print line, self.snum, self.cnum
                self.suffix = '_at%s.sh%i.cf%i' % (self.atom,self.snum,self.cnum)
