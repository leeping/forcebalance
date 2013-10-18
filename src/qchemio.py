""" @package forcebalance.qchemio Q-Chem input file parser. """

import os
from re import match, sub
from forcebalance import BaseReader
from forcebalance.nifty import *
from forcebalance.molecule import Molecule

from forcebalance.output import getLogger
logger=getLogger(__name__)

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
                logger.info(line + '\n')
            elif all([isfloat(i) for i in s]):
                self.cnum += 1
                logger.info("%s %s %s\n" % (line, str(self.snum), str(self.cnum)))
                self.suffix = '_at%s.sh%i.cf%i' % (self.atom,self.snum,self.cnum)

def QChem_Dielectric_Energy(fnm,wq):
    QCIn = Molecule(fnm)
    for i in range(QCIn.na):
        # Q-Chem crashes if it doesn't recognize the chemical element
        if QCIn.Data['elem'][i] in ['M','L']:
            QCIn.Data['elem'][i] = 'He'
    CalcDir=os.path.splitext(fnm)[0]+".d"
    GoInto(CalcDir)
    digits = len(str(QCIn.ns))
    for i in range(QCIn.ns):
        sdir = "%%0%ii" % digits % i
        GoInto(sdir)
        QCIn.write("qchem.in",select=i)
        queue_up(wq,"qchem40 qchem.in qchem.out",input_files=["qchem.in"],output_files=["qchem.out"],verbose=False)
        Leave(sdir)
    wq_wait(wq,verbose=False)
    PCM_Energies = []
    for i in range(QCIn.ns):
        sdir = "%%0%ii" % digits % i
        GoInto(sdir)
        for line in open("qchem.out"):
            if "PCM electrostatic energy" in line:
                PCM_Energies.append(float(line.split()[-2]))
        Leave(sdir)
    Leave(CalcDir)
    return np.array(PCM_Energies) * 2625.5
