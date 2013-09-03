try:
    __import__("numpy")
except ImportError:
    print "Could not load numpy module, exiting..."
    exit()

try:
    __import__("scipy")
except ImportError:
    print "Could not load scipy module, exiting..."
    exit()

from re import split, findall
from collections import defaultdict, OrderedDict
import pkg_resources
__version__ = pkg_resources.get_distribution("forcebalance").version

from collections import OrderedDict
from parser import tgt_opts_defaults, gen_opts_defaults

class BaseClass(object):
    """ Provides some nifty functions that are common to all ForceBalance classes. """

    def __init__(self, options):
        self.PrintOptionDict  = OrderedDict()
        self.verbose_options  = options['verbose_options']
        
    def set_option(self, in_dict, src_key, dest_key = None, val = None, default = None, forceprint=False):
        if dest_key == None:
            dest_key = src_key
        if val == None:
            val = in_dict[src_key]
        if default == None:
            if src_key in gen_opts_defaults: 
                default = gen_opts_defaults[src_key]
            elif src_key in tgt_opts_defaults:
                default = tgt_opts_defaults[src_key]
            else: default = None
        if ((val != default or self.verbose_options) and dest_key != 'root') or forceprint:
            self.PrintOptionDict[dest_key] = val
        return super(BaseClass,self).__setattr__(dest_key, val)

class BaseReader(object):
    """ The 'reader' class.  It serves two main functions:

    1) When parsing a text force field file, the 'feed' method is
    called once for every line.  Calling the 'feed' method stores the
    internal variables that are needed for making the unique parameter
    identifier.

    2) The 'reader' also stores the 'pdict' dictionary, which is
    needed for building the matrix of rescaling factors.  This is not
    needed for the XML force fields, so in XML force fields pdict is
    replaced with a string called "XML_Override".

    """

    def __init__(self,fnm):
        self.ln     = 0
        self.itype  = fnm
        self.suffix = ''
        self.pdict  = {}
        ## The mapping of (this residue, atom number) to (atom name) for building atom-specific interactions in [ bonds ], [ angles ] etc.
        self.adict      = OrderedDict()
        ## The mapping of (molecule name) to a dictionary of  of atom types for the atoms in that residue.
        #self.moleculedict = OrderedDict()
        ## The listing of 'RES:ATOMNAMES' for atom names in the line
        ## This is obviously a placeholder.
        self.molatom = ("Sniffy",["Mao","Schmao"])

        self.Molecules = OrderedDict()
        self.AtomTypes = OrderedDict()

    def Split(self, line):
        return line.split()
    
    def Whites(self, line):
        return findall('[ ]+',line)
    
    def feed(self,line):
        self.ln += 1
        
    def build_pid(self, pfld):
        """ Returns the parameter type (e.g. K in BONDSK) based on the
        current interaction type.

        Both the 'pdict' dictionary (see gmxio.pdict) and the
        interaction type 'state' (here, BONDS) are needed to get the
        parameter type.

        If, however, 'pdict' does not contain the ptype value, a suitable
        substitute is simply the field number.

        Note that if the interaction type state is not set, then it
        defaults to the file name, so a generic parameter ID is
        'filename.line_num.field_num'
        
        """
        #print self.pdict[self.itype][pfld]
        ptype = self.pdict.get(self.itype,{}).get(pfld,':%i.%i' % (self.ln,pfld))
        return self.itype+ptype+self.suffix

# import parser, forcefield, optimizer, objective, output
