"""@package BaseReader Base class for force field line reader.

@author Lee-Ping Wang
@date 12/2011
"""

class BaseReader(object):
    def __init__(self,fnm):
        self.ln     = 0
        self.itype  = fnm
        self.suffix = ''
        self.pdict  = {}
    
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

