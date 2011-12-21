""" @package qchemio Q-Chem input file parser. """

from basereader import BaseReader

class Reader(BaseReader):
    """Finite state machine for parsing Q-Chem input files.
    
    """
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(Reader,self).__init__(fnm)
