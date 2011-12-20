""" @package custom_io Custom force field parser.

We take advantage of the sections in GROMACS and the 'interaction
type' concept, but these interactions are not supported in GROMACS;
rather, they are computed within our program.

@author Lee-Ping Wang
@date 12/2011
"""

from re import match

## Types of counterpoise correction
CPTYPES = [None, 'CPGAUSS', 'CPEXPG', 'CPGEXP']
## Types of NDDO correction
NDTYPES = [None]

##Section -> Interaction type dictionary.
FDICT = {
    'counterpoise'  : CPTYPES    }

##Interaction type -> Parameter Dictionary.
PDICT = {'CPGAUSS':{3:'A', 4:'B', 5:'C'},
         'CPGEXP' :{3:'A', 4:'B', 5:'G', 6:'X'},
         'CPEXPG' :{3:'A1', 4:'B', 5:'X0', 6:'A2'}
         }

def makeaftype(line, section):
    """ Given a line, section name and list of atom names,
    return the interaction type and the atoms involved.

    @param[in] line     The line of data
    @param[in] section  The section that we're in
    @return atom    The atoms involved in the interaction
    @return ftype   The interaction type
    
    """
    sline = line.split()
    atom = []
    ftype = None
    # No sense in doing anything for an empty line or a comment line.
    if len(sline) == 0 or match('^;', line):
        return None, None
    # Now go through all the cases.
    if section == 'counterpoise':
        atom = sline[0] < sline[1] and \
               [sline[0], sline[1]] or [sline[1], sline[0]]
        ftype = CPTYPES[int(sline[2])]
    elif section == 'NDDO':
        # NDDO hasn't been tested since the refactoring.
        ftype = '_'.join(['NDDO', sline[0], sline[1]])
    else:
        return [],"Confused"
    return atom, ftype
