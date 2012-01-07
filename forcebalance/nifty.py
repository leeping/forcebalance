"""@package nifty Nifty functions, intended to be imported by any module.

Named after the mighty Sniffy Handy Nifty (King Sniffy)

@author Lee-Ping Wang
@date 12/2011
"""

import os
from re import match, sub
from numpy import array, diag, dot, mat, transpose
from numpy.linalg import norm, svd

## Boltzmann constant
kb = 0.0083144100163
## Q-Chem to GMX unit conversion for energy
eqcgmx = 2625.5002
## Q-Chem to GMX unit conversion for force
fqcgmx = -49621.9

def isint(word):
    """ONLY matches integers! If you have a decimal point? None shall pass!

    @param[in] word String (for instance, '123', '153.0', '2.', '-354')
    @return answer Boolean which specifies whether the string is an integer (only +/- sign followed by digits)
    
    """
    return match('^[-+]?[0-9]+$',word)

def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, what have you
    CAUTION - this will also match an integer.

    @param[in] word String (for instance, '123', '153.0', '2.', '-354')
    @return answer Boolean which specifies whether the string is any number
    
    """
    return match('^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$',word)

def isdecimal(word):
    """Matches things with a decimal only; see isint and isfloat.

    @param[in] word String (for instance, '123', '153.0', '2.', '-354')
    @return answer Boolean which specifies whether the string is a number with a decimal point
    
    """
    return isfloat(word) and not isint(word)

def orthogonalize(vec1, vec2):
    """Given two vectors vec1 and vec2, project out the component of vec1
    that is along the vec2-direction.

    @param[in] vec1 The projectee (i.e. output is some modified version of vec1)
    @param[in] vec2 The projector (component subtracted out from vec1 is parallel to this)
    @return answer A copy of vec1 but with the vec2-component projected out.
    """
    v2u = vec2/norm(vec2)
    return vec1 - v2u*dot(vec1, v2u)

def pmat2d(mat2d):
    """Printout of a 2-D matrix.

    @param[in] mat2d a 2-D matrix
    """
    for i in range(mat2d.shape[0]):
        for j in range(mat2d.shape[1]):
            print "% .1e" % mat2d[i][j],
        print

def printcool(text,sym="#",bold=False,color=2,bottom='-',minwidth=50):
    """Cool-looking printout for slick formatting of output.

    @param[in] text The string that the printout is based upon.  This function
    will print out the string, ANSI-colored and enclosed in the symbol
    for example:\n
    <tt> ################# </tt>\n
    <tt> ### I am cool ### </tt>\n
    <tt> ################# </tt>
    @param[in] sym The surrounding symbol\n
    @param[in] bold Whether to use bold print
    
    @param[in] color The ANSI color:\n
    1 red\n
    2 green\n
    3 yellow\n
    4 blue\n
    5 magenta\n
    6 cyan\n
    7 white
    
    @param[in] bottom The symbol for the bottom bar

    @param[in] minwidth The minimum width for the box, if the text is very short
    then we insert the appropriate number of padding spaces

    @return bar The bottom bar is returned for the user to print later, e.g. to mark off a 'section'    
    """
    text = text.split('\n')
    width = max(minwidth,max([len(line) for line in text]))
    bar = ''.join([sym for i in range(width + 8)])
    print '\n'+bar
    for line in text:
        padleft = ' ' * ((width - len(line)) / 2)
        padright = ' '* (width - len(line) - len(padleft))
        print "###\x1b[%s9%im%s" % (bold and "1;" or "", color,padleft),line,"%s\x1b[0m###" % padright
    print bar
    return sub(sym,bottom,bar)

def printcool_dictionary(dict,title="General options"):
    """See documentation for printcool; this is a nice way to print out keys/values in a dictionary.

    The keys in the dictionary are sorted before printing out.

    @param[in] dict The dictionary to be printed
    @param[in] title The title of the printout
    """
    bar = printcool(title)
    print '\n'.join(["%-25s %s " % (key,str(dict[key])) for key in sorted([i for i in dict]) if dict[key] != None])
    print bar

def col(vec):
    """
    Given any list, array, or matrix, return a 1-column matrix.

    Input:
    vec  = The input vector that is to be made into a column

    Output:
    A column matrix
    """
    return mat(array(vec).reshape(-1, 1))

def row(vec):
    """Given any list, array, or matrix, return a 1-row matrix.

    @param[in] vec The input vector that is to be made into a row

    @return answer A row matrix
    """
    return mat(array(vec).reshape(1, -1))

def flat(vec):
    """Given any list, array, or matrix, return a single-index array.

    @param[in] vec The data to be flattened
    @return answer The flattened data
    """
    return array(vec).reshape(-1)

def floatornan(word):
    """Returns a big number if we encounter NaN.

    @param[in] word The string to be converted
    @return answer The string converted to a float; if not a float, return 1e10
    @todo I could use suggestions for making this better.
    """
    big = 1e10
    if isfloat(word):
        return float(word)
    else:
        print "Setting %s to % .1e" % big
        return big

def multiopen(arg):
    """
    This function be given any of several variable types
    (single file name, file object, or list of lines, or a list of )
    and give a list of files:

    [file1, file2, file3 ... ]

    each of which can then be iterated over:

    [[file1_line1, file1_line2 ... ], [file2_line1, file2_line2 ... ]]
    """
    if type(arg) == str:
        # A single file name
        fins = [open(arg)]
    elif type(arg) == file:
        # A file object
        fins = [[arg]]
    elif type(arg) == list:
        if all([type(l) == str for l in arg]):
            # A list of lines (as in, open(file).readlines()) is expected to end with \n on most of the lines.
            if any([match("^.*\n$",l) for l in arg]):
                fins = [[arg]]
            # In contrast, a list of file names doesn't have \n characters.
            else:
                fins = [open(l) for l in arg]
        elif all([type(l) == file or type(l) == list for l in arg]):
            fins = arg
        else:
            print "What did you give this program as input?"
            print arg
            exit(1)
    else:
        print "What did you give this program as input?"
        print arg
        exit(1)
    return fins

def remove_if_exists(fnm):
    if os.path.exists(fnm):
        os.remove(fnm)

def invert_svd(X,thresh=1e-8):
    u,s,vh = svd(X, full_matrices=0)
    uh     = mat(transpose(u))
    v      = mat(transpose(vh))
    si     = s.copy()
    for i in range(s.shape[0]):
        if abs(s[i]) > 1e-8:
            si[i] = 1./s[i]
        else:
            si[i] = 0.0
    si     = mat(diag(si))
    Xt     = v*si*uh
    return Xt
