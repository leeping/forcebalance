"""@package forcebalance.nifty Nifty functions, intended to be imported by any module within ForceBalance.

Table of Contents:
- I/O formatting
- Math: Variable manipulation, linear algebra, least squares polynomial fitting
- Pickle: Expand Python's own pickle to accommodate writing XML etree objects
- Commands for submitting things to the Work Queue
- Various file and process management functions
- Development stuff (not commonly used)

Named after the mighty Sniffy Handy Nifty (King Sniffy)

@author Lee-Ping Wang
@date 12/2011
"""

from select import select
import os, sys, shutil
from re import match, sub
import numpy as np
import itertools
import threading
import pickle
import time
import subprocess
from subprocess import PIPE, STDOUT
from collections import OrderedDict, defaultdict

import forcebalance
from forcebalance.output import *
logger = getLogger(__name__)

# import IPython as ip # For debugging

## Boltzmann constant
kb = 0.0083144100163
## Q-Chem to GMX unit conversion for energy
eqcgmx = 2625.5002
## Q-Chem to GMX unit conversion for force
fqcgmx = -49621.9
## One bohr equals this many angstroms
bohrang = 0.529177249

#=========================#
#     I/O formatting      #
#=========================#
def pvec1d(vec1d, precision=1, loglevel=INFO):
    """Printout of a 1-D vector.

    @param[in] vec1d a 1-D vector
    """
    v2a = np.array(vec1d)
    for i in range(v2a.shape[0]):
        logger.log(loglevel, "%% .%ie " % precision % v2a[i])
    logger.log(loglevel, '\n')

def pmat2d(mat2d, precision=1, loglevel=INFO):
    """Printout of a 2-D matrix.

    @param[in] mat2d a 2-D matrix
    """
    m2a = np.array(mat2d)
    for i in range(m2a.shape[0]):
        for j in range(m2a.shape[1]):
            logger.log(loglevel, "%% .%ie " % precision % m2a[i][j])
        logger.log(loglevel, '\n')

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    lzip = [[j for j in i if j != None] for i in list(itertools.izip_longest(*args))]
    return lzip

def encode(l): 	
    return [[len(list(group)),name] for name, group in itertools.groupby(l)]

def segments(e):
    # Takes encoded input.
    begins = np.array([sum([k[0] for k in e][:j]) for j,i in enumerate(e) if i[1] == 1])
    lens = np.array([i[0] for i in e if i[1] == 1])
    return [(i, i+j) for i, j in zip(begins, lens)]

def commadash(l):
    # Formats a list like [27, 28, 29, 30, 31, 88, 89, 90, 91, 100, 136, 137, 138, 139]
    # into '27-31,88-91,100,136-139
    L = sorted(l)
    if len(L) == 0:
        return "(empty)"
    L.append(L[-1]+1)
    LL = [i in L for i in range(L[-1])]
    return ','.join('%i-%i' % (i[0]+1,i[1]) if (i[1]-1 > i[0]) else '%i' % (i[0]+1) for i in segments(encode(LL)))

def uncommadash(s):
    # Takes a string like '27-31,88-91,100,136-139'
    # and turns it into a list like [27, 28, 29, 30, 31, 88, 89, 90, 91, 100, 136, 137, 138, 139]
    L = []
    try:
        for w in s.split(','):
            ws = w.split('-')
            a = int(ws[0])-1
            if len(ws) == 1:
                b = int(ws[0])
            elif len(ws) == 2:
                b = int(ws[1])
            else:
                logger.warning("Dash-separated list cannot exceed length 2\n")
                raise
            if a < 0 or b <= 0 or b <= a:
                if a < 0 or b <= 0:
                    logger.warning("Items in list cannot be zero or negative: %d %d\n" % (a, b))
                else:
                    logger.warning("Second number cannot be larger than first: %d %d\n" % (a, b))
                raise
            newL = range(a,b)
            if any([i in L for i in newL]):
                logger.warning("Duplicate entries found in list\n")
                raise
            L += newL
        if sorted(L) != L:
            logger.warning("List is out of order\n")
            raise
    except:
        raise Exception('Invalid string for converting to list of numbers: %s' % s)
    return L
            
#list(itertools.chain(*[range(*(int(w.split('-')[0])-1, int(w.split('-')[1]) if len(w.split('-')) == 2 else int(w.split('-')[0])))  for w in Mao.split(',')]))

def printcool(text,sym="#",bold=False,color=2,ansi=None,bottom='-',minwidth=50,center=True):
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
    def newlen(l):
        return len(sub("\x1b\[[0-9;]*m","",line))
    text = text.split('\n')
    width = max(minwidth,max([newlen(line) for line in text]))
    bar = ''.join(["=" for i in range(width + 6)])
    bar = sym + bar + sym
    #bar = ''.join([sym for i in range(width + 8)])
    logger.info('\r'+bar + '\n')
    for line in text:
        if center:
            padleft = ' ' * ((width - newlen(line)) / 2)
        else:
            padleft = ''
        padright = ' '* (width - newlen(line) - len(padleft))
        if ansi != None:
            ansi = str(ansi)
            logger.info("%s| \x1b[%sm%s " % (sym, ansi, padleft)+line+" %s\x1b[0m |%s\n" % (padright, sym))
        elif color != None:
            if color == 0 and bold:
                logger.info("%s| \x1b[1m%s " % (sym, padleft) + line + " %s\x1b[0m |%s\n" % (padright, sym))
            elif color == 0:
                logger.info("%s| %s " % (sym, padleft)+line+" %s |%s\n" % (padright, sym))
            else:
                logger.info("%s| \x1b[%s9%im%s " % (sym, bold and "1;" or "", color, padleft)+line+" %s\x1b[0m |%s\n" % (padright, sym))
            # if color == 3 or color == 7:
            #     print "%s\x1b[40m\x1b[%s9%im%s" % (''.join([sym for i in range(3)]), bold and "1;" or "", color, padleft),line,"%s\x1b[0m%s" % (padright, ''.join([sym for i in range(3)]))
            # else:
            #     print "%s\x1b[%s9%im%s" % (''.join([sym for i in range(3)]), bold and "1;" or "", color, padleft),line,"%s\x1b[0m%s" % (padright, ''.join([sym for i in range(3)]))
        else:
            warn_press_key("Inappropriate use of printcool")
    logger.info(bar + '\n')
    botbar = ''.join([bottom for i in range(width + 8)])
    return botbar + '\n'

def printcool_dictionary(Dict,title="General options",bold=False,color=2,keywidth=25,topwidth=50,center=True,leftpad=0):
    """See documentation for printcool; this is a nice way to print out keys/values in a dictionary.

    The keys in the dictionary are sorted before printing out.

    @param[in] dict The dictionary to be printed
    @param[in] title The title of the printout
    """
    if Dict == None: return
    bar = printcool(title,bold=bold,color=color,minwidth=topwidth,center=center)
    def magic_string(str):
        # This cryptic command returns a string with the number of characters specified as a variable. :P
        # Useful for printing nice-looking dictionaries, i guess.
        #print "\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"'))
        return eval("\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"')))
    if isinstance(Dict, OrderedDict): 
        logger.info('\n'.join([' '*leftpad + "%s %s " % (magic_string(str(key)),str(Dict[key])) for key in Dict if Dict[key] != None]))
    else:
        logger.info('\n'.join([' '*leftpad + "%s %s " % (magic_string(str(key)),str(Dict[key])) for key in sorted([i for i in Dict]) if Dict[key] != None]))
    logger.info("\n%s" % bar)

#===============================#
#| Math: Variable manipulation |#
#===============================#
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
        logger.info("Setting %s to % .1e\n" % big)
        return big

def col(vec):
    """
    Given any list, array, or matrix, return a 1-column matrix.

    Input:
    vec  = The input vector that is to be made into a column

    Output:
    A column matrix
    """
    return np.mat(np.array(vec).reshape(-1, 1))

def row(vec):
    """Given any list, array, or matrix, return a 1-row matrix.

    @param[in] vec The input vector that is to be made into a row

    @return answer A row matrix
    """
    return np.mat(np.array(vec).reshape(1, -1))

def flat(vec):
    """Given any list, array, or matrix, return a single-index array.

    @param[in] vec The data to be flattened
    @return answer The flattened data
    """
    return np.array(vec).reshape(-1)

#====================================#
#| Math: Vectors and linear algebra |#
#====================================#
def orthogonalize(vec1, vec2):
    """Given two vectors vec1 and vec2, project out the component of vec1
    that is along the vec2-direction.

    @param[in] vec1 The projectee (i.e. output is some modified version of vec1)
    @param[in] vec2 The projector (component subtracted out from vec1 is parallel to this)
    @return answer A copy of vec1 but with the vec2-component projected out.
    """
    v2u = vec2/np.linalg.norm(vec2)
    return vec1 - v2u*np.dot(vec1, v2u)

def invert_svd(X,thresh=1e-12):
    
    """ 

    Invert a matrix using singular value decomposition. 
    @param[in] X The matrix to be inverted
    @param[in] thresh The SVD threshold; eigenvalues below this are not inverted but set to zero
    @return Xt The inverted matrix

    """

    u,s,vh = np.linalg.svd(X, full_matrices=0)
    uh     = np.mat(np.transpose(u))
    v      = np.mat(np.transpose(vh))
    si     = s.copy()
    for i in range(s.shape[0]):
        if abs(s[i]) > thresh:
            si[i] = 1./s[i]
        else:
            si[i] = 0.0
    si     = np.mat(np.diag(si))
    Xt     = v*si*uh
    return Xt

#==============================#
#|    Linear least squares    |#
#==============================#
def get_least_squares(x, y, w = None, thresh=1e-12):
    """
    @code
     __                  __
    |                      |
    | 1 (x0) (x0)^2 (x0)^3 |
    | 1 (x1) (x1)^2 (x1)^3 |
    | 1 (x2) (x2)^2 (x2)^3 |
    | 1 (x3) (x3)^2 (x3)^3 |
    | 1 (x4) (x4)^2 (x4)^3 |
    |__                  __|

    @endcode

    @param[in] X (2-D array) An array of X-values (see above)
    @param[in] Y (array) An array of Y-values (only used in getting the least squares coefficients)
    @param[in] w (array) An array of weights, hopefully normalized to one.
    @param[out] Beta The least-squares coefficients
    @param[out] Hat The hat matrix that takes linear combinations of data y-values to give fitted y-values (weights)
    @param[out] yfit The fitted y-values
    @param[out] MPPI The Moore-Penrose pseudoinverse (multiply by Y to get least-squares coefficients, multiply by dY/dk to get derivatives of least-squares coefficients)
    """
    # X is a 'tall' matrix.
    X = np.mat(x)
    Y = col(y)
    n_x = X.shape[0]
    n_fit = X.shape[1]
    if n_fit > n_x:
        logger.warning("Argh? It seems like this problem is underdetermined!\n")
    # Build the weight matrix.
    if w != None:
        if len(w) != n_x:
            warn_press_key("The weight array length (%i) must be the same as the number of 'X' data points (%i)!" % len(w), n_x)
        w /= np.mean(w)
        WH = np.mat(np.diag(w**0.5))
    else:
        WH = np.mat(np.eye(n_x))
    # Make the Moore-Penrose Pseudoinverse.
    # if n_fit == n_x:
    #     MPPI = np.linalg.inv(WH*X)
    # else:
    # This resembles the formula (X'WX)^-1 X' W^1/2
    MPPI = np.linalg.pinv(WH*X)
    Beta = MPPI * WH * Y
    Hat = WH * X * MPPI
    yfit = flat(Hat * Y)
    # Return three things: the least-squares coefficients, the hat matrix (turns y into yfit), and yfit
    # We could get these all from MPPI, but I might get confused later on, so might as well do it here :P
    return Beta, Hat, yfit, MPPI

#===========================================#
#| John's statisticalInefficiency function |#
#===========================================#
def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3, warn=True):

    """
    Compute the (cross) statistical inefficiency of (two) timeseries.

    Notes
      The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
      The fast method described in Ref [1] is used to compute g.

    References
      [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
      histogram analysis method for the analysis of simulated and parallel tempering simulations.
      JCTC 3(1):26-41, 2007.

    Examples

    Compute statistical inefficiency of timeseries data with known correlation time.

    >>> import timeseries
    >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
    >>> g = statisticalInefficiency(A_n, fast=True)

    @param[in] A_n (required, numpy array) - A_n[n] is nth value of
    timeseries A.  Length is deduced from vector.

    @param[in] B_n (optional, numpy array) - B_n[n] is nth value of
    timeseries B.  Length is deduced from vector.  If supplied, the
    cross-correlation of timeseries A and B will be estimated instead of
    the autocorrelation of timeseries A.

    @param[in] fast (optional, boolean) - if True, will use faster (but
    less accurate) method to estimate correlation time, described in
    Ref. [1] (default: False)

    @param[in] mintime (optional, int) - minimum amount of correlation
    function to compute (default: 3) The algorithm terminates after
    computing the correlation time out to mintime when the correlation
    function furst goes negative.  Note that this time may need to be
    increased if there is a strong initial negative peak in the
    correlation function.

    @return g The estimated statistical inefficiency (equal to 1 + 2
    tau, where tau is the correlation time).  We enforce g >= 1.0.

    """

    # Create numpy copies of input arguments.
    A_n = np.array(A_n)
    if B_n is not None:
        B_n = np.array(B_n)
    else:
        B_n = np.array(A_n)
    # Get the length of the timeseries.
    N = A_n.size
    # Be sure A_n and B_n have the same dimensions.
    if(A_n.shape != B_n.shape):
        raise ParameterError('A_n and B_n must have same dimensions.')
    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0
    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()
    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B
    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1
    # Trap the case where this covariance is zero, and we cannot proceed.
    if(sigma2_AB == 0):
        if warn:
            logger.warning('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency')
        return 1.0
    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while (t < N-1):
        # compute normalized fluctuation correlation function at time t
        C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break
        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)
        # Increment t and the amount by which we increment t.
        t += increment
        # Increase the interval if "fast mode" is on.
        if fast: increment += 1
    # g must be at least unity
    if (g < 1.0): g = 1.0
    # Return the computed statistical inefficiency.
    return g

#==============================#
#|      XML Pickle stuff      |#
#==============================#
try:
    from lxml import etree
except: 
    logger.warning("lxml module import failed (You can't use OpenMM or XML force fields)\n")
## Pickle uses 'flags' to pickle and unpickle different variable types.
## Here we use the letter 'x' to signify that the variable type is an XML file.
XMLFILE='x'

class Pickler_LP(pickle.Pickler):
    """ A subclass of the python Pickler that implements pickling of _ElementTree types. """
    def __init__(self, file, protocol=None):
        pickle.Pickler.__init__(self, file, protocol)
        ## The element tree is saved as a string.
        def save_etree(self, obj):
            try:
                ## Convert the element tree to string.
                String = etree.tostring(obj)
                ## The rest is copied from the Pickler class
                if self.bin:
                    logger.error("self.bin is True, not sure what to do with myself\n")
                    raw_input()
                else:
                    self.write(XMLFILE + repr(String) + '\n')
                self.memoize(String)
            except:
                warn_once("Cannot save XML files; if using OpenMM install libxml2+libxslt+lxml.  Otherwise don't worry.")
        try:
            self.dispatch[etree._ElementTree] = save_etree
        except:
            warn_once("Cannot save XML files; if using OpenMM install libxml2+libxslt+lxml.  Otherwise don't worry.")

class Unpickler_LP(pickle.Unpickler):
    """ A subclass of the python Unpickler that implements unpickling of _ElementTree types. """
    def __init__(self, file):
        pickle.Unpickler.__init__(self, file)
        def load_etree(self):
            try:
                ## This stuff is copied from the Unpickler class
                rep = self.readline()[:-1]
                for q in "\"'": # double or single quote
                    if rep.startswith(q):
                        if not rep.endswith(q):
                            raise ValueError, "insecure string pickle"
                        rep = rep[len(q):-len(q)]
                        break
                else:
                    raise ValueError, "insecure string pickle"
                ## The string is converted to an _ElementTree type before it is finally loaded.
                self.append(etree.ElementTree(etree.fromstring(rep.decode("string-escape"))))
            except:
                warn_once("Cannot load XML files; if using OpenMM install libxml2+libxslt+lxml.  Otherwise don't worry.")
        try:
            self.dispatch[XMLFILE] = load_etree
        except:
            warn_once("Cannot load XML files; if using OpenMM install libxml2+libxslt+lxml.  Otherwise don't worry.")

def lp_dump(obj, file, protocol=None):
    """ Use this instead of pickle.dump for pickling anything that contains _ElementTree types. """
    Pickler_LP(file, protocol).dump(obj)

def lp_load(file):
    """ Use this instead of pickle.load for unpickling anything that contains _ElementTree types. """
    return Unpickler_LP(file).load()

#==============================#
#|      Work Queue stuff      |#
#==============================#
try:
    import work_queue
except:
    logger.warning("Work Queue library import fail (You can't queue up jobs using Work Queue)\n")

# Global variable corresponding to the Work Queue object
WORK_QUEUE = None

# Global variable containing a mapping from target names to Work Queue task IDs
WQIDS = defaultdict(list)

def getWorkQueue():
    global WORK_QUEUE
    return WORK_QUEUE

def getWQIds():
    global WQIDS
    return WQIDS

def createWorkQueue(wq_port, debug=True):
    global WORK_QUEUE
    if debug:
        work_queue.set_debug_flag('all')
    WORK_QUEUE = work_queue.WorkQueue(port=wq_port, catalog=True, exclusive=False, shutdown=False)
    WORK_QUEUE.tasks_failed = 0 # Counter for tasks that fail at the application level
    WORK_QUEUE.specify_name('forcebalance')
    #WORK_QUEUE.specify_keepalive_timeout(8640000)
    WORK_QUEUE.specify_keepalive_interval(8640000)

def destroyWorkQueue():
    # Convenience function to destroy the Work Queue objects.
    global WORK_QUEUE, WQIDS
    WORK_QUEUE = None
    WQIDS = defaultdict(list)

def queue_up(wq, command, input_files, output_files, tgt=None, verbose=True):
    """ 
    Submit a job to the Work Queue.

    @param[in] wq (Work Queue Object)
    @param[in] command (string) The command to run on the remote worker.
    @param[in] input_files (list of files) A list of locations of the input files.
    @param[in] output_files (list of files) A list of locations of the output files.
    """
    global WQIDS
    task = work_queue.Task(command)
    cwd = os.getcwd()
    for f in input_files:
        lf = os.path.join(cwd,f)
        task.specify_input_file(lf,f,cache=False)
    for f in output_files:
        lf = os.path.join(cwd,f)
        task.specify_output_file(lf,f,cache=False)
    task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_FCFS)
    task.specify_tag(command)
    taskid = wq.submit(task)
    if verbose:
        logger.info("Submitting command '%s' to the Work Queue, taskid %i\n" % (command, taskid))
    if tgt != None:
        WQIDS[tgt.name].append(taskid)
    else:
        WQIDS["None"].append(taskid)
    
def queue_up_src_dest(wq, command, input_files, output_files, tgt=None, verbose=True):
    """ 
    Submit a job to the Work Queue.  This function is a bit fancier in that we can explicitly
    specify where the input files come from, and where the output files go to.

    @param[in] wq (Work Queue Object)
    @param[in] command (string) The command to run on the remote worker.
    @param[in] input_files (list of 2-tuples) A list of local and
    remote locations of the input files.
    @param[in] output_files (list of 2-tuples) A list of local and
    remote locations of the output files.
    """
    global WQIDS
    task = work_queue.Task(command)
    for f in input_files:
        # print f[0], f[1]
        task.specify_input_file(f[0],f[1],cache=False)
    for f in output_files:
        # print f[0], f[1]
        task.specify_output_file(f[0],f[1],cache=False)
    task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_FCFS)
    task.specify_tag(command)
    taskid = wq.submit(task)
    if verbose:
        logger.info("Submitting command '%s' to the Work Queue, taskid %i\n" % (command, taskid))
    if tgt != None:
        WQIDS[tgt.name].append(taskid)
    else:
        WQIDS["None"].append(taskid)

def wq_wait1(wq, wait_time=10, wait_intvl=1, print_time=60, verbose=False):
    """ This function waits ten seconds to see if a task in the Work Queue has finished. """
    global WQIDS
    if verbose: logger.info('---\n')
    if wait_intvl >= wait_time:
        wait_time = wait_intvl
        numwaits = 1
    else:
        numwaits = wait_time / wait_intvl
    for sec in range(numwaits):
        task = wq.wait(wait_intvl)
        if task:
            exectime = task.cmd_execution_time/1000000
            if verbose:
                logger.info('A job has finished!\n')
                logger.info('Job name = ' + task.tag + 'command = ' + task.command + '\n')
                logger.info("status = " + task.status + '\n') 
                logger.info("return_status = " + task.return_status)
                logger.info("result = " + task.result) 
                logger.info("host = " + task.hostname + '\n')
                logger.info("execution time = " + exectime) 
                logger.info("total_bytes_transferred = " + task.total_bytes_transferred + '\n')
            if task.result != 0:
                oldid = task.id
                oldhost = task.hostname
                tgtname = "None"
                for tnm in WQIDS:
                    if task.id in WQIDS[tnm]:
                        tgtname = tnm
                        WQIDS[tnm].remove(task.id)
                taskid = wq.submit(task)
                logger.warning("Command '%s' (task %i) failed on host %s (%i seconds), resubmitted: taskid %i\n" % (task.command, oldid, oldhost, exectime, taskid))
                WQIDS[tgtname].append(taskid)
                wq.tasks_failed += 1
            else:
                if exectime > print_time: # Assume that we're only interested in printing jobs that last longer than a minute.
                    logger.info("Command '%s' (task %i) finished successfully on host %s (%i seconds)\n" % (task.command, task.id, task.hostname, exectime))
                for tnm in WQIDS:
                    if task.id in WQIDS[tnm]:
                        WQIDS[tnm].remove(task.id)
                del task
        try:
            # Full workers were added with CCTools 4.0.1
            nbusy = wq.stats.workers_busy + wq.stats.workers_full
        except:
            nbusy = wq.stats.workers_busy

        try:
            Complete = wq.stats.total_tasks_complete - wq.tasks_failed
            Total = wq.stats.total_tasks_dispatched - wq.tasks_failed
        except:
            logger.warning("wq object has no tasks_failed attribute, please use createWorkQueue() function.\n")
            Complete = wq.stats.total_tasks_complete
            Total = wq.stats.total_tasks_dispatched
            
        if verbose:
            logger.info("Workers: %i init, %i ready, %i busy, %i total joined, %i total removed\n" \
                % (wq.stats.workers_init, wq.stats.workers_ready, nbusy, wq.stats.total_workers_joined, wq.stats.total_workers_removed))
            logger.info("Tasks: %i running, %i waiting, %i total dispatched, %i total complete\n" \
                % (wq.stats.tasks_running,wq.stats.tasks_waiting,Total,Complete))
            logger.info("Data: %i / %i kb sent/received\n" % (wq.stats.total_bytes_sent/1000, wq.stats.total_bytes_received/1024))
        else:
            logger.info("\r%s : %i/%i workers busy; %i/%i jobs complete\r" %\
            (time.ctime(),
             nbusy, (wq.stats.total_workers_joined - wq.stats.total_workers_removed),
             Complete, Total)) 
            if time.time() - wq_wait1.t0 > 900:
                wq_wait1.t0 = time.time()
                logger.info('\n')
wq_wait1.t0 = time.time()

def wq_wait(wq, wait_time=10, wait_intvl=10, print_time=60, verbose=False):
    """ This function waits until the work queue is completely empty. """
    while not wq.empty():
        wq_wait1(wq, wait_time=wait_time, wait_intvl=wait_intvl, print_time=print_time, verbose=verbose)

#=====================================#
#| File and process management stuff |#
#=====================================#
# Search for exactly one file with a provided extension.
# ext: File extension
# arg: String name of an argument
# kwargs: Dictionary of keyword arguments.
def onefile(ext, arg=None):
    fnm = None
    if arg != None:
        if os.path.exists(arg):
            fnm = os.path.basename(arg)
        else:
            logger.warning("File specified by %s (%s) does not exist - will try to autodetect\n" % (ext, arg))

    if fnm == None:
        cwd = os.getcwd()
        ls = [i for i in os.listdir(cwd) if i.endswith('.%s' % ext)]
        if len(ls) != 1:
            logger.info("Found %i .%s files in %s\n" % (len(ls), ext, cwd))
        else:
            fnm = os.path.basename(ls[0])
            logger.info("Autodetected .%s in %s\n" % (fnm, cwd))
    return fnm

def GoInto(Dir):
    if os.path.exists(Dir):
        if os.path.isdir(Dir): pass
        else: raise Exception("Tried to create directory %s, it exists but isn't a directory" % newdir)
    else:
        os.makedirs(Dir)
    os.chdir(Dir)

def allsplit(Dir):
    # Split a directory into all directories involved.
    s = os.path.split(os.path.normpath(Dir))
    if s[1] == '' or s[1] == '.' : return []
    return allsplit(s[0]) + [s[1]]

def Leave(Dir):
    if os.path.split(os.getcwd())[1] != Dir:
        raise Exception("Trying to leave directory %s, but we're actually in directory %s (check your code)" % (Dir,os.path.split(os.getcwd())[1]))
    for i in range(len(allsplit(Dir))):
        os.chdir('..')

# Dictionary containing specific error messages for specific missing files or file patterns
specific_lst = [(['mdrun','grompp','trjconv','g_energy','g_traj'], "Make sure to install GROMACS and add it to your path (or set the gmxpath option)"),
                (['force.mdin', 'stage.leap'], "This file is needed for setting up AMBER force matching targets"),
                (['conf.pdb', 'mono.pdb'], "This file is needed for setting up OpenMM condensed phase property targets"),
                (['liquid.xyz', 'liquid.key', 'mono.xyz', 'mono.key'], "This file is needed for setting up OpenMM condensed phase property targets"),
                (['dynamic', 'analyze', 'minimize', 'testgrad', 'vibrate', 'optimize', 'polarize', 'superpose'], "Make sure to install TINKER and add it to your path (or set the tinkerpath option)"),
                (['runcuda.sh', 'npt.py', 'npt_tinker.py'], "This file belongs in the ForceBalance source directory, not sure why it is missing"),
                (['input.xyz'], "This file is needed for TINKER molecular property targets"),
                (['.*key$', '.*xyz$'], "I am guessing this file is probably needed by TINKER"),
                (['.*gro$', '.*top$', '.*itp$', '.*mdp$', '.*ndx$'], "I am guessing this file is probably needed by GROMACS")
                ]

# Build a dictionary mapping all of the keys in the above lists to their error messages
specific_dct = dict(list(itertools.chain(*[[(j,i[1]) for j in i[0]] for i in specific_lst])))

def MissingFileInspection(fnm):
    fnm = os.path.split(fnm)[1]
    answer = ""
    for key in specific_dct:
        if answer == "":
            answer += "\n"
        if match(key, fnm):
            answer += "%s\n" % specific_dct[key]
    return answer

def wopen(dest):
    """ Never write to symbolic links. """
    if os.path.islink(dest):
        raise Exception("Tried to write to a symbolic link but that's against the rules; probably indicates a coding error.")
    return open(dest,'w')

def LinkFile(src, dest, nosrcok = False):
    if os.path.abspath(src) == os.path.abspath(dest): return
    if os.path.exists(src):
        if os.path.exists(dest):
            if os.path.islink(dest): pass
            else: raise Exception("Tried to create symbolic link %s to %s, destination exists but isn't a symbolic link" % (src, dest))
        else:
            os.symlink(src, dest)
    else:
        if not nosrcok:
            raise Exception("Tried to create symbolic link %s to %s, but source file doesn't exist%s" % (src,dest,MissingFileInspection(src)))
    

def CopyFile(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            if os.path.islink(dest): 
                raise Exception("Tried to copy %s to %s, destination exists but it's a symbolic link" % (src, dest))
        else:
            shutil.copy2(src, dest)
    else:
        raise Exception("Tried to copy %s to %s, but source file doesn't exist%s" % (src,dest,MissingFileInspection(src)))

def link_dir_contents(abssrcdir, absdestdir):
    for fnm in os.listdir(abssrcdir):
        srcfnm = os.path.join(abssrcdir, fnm)
        destfnm = os.path.join(absdestdir, fnm)
        if os.path.isfile(srcfnm):
            if not os.path.exists(destfnm):
                #print "Linking %s to %s" % (srcfnm, destfnm)
                os.symlink(srcfnm, destfnm)

def remove_if_exists(fnm):
    """ Remove the file if it exists (doesn't return an error). """
    if os.path.exists(fnm):
        os.remove(fnm)

def which(fnm):
    # Get the location of a file.  Works only on UNIX-like file systems.
    try:
        return os.path.split(os.popen('which %s 2> /dev/null' % fnm).readlines()[0].strip())[0]
    except:
        return ''

# Thanks to cesarkawakami on #python (IRC freenode) for this code.
class LineChunker(object):
    def __init__(self, callback):
        self.callback = callback
        self.buf = ""

    def push(self, data):
        self.buf += data
        self.nomnom()

    def close(self):
        if self.buf:
            self.callback(self.buf + "\n")

    def nomnom(self):
        # Splits buffer by new line or carriage return, and passes
        # the splitted results onto processing.
        while "\n" in self.buf or "\r" in self.buf:
            chunk, sep, self.buf = re.split(r"(\r|\n)", self.buf, maxsplit=1)
            self.callback(chunk + sep)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

def _exec(command, print_to_screen = False, outfnm = None, logfnm = None, stdin = "", print_command = True, copy_stdout = True, copy_stderr = False, persist = False, expand_cr=False, **kwargs):
    """Runs command line using subprocess, optionally returning stdout.
    Options:
    command (required) = Name of the command you want to execute
    outfnm (optional) = Name of the output file name (overwritten if exists)
    logfnm (optional) = Name of the log file name (appended if exists)
    stdin (optional) = A string to be passed to stdin, as if it were typed (use newline character to mimic Enter key)
    print_command = Whether to print the command.
    copy_stdout = Copy the stdout stream; can set to False in strange situations
    copy_stderr = Copy the stderr stream to the stdout stream; useful for GROMACS which prints out everything to stderr (argh.)
    expand_cr = Whether to expand carriage returns into newlines (useful for GROMACS mdrun).
    persist = Continue execution even if the command gives a nonzero return code.
    """
    # Dictionary of options to be passed to the Popen object.
    cmd_options={'shell':(type(command) is str), 'stdin':PIPE, 'stdout':PIPE, 'stderr':PIPE, 'universal_newlines':expand_cr}

    # "write to file" : Function for writing some characters to the log and/or output files.
    def wtf(out):
        if logfnm != None: 
            with open(logfnm,'a+') as f: 
                f.write(out)
                f.flush()
        if outfnm != None: 
            with open(outfnm,'w+' if wtf.first else 'a+') as f: 
                f.write(out)
                f.flush()
        wtf.first = False
    wtf.first = True

    # Preserve backwards compatibility; sometimes None gets passed to stdin.
    if stdin == None: stdin = ""

    if print_command:
        logger.info("Executing process: \x1b[92m%-50s\x1b[0m%s%s%s\n" % (' '.join(command) if type(command) is list else command, 
                                                               " Output: %s" % outfnm if outfnm != None else "", 
                                                               " Append: %s" % logfnm if logfnm != None else "", 
                                                               (" Stdin: %s" % stdin.replace('\n','\\n')) if stdin else ""))
        wtf("Executing process: %s%s\n" % (command, (" Stdin: %s" % stdin.replace('\n','\\n')) if stdin else ""))

    cmd_options.update(kwargs)
    p = subprocess.Popen(command, **cmd_options)

    # Write the stdin stream to the process.
    p.stdin.write(stdin)
    p.stdin.close()

    #===============================================================#
    #| Read the output streams from the process.  This is a bit    |#
    #| complicated because programs like GROMACS tend to print out |#
    #| stdout as well as stderr streams, and also carriage returns |#
    #| along with newline characters.                              |#
    #===============================================================#
    # stdout and stderr streams of the process.
    streams = [p.stdout, p.stderr]
    # These are functions that take chunks of lines (read) as inputs.
    def process_out(read):
        if print_to_screen: sys.stdout.write(read)
        if copy_stdout: 
            process_out.stdout += read
            wtf(read)
    process_out.stdout = ""

    def process_err(read):
        if print_to_screen: sys.stderr.write(read)
        process_err.stderr += read
        if copy_stderr: 
            process_out.stdout += read
            wtf(read)
    process_err.stderr = ""
    # This reads the streams one byte at a time, and passes it to the LineChunker
    # which splits it by either newline or carriage return.
    # If the stream has ended, then it is removed from the list.
    with LineChunker(process_out) as out_chunker, LineChunker(process_err) as err_chunker:
        while True:
            to_read, _, _ = select(streams, [], [])

            for fh in to_read:
                if fh is p.stdout:
                    read = fh.read(1)
                    if not read: streams.remove(p.stdout)
                    else: out_chunker.push(read)
                elif fh is p.stderr:
                    read = fh.read(1)
                    if not read: streams.remove(p.stderr)
                    else: err_chunker.push(read)
                else:
                    raise RuntimeError
            if len(streams) == 0: break

    p.wait()

    if p.returncode != 0:
        if process_err.stderr:
            logger.warning("Received an error message:\n")
            logger.warning("\n[====] \x1b[91mError Message\x1b[0m [====]\n")
            logger.warning(process_err.stderr)
            logger.warning("[====] \x1b[91mEnd o'Message\x1b[0m [====]\n")
        if persist:
            logger.info("%s gave a return code of %i (it may have crashed) -- carrying on\n" % (command, p.returncode))
        else:
            # This code (commented out) would not throw an exception, but instead exit with the returncode of the crashed program.
            # sys.stderr.write("\x1b[1;94m%s\x1b[0m gave a return code of %i (\x1b[91mit may have crashed\x1b[0m)\n" % (command, p.returncode))
            # sys.exit(p.returncode)
            raise Exception("\x1b[1;94m%s\x1b[0m gave a return code of %i (\x1b[91mit may have crashed\x1b[0m)\n" % (command, p.returncode))
        
    # Return the output in the form of a list of lines, so we can loop over it using "for line in output".
    Out = process_out.stdout.split('\n')
    if Out[-1] == '':
        Out = Out[:-1]
    return Out

def warn_press_key(warning, timeout=10):
    if type(warning) is str:
        logger.warning(warning + '\n')
    elif type(warning) is list:
        for line in warning:
            logger.warning(line + '\n')
    else:
        logger.warning("You're not supposed to pass me a variable of this type: " + str(type(warning)))
    if sys.stdin.isatty():
        logger.warning("\x1b[1;91mPress Enter or wait %i seconds (I assume no responsibility for what happens after this!)\x1b[0m\n" % timeout)
        try: rlist, wlist, xlist = select([sys.stdin], [], [], timeout)
        except: pass

def warn_once(warning, warnhash = None):
    """ Prints a warning but will only do so once in a given run. """
    if warnhash == None:
        warnhash = warning
    if warnhash in warn_once.already:
        return
    warn_once.already.add(warnhash)
    if type(warning) is str:
        logger.info(warning + '\n')
    elif type(warning) is list:
        for line in warning:
            logger.info(line + '\n')
warn_once.already = set()

#=========================================#
#| Development stuff (not commonly used) |#
#=========================================#
def concurrent_map(func, data):
    """
    Similar to the bultin function map(). But spawn a thread for each argument
    and apply `func` concurrently.

    Note: unlike map(), we cannot take an iterable argument. `data` should be an
    indexable sequence.
    """

    N = len(data)
    result = [None] * N

    # wrapper to dispose the result in the right slot
    def task_wrapper(i):
        result[i] = func(data[i])

    threads = [threading.Thread(target=task_wrapper, args=(i,)) for i in xrange(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return result


def multiopen(arg):
    """
    This function be given any of several variable types
    (single file name, file object, or list of lines, or a list of the above)
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
            logger.info("What did you give this program as input?\n")
            logger.info(str(arg) + '\n')
            exit(1)
    else:
        logger.info("What did you give this program as input?\n")
        logger.info(str(arg) + '\n')
        exit(1)
    return fins
