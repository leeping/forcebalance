"""@package nifty Nifty functions, intended to be imported by any module within ForceBalance.

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

import os
from re import match, sub
import numpy as np
from numpy import array, diag, dot, eye, mat, mean, transpose
from numpy.linalg import norm, svd
import threading
import pickle
import time, datetime
import subprocess
from subprocess import PIPE, STDOUT
from collections import OrderedDict
import IPython as ip

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
def pvec1d(vec1d, precision=1):
    """Printout of a 1-D vector.

    @param[in] vec1d a 1-D vector
    """
    v2a = array(vec1d)
    for i in range(v2a.shape[0]):
        print "%% .%ie" % precision % v2a[i],
    print

def pmat2d(mat2d, precision=1):
    """Printout of a 2-D matrix.

    @param[in] mat2d a 2-D matrix
    """
    m2a = array(mat2d)
    for i in range(m2a.shape[0]):
        for j in range(m2a.shape[1]):
            print "%% .%ie" % precision % m2a[i][j],
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
        print "%s\x1b[%s9%im%s" % (''.join([sym for i in range(3)]), bold and "1;" or "", color, padleft),line,"%s\x1b[0m%s" % (padright, ''.join([sym for i in range(3)]))
    print bar
    return sub(sym,bottom,bar)

def printcool_dictionary(Dict,title="General options",bold=False,color=2,keywidth=25,topwidth=50):
    """See documentation for printcool; this is a nice way to print out keys/values in a dictionary.

    The keys in the dictionary are sorted before printing out.

    @param[in] dict The dictionary to be printed
    @param[in] title The title of the printout
    """
    bar = printcool(title,bold=bold,color=color,minwidth=topwidth)
    def magic_string(str):
        # This cryptic command returns a string with the number of characters specified as a variable. :P
        # Useful for printing nice-looking dictionaries, i guess.
        #print "\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"'))
        return eval("\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"')))
    if isinstance(Dict, OrderedDict): 
        print '\n'.join(["%s %s " % (magic_string(key),str(Dict[key])) for key in Dict if Dict[key] != None])
    else:
        print '\n'.join(["%s %s " % (magic_string(key),str(Dict[key])) for key in sorted([i for i in Dict]) if Dict[key] != None])
    print bar

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
        print "Setting %s to % .1e" % big
        return big

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
    v2u = vec2/norm(vec2)
    return vec1 - v2u*dot(vec1, v2u)

def invert_svd(X,thresh=1e-6):
    
    """ 

    Invert a matrix using singular value decomposition. 
    @param[in] X The matrix to be inverted
    @param[in] thresh The SVD threshold; eigenvalues below this are not inverted but set to zero
    @return Xt The inverted matrix

    """

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

#==============================#
#|    Linear least squares    |#
#==============================#
def get_least_squares(x, y, w = None):
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
    X = mat(x)
    Y = col(y)
    n_x = X.shape[0]
    n_fit = X.shape[1]
    if n_fit >= n_x:
        print "Argh? It seems like this problem is underdetermined!"
    # Build the weight matrix.
    if w != None:
        if len(w) != n_x:
            warn_press_key("The weight array length (%i) must be the same as the number of 'X' data points (%i)!" % len(w), n_x)
        w /= mean(w)
        W = mat(diag(w))
    else:
        W = mat(eye(n_x))
    # Make the Moore-Penrose Pseudoinverse.
    MPPI = invert_svd(X.T * W * X) * X.T * W
    Beta = MPPI * Y
    Hat = X * MPPI
    yfit = flat(Hat * Y)
    # Return three things: the least-squares coefficients, the hat matrix (turns y into yfit), and yfit
    # We could get these all from MPPI, but I might get confused later on, so might as well do it here :P
    return Beta, Hat, yfit, MPPI

#==============================#
#|      XML Pickle stuff      |#
#==============================#
try:
    from lxml import etree
except: 
    print "lxml module import failed (You can't use OpenMM or XML force fields)"
## Pickle uses 'flags' to pickle and unpickle different variable types.
## Here we use the letter 'x' to signify that the variable type is an XML file.
XMLFILE='x'

class Pickler_LP(pickle.Pickler):
    """ A subclass of the python Pickler that implements pickling of _ElementTree types. """
    def __init__(self, file, protocol=None):
        pickle.Pickler.__init__(self, file, protocol)
        ## The element tree is saved as a string.
        def save_etree(self, obj):
            ## Convert the element tree to string.
            String = etree.tostring(obj)
            ## The rest is copied from the Pickler class
            if self.bin:
                print "self.bin is True, not sure what to do with myself"
                raw_input()
            else:
                self.write(XMLFILE + repr(String) + '\n')
            self.memoize(String)
        self.dispatch[etree._ElementTree] = save_etree

class Unpickler_LP(pickle.Unpickler):
    """ A subclass of the python Unpickler that implements unpickling of _ElementTree types. """
    def __init__(self, file):
        pickle.Unpickler.__init__(self, file)
        def load_etree(self):
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
        self.dispatch[XMLFILE] = load_etree

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
    print "Work Queue library import fail (You can't queue up jobs using Work Queue)"

def queue_up(wq, command, input_files, output_files, verbose=True):
    """ 
    Submit a job to the Work Queue.

    @param[in] wq (Work Queue Object) A Work Queue (probably a member of a fitting simulation)
    @param[in] command (string) The command to run on the remote worker.
    @param[in] input_files (list of files) A list of locations of the input files.
    @param[in] output_files (list of files) A list of locations of the output files.
    """

    task = work_queue.Task(command)
    cwd = os.getcwd()
    for f in input_files:
        lf = os.path.join(cwd,f)
        task.specify_input_file(lf,f)
    for f in output_files:
        lf = os.path.join(cwd,f)
        task.specify_output_file(lf,f)
    task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_FCFS)
    task.specify_tag(command)
    if verbose:
        print "Submitting command '%s' to the Work Queue" % command
    wq.submit(task)
    
def queue_up_src_dest(wq, command, input_files, output_files, verbose=True):
    """ 
    Submit a job to the Work Queue.  This function is a bit fancier in that we can explicitly
    specify where the input files come from, and where the output files go to.

    @param[in] wq (Work Queue Object) A Work Queue (probably a member of a fitting simulation)
    @param[in] command (string) The command to run on the remote worker.
    @param[in] input_files (list of 2-tuples) A list of local and
    remote locations of the input files.
    @param[in] output_files (list of 2-tuples) A list of local and
    remote locations of the output files.
    """
    task = work_queue.Task(command)
    for f in input_files:
        print f[0], f[1]
        task.specify_input_file(f[0],f[1])
    for f in output_files:
        print f[0], f[1]
        task.specify_output_file(f[0],f[1])
    task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_FCFS)
    task.specify_tag(command)
    if verbose:
        print "Submitting command '%s' to the Work Queue" % command
    wq.submit(task)

def wq_wait(wq, verbose=False):
    """ This function waits until the work queue is completely empty. """
    printcount = 1
    while not wq.empty():
        if verbose: print '---'
        task = wq.wait(10)
        if task:
            try:
                exectime = task.cmd_execution_time/1000000 # Work Queue 3.6.0
            except:
                exectime = task.computation_time/1000000   # Work Queue <= 3.5.2
            if verbose:
                print 'A job has finished!'
                print 'Job name = ', task.tag, 'command = ', task.command
                print "status = ", task.status, 
                print "return_status = ", task.return_status, 
                print "result = ", task.result, 
                print "host = ", task.host
                print "execution time = ", exectime, 
                print "total_bytes_transferred = ", task.total_bytes_transferred
            if task.result != 0:
                #ip.embed()
                print "Command '%s' failed on host %s (%i seconds), resubmitting" % (task.command, task.host, exectime)
                wq.submit(task)
            else:
                if exectime > 60: # Assume that we're only interested in printing jobs that last longer than a minute.
                    print "Command '%s' finished succesfully on host %s (%i seconds)" % (task.command, task.host, exectime)
                del task
        else:
            printcount += 1
        if verbose:
            print "Workers: %i init, %i ready, %i busy, %i total joined, %i total removed" \
                % (wq.stats.workers_init, wq.stats.workers_ready, wq.stats.workers_busy, wq.stats.total_workers_joined, wq.stats.total_workers_removed)
            print "Tasks: %i running, %i waiting, %i total dispatched, %i total complete" \
                % (wq.stats.tasks_running,wq.stats.tasks_waiting,wq.stats.total_tasks_dispatched,wq.stats.total_tasks_complete)
            print "Data: %i / %i kb sent/received" % (wq.stats.total_bytes_sent/1000, wq.stats.total_bytes_received/1024)
        else:
            print "%s : %i/%i workers busy; %i/%i jobs complete\r" % (datetime.datetime.fromtimestamp(time.mktime(datetime.datetime.now().timetuple())).ctime(),
                                                                      wq.stats.workers_busy, (wq.stats.total_workers_joined - wq.stats.total_workers_removed),
                                                                      wq.stats.total_tasks_complete, wq.stats.total_tasks_dispatched), 
            if printcount % 90 == 89:
                # Print a new line every 15 minutes.
                print

#=====================================#
#| File and process management stuff |#
#=====================================#
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

def LinkFile(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            if os.path.islink(dest): pass
            else: raise Exception("Tried to create symbolic link %s to %s, destination exists but isn't a symbolic link" % (src, dest))
        else:
            os.path.symlink(src, dest)
    else:
        raise Exception("Tried to create symbolic link %s to %s, but source file doesn't exist" % src)

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
        return os.path.split(os.popen('which %s' % fnm).readlines()[0].strip())[0]
    except:
        return ''

def _exec(command, print_to_screen = False, logfnm = None, stdin = None, print_command = True):
    """Runs command line using subprocess, optionally returning stdout"""
    print_to_file = (logfnm != None)
    if print_to_file:
        f = open(logfnm,'a')
    if print_command:
        print "Executing process: \x1b[92m%-50s\x1b[0m%s%s" % (' '.join(command) if type(command) is list else command, 
                                                               " Logfile: %s" % logfnm if logfnm != None else "", 
                                                               " Stdin: %s" % stdin.replace('\n','\\n') if stdin != None else "")
        if print_to_file:
            print >> f, "Executing process: %s%s" % (command, " Stdin: %s" % stdin.replace('\n','\\n') if stdin != None else "")
    if stdin == None:
        p = subprocess.Popen(command, shell=(type(command) is str), stdout = PIPE, stderr = PIPE)
        if print_to_screen:
            Output = []
            Error = []
            while True:
                line = p.stdout.readline()
                try:
                    Error.append(p.stderr.readline())
                except: pass
                if not line:
                    break
                print line,
            Output.append(line)
            print Error
        else:
            Output, Error = p.communicate()
    else:
        p = subprocess.Popen(command, shell=(type(command) is str), stdin = PIPE, stdout = PIPE, stderr = PIPE)
        Output, Error = p.communicate(stdin)
    if logfnm != None:
        f.write(Output)
        f.close()
    if p.returncode != 0:
        print "Received an error message:"
        print Error
        warn_press_key("%s gave a return code of %i (it may have crashed)" % (command, p.returncode))
    return Output

def warn_press_key(warning):
    if type(warning) is str:
        print warning
    elif type(warning) is list:
        for line in warning:
            print line
    else:
        print "You're not supposed to pass me a variable of this type:", type(warning)
    print "\x1b[1;91mPress Enter (I assume no responsibility for what happens after this!)\x1b[0m"
    raw_input()

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
            print "What did you give this program as input?"
            print arg
            exit(1)
    else:
        print "What did you give this program as input?"
        print arg
        exit(1)
    return fins
