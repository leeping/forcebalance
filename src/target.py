""" Target base class from which all ForceBalance fitting targets are derived. """

import abc
import os
import subprocess
import shutil
import numpy as np
import time
from collections import OrderedDict
import tarfile
import forcebalance
from forcebalance.nifty import row, col, printcool_dictionary, link_dir_contents, createWorkQueue, getWorkQueue, wq_wait1, getWQIds, wopen, warn_press_key, _exec, lp_load
from forcebalance.finite_difference import fdwrap_G, fdwrap_H, f1d2p, f12d3p, in_fd
from forcebalance.optimizer import Counter
from forcebalance.output import getLogger
logger = getLogger(__name__)

class Target(forcebalance.BaseClass):
    
    """
    Base class for all fitting targets.
    
    In ForceBalance a Target is defined as a set of reference data
    plus a corresponding method to simulate that data using the force field.
    
    The 'computable quantities' may include energies and forces where the
    reference values come from QM calculations (energy and force matching),
    energies from an EDA analysis (Maybe in the future, FDA?), molecular
    properties (like polarizability, refractive indices, multipole moments
    or vibrational frequencies), relative entropies, and bulk properties.
    Single-molecule or bulk properties can even come from the experiment!
    
    The central idea in ForceBalance is that each quantity makes a
    contribution to the overall objective function.  So we can build force
    fields that fit several quantities at once, rather than putting all of
    our chips behind energy and force matching.  In the future
    ForceBalance may even include multiobjective optimization into the
    optimizer.
    
    The optimization is done by way of minimizing an 'objective
    function', which is comprised of squared differences between the
    computed and reference values.  These differences are not computed
    in this file, but rather in subclasses that use Target
    as a base class.  Thus, the contents of Target itself
    are meant to be as general as possible, because the pertinent
    variables apply to all types of fitting targets.

    An important node: Target requires that all subclasses
    have a method get(self,mvals,AGrad=False,AHess=False)
    that does the following:
    
    Inputs: 
    mvals        = The parameter vector, which modifies the force field
    (Note to self: We include mvals with each Target because we can create
    copies of the force field and do finite difference derivatives)
    AGrad, AHess = Boolean switches for computing analytic gradients and Hessians

    Outputs:
    Answer       = {'X': Number, 'G': array(NP), 'H': array((NP,NP)) }
    'X'          = The objective function itself
    'G'          = The gradient, elements not computed analytically are zero
    'H'          = The Hessian, elements not computed analytically are zero

    This is the only global requirement of a Target.
    Obviously 'get' itself is not defined here, because its
    calculation will depend entirely on specifically which target
    we wish to use.  However, this should give us a unified framework
    which will faciliate rapid implementation of Targets.

    Future work:
    Robert suggested that I could enable automatic detection of which
    parameters need to be computed by finite difference.  Not a bad idea. :)

    """

    __metaclass__ = abc.ABCMeta
    
    def __init__(self,options,tgt_opts,forcefield):
        """
        All options here are intended to be usable by every
        conceivable type of target (in other words, only
        add content here if it's widely applicable.)

        If we want to add attributes that are more specific
        (i.e. a set of reference forces for force matching), they
        are added in the subclass AbInitio that inherits from
        Target.

        """
        super(Target, self).__init__(options)
        #======================================#
        # Options that are given by the parser #
        #======================================#
        ## Root directory of the whole project
        self.set_option(options, 'root')
        ## Name of the target
        self.set_option(tgt_opts, 'name')
        if self.name in ["forcefield-remote"]:
            logger.error("forcefield-remote is not an allowed target name (reserved)")
            raise RuntimeError
        ## Type of target
        self.set_option(tgt_opts, 'type')
        ## Relative weight of the target
        self.set_option(tgt_opts, 'weight')
        ## Switch for finite difference gradients
        self.set_option(tgt_opts, 'fdgrad')
        ## Switch for finite difference Hessians
        self.set_option(tgt_opts, 'fdhess')
        ## Switch for FD gradients + Hessian diagonals
        self.set_option(tgt_opts, 'fdhessdiag')
        ## How many seconds to sleep (if any)
        self.set_option(tgt_opts, 'sleepy')
        ## Parameter types that trigger FD gradient elements
        self.set_option(None, None, 'fd1_pids', [i.upper() for i in tgt_opts['fd_ptypes']], default = [])
        self.set_option(None, None, 'fd2_pids', [i.upper() for i in tgt_opts['fd_ptypes']], default = [])
        ## Parameter types that trigger FD Hessian elements
        ## Finite difference step size
        self.set_option(options, 'finite_difference_h', 'h')
        ## Whether to make backup files
        self.set_option(options, 'backup')
        ## Directory to read data from.
        self.set_option(tgt_opts, 'read', 'rd')
        if self.rd is not None: self.rd = self.rd.strip("/")
        ## Iteration where we turn on zero-gradient skipping.
        self.set_option(options, 'zerograd')
        ## Gradient norm below which we skip.
        self.set_option(tgt_opts, 'epsgrad')
        ## Dictionary of whether to call the derivatives.
        self.pgrad = range(forcefield.np)
        self.OptionDict['pgrad'] = self.pgrad

        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Relative directory of target
        if os.path.exists('targets'):
            tgtdir = 'targets'
        elif os.path.exists('simulations'):
            tgtdir = 'simulations'
        elif os.path.exists('targets.tar.bz2'):
            logger.info("Extracting targets folder from archive.\n")
            _exec("tar xvjf targets.tar.bz2")
            tgtdir = 'targets'
        elif os.path.exists('targets.tar.gz'):
            logger.info("Extracting targets folder from archive.\n")
            _exec("tar xvzf targets.tar.gz")
            tgtdir = 'targets'
        else:
            logger.error('\x1b[91mThe targets directory is missing!\x1b[0m\nDid you finish setting up the target data?\nPlace the data in a directory called "targets" or "simulations"\n')
            raise RuntimeError
        self.set_option(None, None, 'tgtdir', os.path.join(tgtdir,self.name))
        ## Temporary (working) directory; it is temp/(target_name)
        ## Used for storing temporary variables that don't change through the course of the optimization
        if 'input_file' in options and options['input_file'] is not None:
            self.tempbase    = os.path.splitext(options['input_file'])[0]+'.tmp'
        else:
            self.tempbase    = "temp"
        self.tempdir     = os.path.join(self.tempbase, self.name)
        ## self.tempdir     = os.path.join('temp',self.name)
        ## The directory in which the simulation is running - this can be updated.
        self.rundir      = self.tempdir
        ## Need the forcefield (here for now)
        self.FF          = forcefield
        ## Counts how often the objective function was computed
        self.xct         = 0
        ## Counts how often the gradient was computed
        self.gct         = 0
        ## Counts how often the Hessian was computed
        self.hct         = 0
        ## Whether to read indicate.log from file when restarting an aborted run.
        self.read_indicate    = True
        ## Whether to write indicate.log at every iteration (true for all but remote.)
        self.write_indicate   = True
        ## Whether to read objective.p from file when restarting an aborted run.
        self.read_objective       = True
        ## Whether to write objective.p at every iteration (true for all but remote.)
        self.write_objective      = True
        ## Create a new temp directory.
        if not options['continue']: 
            self.refresh_temp_directory()
        else:
            if not os.path.exists(os.path.join(self.root,self.tempdir)):
                os.makedirs(os.path.join(self.root,self.tempdir))
        ## This flag specifies whether the target has been evaluated yet.
        self.evaluated = False
        ## This flag specifies whether the previous optimization step was good.
        self.goodstep = False

    def get_X(self,mvals=None,customdir=None):
        """Computes the objective function contribution without any parametric derivatives"""
        Ans = self.meta_get(mvals,0,0,customdir=customdir)
        self.xct += 1
        if Ans['X'] != Ans['X']:
            return {'X':1e10, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np,self.FF.np))}
        return Ans

    def read_0grads(self):
        
        """ Read a file from the target directory containing names of
        parameters that don't contribute to the gradient. 

        *Note* that we are checking the derivatives of the objective
        function, and not the derivatives of the quantities that go
        into building the objective function.  However, it is the
        quantities that we actually differentiate.  Since there is a
        simple chain rule relationship, the parameters that do/don't
        contribute to the objective function/quantities are the same.

        However, property gradients do contribute to objective
        function Hessian elements, so we cannot use the same mechanism
        for excluding the calculation of property Hessians.  This is
        mostly fine since we rarely if ever calculate an explicit
        property Hessian. """
        
        zero_prm = os.path.join(self.root, self.tgtdir, 'zerograd.txt')
        # If the 'zero parameters' text file exists, then we load
        # the parameter names from the file for exclusion.
        pgrad0 = self.pgrad[:]
        self.pgrad = range(self.FF.np)
        if os.path.exists(zero_prm):
            for ln, line in enumerate(open(zero_prm).readlines()):
                pid = line.strip()
                # If a parameter name exists in the map, then
                # the derivative is switched off for this target.
                if pid in self.FF.map and self.FF.map[pid] in self.pgrad:
                    self.pgrad.remove(self.FF.map[pid])
        for i in pgrad0:
            if i not in self.pgrad:
                pass
                # logger.info("Parameter %s was deactivated in %s\n" % (i, self.name))
        for i in self.pgrad:
            if i not in pgrad0:
                logger.info("Parameter %s was reactivated in %s\n" % (i, self.name))
        # Set pgrad in the OptionDict so remote targets may use it.
        self.OptionDict['pgrad'] = self.pgrad

    def write_0grads(self, Ans):

        """ Write a file to the target directory containing names of
        parameters that don't contribute to the gradient. """

        zero_prm = os.path.join(self.root, self.tgtdir, 'zerograd.txt')
        if os.path.exists(zero_prm):
            zero_pids = [i.strip() for i in open(zero_prm).readlines()]
        else:
            zero_pids = []
        for i in range(self.FF.np):
            # Check whether this parameter number has a nonzero gradient.
            if abs(Ans['G'][i]) <= self.epsgrad:
                # Write parameter names corresponding to this parameter number.
                for pid in self.FF.map:
                    if self.FF.map[pid] == i and pid not in zero_pids:
                        logger.info("Adding %s to zero_pids in %s\n" % (i, self.name))
                        zero_pids.append(pid)
            # If a parameter number has a nonzero gradient, then the parameter
            # names associated with this parameter number are removed from the list.
            # (Not sure if this will ever happen.)
            if abs(Ans['G'][i]) > self.epsgrad:
                for pid in self.FF.map:
                    if self.FF.map[pid] == i and pid in zero_pids:
                        logger.info("Removing %s from zero_pids in %s\n" % (i, self.name))
                        zero_pids.remove(pid)
        if len(zero_pids) > 0:
            fout = open(zero_prm, 'w')
            for pid in zero_pids:
                print >> fout, pid
            fout.close()

    def get_G(self,mvals=None,customdir=None):
        """Computes the objective function contribution and its gradient.

        First the low-level 'get' method is called with the analytic gradient
        switch turned on.  Then we loop through the fd1_pids and compute
        the corresponding elements of the gradient by finite difference,
        if the 'fdgrad' switch is turned on.  Alternately we can compute
        the gradient elements and diagonal Hessian elements at the same time
        using central difference if 'fdhessdiag' is turned on.

        In this function we also record which parameters cause a
        nonzero change in the objective function contribution.
        Parameters which do not change the objective function will
        not be differentiated in subsequent calculations.  This is
        recorded in a text file in the targets directory.

        """
        Ans = self.meta_get(mvals,1,0,customdir=customdir)
        for i in self.pgrad:
            if any([j in self.FF.plist[i] for j in self.fd1_pids]) or 'ALL' in self.fd1_pids:
                if self.fdhessdiag:
                    Ans['G'][i], Ans['H'][i,i] = f12d3p(fdwrap_G(self,mvals,i),self.h,f0 = Ans['X'])
                elif self.fdgrad:
                    Ans['G'][i] = f1d2p(fdwrap_G(self,mvals,i),self.h,f0 = Ans['X'])
        self.gct += 1
        if Counter() == self.zerograd and self.zerograd >= 0: 
            self.write_0grads(Ans)
        return Ans

    def get_H(self,mvals=None,customdir=None):
        """Computes the objective function contribution and its gradient / Hessian.

        First the low-level 'get' method is called with the analytic gradient
        and Hessian both turned on.  Then we loop through the fd1_pids and compute
        the corresponding elements of the gradient by finite difference,
        if the 'fdgrad' switch is turned on.

        This is followed by looping through the fd2_pids and computing the corresponding
        Hessian elements by finite difference.  Forward finite difference is used
        throughout for the sake of speed.
        """
        Ans = self.meta_get(mvals,1,1,customdir=customdir)
        if self.fdhess:
            for i in self.pgrad:
                if any([j in self.FF.plist[i] for j in self.fd1_pids]) or 'ALL' in self.fd1_pids:
                    Ans['G'][i] = f1d2p(fdwrap_G(self,mvals,i),self.h,f0 = Ans['X'])
            for i in self.pgrad:
                if any([j in self.FF.plist[i] for j in self.fd2_pids]) or 'ALL' in self.fd2_pids:
                    FDSlice = f1d2p(fdwrap_H(self,mvals,i),self.h,f0 = Ans['G'])
                    Ans['H'][i,:] = FDSlice
                    Ans['H'][:,i] = FDSlice
        elif self.fdhessdiag:
            for i in self.pgrad:
                if any([j in self.FF.plist[i] for j in self.fd2_pids]) or 'ALL' in self.fd2_pids:
                    Ans['G'][i], Ans['H'][i,i] = f12d3p(fdwrap_G(self,mvals,i),self.h, f0 = Ans['X'])
        if Counter() == self.zerograd and self.zerograd >= 0: 
            self.write_0grads(Ans)
        self.hct += 1
        return Ans
    
    def link_from_tempdir(self,absdestdir):
        link_dir_contents(os.path.join(self.root,self.tempdir), absdestdir)

    def refresh_temp_directory(self):
        """ Back up the temporary directory if desired, delete it
        and then create a new one."""
        cwd = os.getcwd()
        abstempdir = os.path.join(self.root,self.tempdir)
        if self.backup:
            bakdir = os.path.join(os.path.splitext(self.tempbase)[0]+'.bak')
            if not os.path.exists(bakdir):
                os.makedirs(bakdir)
            if os.path.exists(abstempdir):
                os.chdir(self.tempbase)
                FileCount = 0
                while True:
                    CandFile = os.path.join(self.root,bakdir,"%s_%i.tar.bz2" % (self.name,FileCount))
                    if os.path.exists(CandFile):
                        FileCount += 1
                    else:
                        # I could use the tarfile module here
                        logger.info("Backing up: " + self.tempdir + ' to: ' + "%s/%s_%i.tar.bz2\n" % (bakdir,self.name,FileCount))
                        subprocess.call(["tar","cjf",CandFile,self.name])
                        shutil.rmtree(self.name)
                        break
                os.chdir(cwd)
        # Delete the temporary directory
        shutil.rmtree(abstempdir,ignore_errors=True)
        # Create a new temporary directory from scratch
        os.makedirs(abstempdir)

    @abc.abstractmethod
    def get(self,mvals,AGrad=False,AHess=False):

        """ 
        
        Every target must be able to return a contribution
        to the objective function - however, this must be implemented
        in the specific subclass.  See abinitio for an
        example.

        """
        
        logger.error('The get method is not implemented in the Target base class\n')
        raise NotImplementedError

    def check_files(self, there):

        """ Check this directory for the presence of readable files when the 'read' option is set. """

        there = os.path.abspath(there)
        if all([any([i == j for j in os.listdir(there)]) for i in ["objective.p", "indicate.log"]]):
            return True
        return False

    def read(self,mvals,AGrad=False,AHess=False):

        """ 

        Read data from disk for the initial optimization step if the
        user has provided the directory to the "read" option.  

        """
        mvals1 = np.loadtxt('mvals.txt')

        if len(mvals) > 0 and (np.max(np.abs(mvals1 - mvals)) > 1e-3):
            warn_press_key("mvals from mvals.txt does not match up with get! (Are you reading data from a previous run?)\nmvals(call)=%s mvals(disk)=%s" % (mvals, mvals1))
        
        return lp_load('objective.p')

    def absrd(self, inum=None):

        """ 
        Supply the correct directory specified by user's "read" option.
        """
        
        if self.evaluated:
            logger.error("Tried to read from disk, but not allowed because this target is evaluated already\n")
            raise RuntimeError
        if self.rd is None:
            logger.error("The directory for reading is not set\n")
            raise RuntimeError

        # Current directory. Move back into here after reading data.
        here = os.getcwd()
        # Absolute path for the directory to read data from.
        if os.path.isabs(self.rd):
            abs_rd = self.rd
        else:
            abs_rd = os.path.join(self.root, self.rd)
        # Check for directory existence.
        if not os.path.exists(abs_rd):
            logger.error("Provided path %s does not exist\n" % self.rd)
            raise RuntimeError
        # Figure out which directory to go into.
        s = os.path.split(self.rd)
        have_data = 0
        if s[-1].startswith('iter_'):
            # Case 1: User has provided a specific directory to read from.
            there = abs_rd
            if not self.check_files(there):
                logger.error("Provided path %s does not contain remote target output\n" % self.rd)
                raise RuntimeError
            have_data = 1
        elif s[-1] == self.name:
            # Case 2: User has provided the target name.
            iterints = [int(d.replace('iter_','')) for d in os.listdir(abs_rd) if os.path.isdir(os.path.join(abs_rd, d))]
            for i in sorted(iterints)[::-1]:
                there = os.path.join(abs_rd, 'iter_%04i' % i)
                if self.check_files(there):
                    have_data = 1
                    break
        else:
            # Case 3: User has provided something else (must contain the target name in the next directory down.)
            if not os.path.exists(os.path.join(abs_rd, self.name)):
                logger.error("Target directory %s does not exist in %s\n" % (self.name, self.rd))
                raise RuntimeError
            iterints = [int(d.replace('iter_','')) for d in os.listdir(os.path.join(abs_rd, self.name)) if os.path.isdir(os.path.join(abs_rd, self.name, d))]
            for i in sorted(iterints)[::-1]:
                there = os.path.join(abs_rd, self.name, 'iter_%04i' % i)
                if self.check_files(there):
                    have_data = 1
                    break
        if not have_data:
            logger.error("Did not find data to read in %s\n" % self.rd)
            raise RuntimeError

        if inum is not None:
            there = os.path.join(os.path.split(there)[0],'iter_%04i' % inum)
        return there

    def maxrd(self):

        """ Supply the latest existing temp-directory containing valid data. """
        
        abs_rd = os.path.join(self.root, self.tempdir)

        iterints = [int(d.replace('iter_','')) for d in os.listdir(abs_rd) if os.path.isdir(os.path.join(abs_rd, d))]
        for i in sorted(iterints)[::-1]:
            there = os.path.join(abs_rd, 'iter_%04i' % i)
            if self.check_files(there):
                return i

        return -1

    def maxid(self):

        """ Supply the latest existing temp-directory. """
        
        abs_rd = os.path.join(self.root, self.tempdir)

        iterints = [int(d.replace('iter_','')) for d in os.listdir(abs_rd) if os.path.isdir(os.path.join(abs_rd, d))]
        return sorted(iterints)[-1]

    def meta_indicate(self, customdir=None):

        """ 

        Wrap around the indicate function, so it can print to screen and
        also to a file.  If reading from checkpoint file, don't call
        the indicate() function, instead just print the file contents
        to the screen.
        
        """
        # Using the module level logger
        logger = getLogger(__name__)
        # Note that reading information is not supported for custom folders (e.g. microiterations during search)
        if self.rd is not None and (not self.evaluated) and self.read_indicate and customdir is None:
            # Move into the directory for reading data, 
            cwd = os.getcwd()
            os.chdir(self.absrd())
            logger.info(open('indicate.log').read())
            os.chdir(cwd)
        else:
            if self.write_indicate:
                # Go into the directory where the job is running
                cwd = os.getcwd()
                os.chdir(os.path.join(self.root, self.rundir))
                # If indicate.log already exists then we've made some kind of mistake.
                if os.path.exists('indicate.log'):
                    logger.error('indicate.log should not exist yet in this directory: %s\n' % os.getcwd())
                    raise RuntimeError
                # Add a handler for printing to screen and file
                logger = getLogger("forcebalance")
                hdlr = forcebalance.output.RawFileHandler('indicate.log')
                logger.addHandler(hdlr)
            # Execute the indicate function
            self.indicate()
            if self.write_indicate:
                # Remove the handler (return to normal printout)
                logger.removeHandler(hdlr)
                # Return to the module level logger
                logger = getLogger(__name__)
                # The module level logger now prints the indicator
                logger.info(open('indicate.log').read())
                # Go back to the directory where we were
                os.chdir(cwd)
        
    def meta_get(self, mvals, AGrad=False, AHess=False, customdir=None):
        """ 
        Wrapper around the get function.  
        Create the directory for the target, and then calls 'get'.
        If we are reading existing data, go into the appropriate read directory and call read() instead.
        The 'get' method should not worry about the directory that it's running in.
        
        """
        ## Directory of the current iteration; if not None, then the simulation runs under
        ## temp/target_name/iteration_number
        ## The 'customdir' is customizable and can go below anything
        cwd = os.getcwd()
        
        absgetdir = os.path.join(self.root,self.tempdir)
        if Counter() is not None:
            # Not expecting more than ten thousand iterations
            if Counter() > 10000:
                logger.error('Cannot handle more than 10000 iterations due to current directory structure.  Consider revising code.\n')
                raise RuntimeError
            iterdir = "iter_%04i" % Counter()
            absgetdir = os.path.join(absgetdir,iterdir)
        if customdir is not None:
            absgetdir = os.path.join(absgetdir,customdir)

        if not os.path.exists(absgetdir):
            os.makedirs(absgetdir)
        os.chdir(absgetdir)
        self.link_from_tempdir(absgetdir)
        self.rundir = absgetdir.replace(self.root+'/','')
        ## Read existing information from disk (i.e. when recovering an aborted run)
        # Note that reading information is not supported for custom folders (e.g. microiterations during search)
        if self.rd is not None and (not self.evaluated) and self.read_objective and customdir is None:
            os.chdir(self.absrd())
            logger.info("Reading objective function information from %s\n" % os.getcwd())
            Answer = self.read(mvals, AGrad, AHess)
            os.chdir(absgetdir)
        else:
            ## Evaluate the objective function.
            Answer = self.get(mvals, AGrad, AHess)
            if self.write_objective:
                forcebalance.nifty.lp_dump(Answer, 'objective.p')

        ## Save the force field files to this directory, so that it
        ## reflects the objective function and properties that were
        ## printed out.
        if not in_fd(): 
            self.FF.make(mvals)

        os.chdir(cwd)
        
        return Answer

    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        return

    def stage(self, mvals, AGrad=False, AHess=False, customdir=None, firstIteration=False):
        """ 

        Stages the directory for the target, and then launches Work Queue processes if any.
        The 'get' method should not worry about the directory that it's running in.
        
        """
        if self.sleepy > 0:
            logger.info("Sleeping for %i seconds as directed...\n" % self.sleepy)
            time.sleep(self.sleepy)
        ## Directory of the current iteration; if not None, then the simulation runs under
        ## temp/target_name/iteration_number
        ## The 'customdir' is customizable and can go below anything
        cwd = os.getcwd()
        
        absgetdir = os.path.join(self.root,self.tempdir)
        if Counter() is not None:
            ## Not expecting more than ten thousand iterations
            iterdir = "iter_%04i" % Counter()
            absgetdir = os.path.join(absgetdir,iterdir)
        if customdir is not None:
            absgetdir = os.path.join(absgetdir,customdir)
        ## Go into the directory where get() will be executed.
        if not os.path.exists(absgetdir):
            os.makedirs(absgetdir)
        os.chdir(absgetdir)
        self.link_from_tempdir(absgetdir)
        ## Write mathematical parameters to file; will be used to checkpoint calculation.
        if not in_fd():
            np.savetxt('mvals.txt', mvals)
        ## Read in file that specifies which derivatives may be skipped.
        if Counter() >= self.zerograd and self.zerograd >= 0: 
            self.read_0grads()
        self.rundir = absgetdir.replace(self.root+'/','')
        ## Submit jobs to the Work Queue.
        if self.rd is None or (not firstIteration): 
            self.submit_jobs(mvals, AGrad, AHess)
        elif customdir is not None:
            # Allows us to submit micro-iteration jobs for remote targets
            self.submit_jobs(mvals, AGrad, AHess)
        os.chdir(cwd)
        
        return

    def wq_complete(self):
        """ This method determines whether the Work Queue tasks for the current target have completed. """
        wq = getWorkQueue()
        WQIds = getWQIds()
        if wq is None:
            return True
        elif wq.empty():
            WQIds[self.name] = []
            return True
        elif len(WQIds[self.name]) == 0:
            return True
        else:
            wq_wait1(wq, wait_time=30)
            if len(WQIds[self.name]) == 0:
                return True
            else:
                return False

    def printcool_table(self, data=OrderedDict([]), headings=[], banner=None, footnote=None, color=0):
        """ Print target information in an organized table format.  Implemented 6/30 because
        multiple targets are already printing out tabulated information in very similar ways.
        This method is a simple wrapper around printcool_dictionary.  

        The input should be something like:

        @param data Column contents in the form of an OrderedDict, with string keys and list vals.
        The key is printed in the leftmost column and the vals are printed in the other columns.
        If non-strings are passed, they will be converted to strings (not recommended).
        
        @param headings Column headings in the form of a list.  It must be equal to the number to the list length
        for each of the "vals" in OrderedDict, plus one.  Use "\n" characters to specify long
        column names that may take up more than one line.

        @param banner Optional heading line, which will be printed at the top in the title.
        @param footnote Optional footnote line, which will be printed at the bottom.
        
        """
        tline="Target: %s  Type: %s  Objective = %.5e" % (self.name, self.__class__.__name__, self.objective)
        nc = len(headings)
        if banner is not None:
            tlines = [banner, tline]
        else:
            tlines = [tline]
        # Sanity check.
        for val in data.values():
            if (len(val)+1) != nc:
                logger.error('There are %i column headings, so the values in the data dictionary must be lists of length %i (currently %i)\n' % (nc, nc-1, len(val)))
                raise RuntimeError
        cwidths = [0 for i in range(nc)]
        # Figure out maximum column width.
        # First look at all of the column headings...
        crows = []
        for cnum, cname in enumerate(headings):
            crows.append(len(cname.split('\n')))
            for l in cname.split('\n'):
                cwidths[cnum] = max(cwidths[cnum], len(l))
        # Then look at the row names to stretch out the first column width...
        for k in data.keys():
            cwidths[0] = max(cwidths[0], len(str(k)))
        # Then look at the data values to stretch out the other column widths.
        for v in data.values():
            for n, f in enumerate(v):
                cwidths[n+1] = max(cwidths[n+1], len(str(f)))
        for i in range(1, len(cwidths)):
            cwidths[i] += 2
        if cwidths[0] < 15:
            cwidths[0] = 15
        cblocks = [['' for i in range(max(crows) - len(cname.split('\n')))] + cname.split('\n') for cnum, cname in enumerate(headings)]
        # The formatting line consisting of variable column widths
        fline = ' '.join("%%%s%is" % (("-" if i==0 else ""), j) for i, j in enumerate(cwidths))
        vline = ' '.join(["%%%is" % j for i, j in enumerate(cwidths) if i > 0])
        clines = [fline % (tuple(cblocks[j][i] for j in range(nc))) for i in range(max(crows))]
        tlines += clines
        PrintDict = OrderedDict([(key, vline % (tuple(val))) for key, val in data.items()])
        if len(clines[0]) > len(tlines[0]):
            centers = [0, 1]
        else:
            centers = [0]
        printcool_dictionary(PrintDict, title='\n'.join(tlines), keywidth=cwidths[0], center=[i in centers for i in range(len(tlines))], leftpad=4, color=color)

    def serialize_ff(self, mvals, outside=None):
        """ 
        This code writes a force field pickle file to an folder in
        "job.tmp/dnm/forcebalance.p", because it takes
        time to compress and most targets can simply reuse this file.
        
        Inputs:
        mvals = Mathematical parameter values
        outside = Write this file outside the targets directory
        """
        cwd = os.getcwd()
        if outside is not None:
            self.ffpd = cwd.replace(os.path.join(self.root, self.tempdir), os.path.join(self.root, self.tempbase, outside))
        else:
            self.ffpd = os.path.abspath(os.path.join(self.root, self.rundir))
        if not os.path.exists(self.ffpd): os.makedirs(self.ffpd)
        os.chdir(self.ffpd)
        makeffp = False
        if (os.path.exists("mvals.txt") and os.path.exists("forcefield.p")):
            mvalsf = np.loadtxt("mvals.txt")
            if len(mvalsf) > 0 and np.max(np.abs(mvals - mvalsf)) != 0.0:
                makeffp = True
        else:
            makeffp = True
        if makeffp:
            # logger.info("Writing force field to: %s\n" % self.ffpd)
            self.FF.make(mvals)
            np.savetxt("mvals.txt", mvals)
            forcebalance.nifty.lp_dump((self.FF, mvals), 'forcefield.p')
        os.chdir(cwd)
        forcebalance.nifty.LinkFile(os.path.join(self.ffpd, 'forcefield.p'), 'forcefield.p')
               
class RemoteTarget(Target):
    def __init__(self,options,tgt_opts,forcefield):
        super(RemoteTarget, self).__init__(options,tgt_opts,forcefield)
        
        self.r_options = options.copy()
        self.r_options["type"]="single"
        self.set_option(tgt_opts, "remote_prefix", "rpfx")
        self.set_option(tgt_opts, "remote_backup", "rbak")
        
        self.r_tgt_opts = tgt_opts.copy()
        self.r_tgt_opts["remote"]=False
        
        tar = tarfile.open(name="%s/target.tar.bz2" % (self.tempdir), mode='w:bz2', dereference=True)
        tar.add("%s/targets/%s" % (self.root, self.name), arcname = "targets/%s" % self.name)
        tar.close()
        
        self.remote_indicate = ""

        if options['wq_port'] == 0:
            logger.error("Please set the Work Queue port to use Remote Targets.\n")
            raise RuntimeError

        # Remote target will read objective.p and indicate.log at the same time,
        # and it uses a different mechanism because it does this at every iteration (not just the 0th).
        self.read_indicate = False
        self.write_indicate = False
        self.write_objective = False

    def submit_jobs(self, mvals, AGrad=False, AHess=False):

        id_string = "%s_iter%04i" % (self.name, Counter())

        self.serialize_ff(mvals, outside="forcefield-remote")
        forcebalance.nifty.lp_dump((AGrad, AHess, id_string, self.r_options, self.r_tgt_opts, self.pgrad),'options.p')
        
        # Link in the rpfx script.
        if len(self.rpfx) > 0:
            forcebalance.nifty.LinkFile(os.path.join(os.path.split(__file__)[0],"data",self.rpfx),self.rpfx)
        forcebalance.nifty.LinkFile(os.path.join(os.path.split(__file__)[0],"data","rtarget.py"),"rtarget.py")
        forcebalance.nifty.LinkFile(os.path.join(self.root, self.tempdir, "target.tar.bz2"),"target.tar.bz2")
        
        wq = getWorkQueue()
        
        # logger.info("Sending target '%s' to work queue for remote evaluation\n" % self.name)
        # input:
        #   forcefield.p: pickled force field
        #   options.p: pickled mvals, options
        #   rtarget.py: remote target evaluation script
        #   target.tar.bz2: tarred target
        # output:
        #   objective.p: pickled objective function dictionary
        #   indicate.log: results of target.indicate() written to file
        # if len(self.rpfx) > 0 and self.rpfx not in ['rungmx.sh', 'runcuda.sh']:
        #     logger.error('Unsupported prefix script for launching remote target')
        #     raise RuntimeError
        forcebalance.nifty.queue_up(wq, "%spython rtarget.py > rtarget.out 2>&1" % (("sh %s%s " % (self.rpfx, " -b" if self.rbak else "")) 
                                                                                    if len(self.rpfx) > 0 else ""),
                                    ["forcefield.p", "options.p", "rtarget.py", "target.tar.bz2"] + ([self.rpfx] if len(self.rpfx) > 0 else []),
                                    ['objective.p', 'indicate.log', 'rtarget.out'],
                                    tgt=self, tag=self.name, verbose=False)

    def read(self,mvals,AGrad=False,AHess=False):
        return self.get(mvals, AGrad, AHess)

    def get(self,mvals,AGrad=False,AHess=False):
        with open('indicate.log', 'r') as f:
            self.remote_indicate = f.read()
        return lp_load('objective.p')
        
    def indicate(self):
        logger.info(self.remote_indicate)


