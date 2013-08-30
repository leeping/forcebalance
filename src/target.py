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
from forcebalance.nifty import row,col,printcool_dictionary, link_dir_contents, createWorkQueue, getWorkQueue, wq_wait1, getWQIds
from forcebalance.finite_difference import fdwrap_G, fdwrap_H, f1d2p, f12d3p
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
    Answer       = {'X': Number, 'G': numpy.array(np), 'H': numpy.array((np,np)) }
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
                                                                 
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Relative directory of target
        if os.path.exists('targets'):
            tgtdir = 'targets'
        elif os.path.exists('simulations'):
            tgtdir = 'simulations'
        else:
            raise Exception('\x1b[91mThe targets directory is missing!\x1b[0m\nDid you finish setting up the target data?\nPlace the data in a directory called "targets" or "simulations"')
        self.set_option(None, None, 'tgtdir', os.path.join(tgtdir,self.name))
        ## Temporary (working) directory; it is temp/(target_name)
        ## Used for storing temporary variables that don't change through the course of the optimization
        self.tempdir     = os.path.join('temp',self.name)
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
        
        #======================================#
        #          UNDER DEVELOPMENT           #
        #======================================#
        # Create a new temp directory.
        self.refresh_temp_directory()

    def get_X(self,mvals=None):
        """Computes the objective function contribution without any parametric derivatives"""
        Ans = self.sget(mvals,0,0)
        self.xct += 1
        if Ans['X'] != Ans['X']:
            return {'X':1e10, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np,self.FF.np))}
        return Ans

    def get_G(self,mvals=None):
        """Computes the objective function contribution and its gradient.

        First the low-level 'get' method is called with the analytic gradient
        switch turned on.  Then we loop through the fd1_pids and compute
        the corresponding elements of the gradient by finite difference,
        if the 'fdgrad' switch is turned on.  Alternately we can compute
        the gradient elements and diagonal Hessian elements at the same time
        using central difference if 'fdhessdiag' is turned on.
        """
        Ans = self.sget(mvals,1,0)
        for i in range(self.FF.np):
            if any([j in self.FF.plist[i] for j in self.fd1_pids]) or 'ALL' in self.fd1_pids:
                if self.fdhessdiag:
                    Ans['G'][i], Ans['H'][i,i] = f12d3p(fdwrap_G(self,mvals,i),self.h,f0 = Ans['X'])
                elif self.fdgrad:
                    Ans['G'][i] = f1d2p(fdwrap_G(self,mvals,i),self.h,f0 = Ans['X'])
        self.gct += 1
        return Ans

    def get_H(self,mvals=None):
        """Computes the objective function contribution and its gradient / Hessian.

        First the low-level 'get' method is called with the analytic gradient
        and Hessian both turned on.  Then we loop through the fd1_pids and compute
        the corresponding elements of the gradient by finite difference,
        if the 'fdgrad' switch is turned on.

        This is followed by looping through the fd2_pids and computing the corresponding
        Hessian elements by finite difference.  Forward finite difference is used
        throughout for the sake of speed.
        """
        Ans = self.sget(mvals,1,1)
        if self.fdhess:
            for i in range(self.FF.np):
                if any([j in self.FF.plist[i] for j in self.fd1_pids]) or 'ALL' in self.fd1_pids:
                    Ans['G'][i] = f1d2p(fdwrap_G(self,mvals,i),self.h,f0 = Ans['X'])
            for i in range(self.FF.np):
                if any([j in self.FF.plist[i] for j in self.fd2_pids]) or 'ALL' in self.fd2_pids:
                    FDSlice = f1d2p(fdwrap_H(self,mvals,i),self.h,f0 = Ans['G'])
                    Ans['H'][i,:] = FDSlice
                    Ans['H'][:,i] = FDSlice
        elif self.fdhessdiag:
            for i in range(self.FF.np):
                if any([j in self.FF.plist[i] for j in self.fd2_pids]) or 'ALL' in self.fd2_pids:
                    Ans['G'][i], Ans['H'][i,i] = f12d3p(fdwrap_G(self,mvals,i),self.h, f0 = Ans['X'])
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
            if not os.path.exists(os.path.join(self.root,'backups')):
                os.makedirs(os.path.join(self.root,'backups'))
            if os.path.exists(abstempdir):
                os.chdir(os.path.join(self.root,"temp"))
                FileCount = 0
                while True:
                    CandFile = os.path.join(self.root,'backups',"%s_%i.tar.bz2" % (self.name,FileCount))
                    if os.path.exists(CandFile):
                        FileCount += 1
                    else:
                        # I could use the tarfile module here
                        logger.info("Backing up: " + self.tempdir + ' to: ' + "backups/%s_%i.tar.bz2\n" % (self.name,FileCount))
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
        
        raise NotImplementedError('The get method is not implemented in the Target base class')

    def sget(self, mvals, AGrad=False, AHess=False, customdir=None):
        """ 

        Stages the directory for the target, and then calls 'get'.
        The 'get' method should not worry about the directory that it's running in.
        
        """
        ## Directory of the current iteration; if not None, then the simulation runs under
        ## temp/target_name/iteration_number
        ## The 'customdir' is customizable and can go below anything
        cwd = os.getcwd()
        
        absgetdir = os.path.join(self.root,self.tempdir)
        if Counter() is not None:
            # Not expecting more than ten thousand iterations
            iterdir = "iter_%04i" % Counter()
            absgetdir = os.path.join(absgetdir,iterdir)
        if customdir is not None:
            absgetdir = os.path.join(absgetdir,customdir)

        if not os.path.exists(absgetdir):
            os.makedirs(absgetdir)
        os.chdir(absgetdir)
        self.link_from_tempdir(absgetdir)
        self.rundir = absgetdir.replace(self.root+'/','')

        Answer = self.get(mvals, AGrad, AHess)
        os.chdir(cwd)
        
        return Answer

    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        return

    def stage(self, mvals, AGrad=False, AHess=False, customdir=None):
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
            # Not expecting more than ten thousand iterations
            iterdir = "iter_%04i" % Counter()
            absgetdir = os.path.join(absgetdir,iterdir)
        if customdir is not None:
            absgetdir = os.path.join(absgetdir,customdir)

        if not os.path.exists(absgetdir):
            os.makedirs(absgetdir)
        os.chdir(absgetdir)
        self.link_from_tempdir(absgetdir)
        self.rundir = absgetdir.replace(self.root+'/','')
        self.submit_jobs(mvals, AGrad, AHess)

        os.chdir(cwd)
        
        return

    def wq_complete(self):
        """ This method determines whether the Work Queue tasks for the current target have completed. """
        wq = getWorkQueue()
        WQIds = getWQIds()
        if wq == None:
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
        tline="Target: %s Type: %s Objective = %.5e" % (self.name, self.__class__.__name__, self.objective)
        nc = len(headings)
        if banner != None:
            tlines = [banner, tline]
        else:
            tlines = [tline]
        # Sanity check.
        for val in data.values():
            if (len(val)+1) != nc:
                raise RuntimeError('There are %i column headings, so the values in the data dictionary must be lists of length %i (currently %i)' % (nc, nc-1, len(val)))
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
            cwidths[0] = max(cwidths[0], len(k))
        # Then look at the data values to stretch out the other column widths.
        for v in data.values():
            for n, f in enumerate(v):
                cwidths[n+1] = max(cwidths[n+1], len(str(f)))
        for i in range(1, len(cwidths)):
            cwidths[i] += 2
        if cwidths[0] < 25:
            cwidths[0] = 25
        cblocks = [['' for i in range(max(crows) - len(cname.split('\n')))] + cname.split('\n') for cnum, cname in enumerate(headings)]
        # The formatting line consisting of variable column widths
        fline = ' '.join("%%%s%is" % (("-" if i==0 else ""), j) for i, j in enumerate(cwidths))
        vline = ' '.join(["%%%is" % j for i, j in enumerate(cwidths) if i > 0])
        clines = [fline % (tuple(cblocks[j][i] for j in range(nc))) for i in range(max(crows))]
        tlines += clines
        PrintDict = OrderedDict([(key, vline % (tuple(val))) for key, val in data.items()])
        printcool_dictionary(PrintDict, title='\n'.join(tlines), keywidth=cwidths[0], center=False, leftpad=4, color=color)
        
class RemoteTarget(Target):
    def __init__(self,options,tgt_opts,forcefield):
        super(RemoteTarget, self).__init__(options,tgt_opts,forcefield)
        
        self.r_options = options.copy()
        self.r_options["type"]="single"
        
        self.r_tgt_opts = tgt_opts.copy()
        self.r_tgt_opts["remote"]=False
        
        tar = tarfile.open(name="temp/%s/target.tar.bz2" % self.name, mode='w:bz2')
        tar.add("%s/targets/%s" % (self.root, self.name), arcname = "targets/%s" % self.name)
        tar.close()
        
        self.remote_indicate = ""
        
    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        n=0
        id_string = "%s_%i-%i" % (self.name, Counter(), n)
        
        while os.path.exists('%s.out' % id_string):
            n+=1
            id_string = "%s_%i-%i" % (self.name, Counter(), n)
        
        with open('forcebalance.p','w') as f: forcebalance.nifty.lp_dump((mvals, AGrad, AHess, id_string, self.r_options, self.r_tgt_opts, self.FF),f)
        
        forcebalance.nifty.LinkFile(os.path.join(os.path.split(__file__)[0],"data","rtarget.py"),"rtarget.py")
        forcebalance.nifty.LinkFile(os.path.join(self.root,"temp", self.name, "target.tar.bz2"),"%s.tar.bz2" % self.name)
        
        wq = getWorkQueue()
        
        logger.info("Sending target '%s' to work queue for remote evaluation\n" % self.name)
        # input:
        #   forcebalance.p: pickled mvals, options, and forcefield
        #   rtarget.py: remote target evaluation script
        #   target.tar.bz2: tarred target
        # output:
        #   objective.p: pickled objective function dictionary
        #   indicate.log: results of target.indicate() written to file
        forcebalance.nifty.queue_up(wq, "python rtarget.py > %s.out 2>&1" % id_string,
            ["forcebalance.p", "rtarget.py", "%s.tar.bz2" % self.name],
            ['objective_%s.p' % id_string, 'indicate_%s.log' % id_string, '%s.out' % id_string],
            tgt=self)
            
        self.id_string = id_string

    def get(self,mvals,AGrad=False,AHess=False):
        with open('indicate_%s.log' % self.id_string, 'r') as f:
            self.remote_indicate = f.read()
        with open('objective_%s.p' % self.id_string, 'r') as f:
            return forcebalance.nifty.lp_load(f)
        
    def indicate(self):
        logger.info(self.remote_indicate + '\n')


