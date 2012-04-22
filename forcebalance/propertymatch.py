""" @package property_match Matching of experimental properties.

@author Lee-Ping Wang
@date 04/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros
from fitsim import FittingSimulation
from molecule import Molecule
from re import match
import subprocess
from subprocess import PIPE

def wq_wait(wq):
    while not wq.empty():
        print '---'
        task = wq.wait(10)
        if task:
            print 'A job has finished!'
            print 'Job name = ', task.tag, 'command = ', task.command
            print 'output', task.output,
            print 'id', task.id
            print "preferred_host = ", task.preferred_host, 
            print "status = ", task.status, 
            print "return_status = ", task.return_status, 
            print "result = ", task.result, 
            print "host = ", task.host
            print "computation_time = ", task.computation_time/1000000, 
            print "total_bytes_transferred = ", task.total_bytes_transferred,
            if task.result != 0:
                wq.submit(task)
            else:
                del task
        print "Workers: %i init, %i ready, %i busy, %i total joined, %i total removed" \
            % (wq.stats.workers_init, wq.stats.workers_ready, wq.stats.workers_busy, wq.stats.total_workers_joined, wq.stats.total_workers_removed)
        print "Tasks: %i running, %i waiting, %i total dispatched, %i total complete" \
            % (wq.stats.tasks_running,wq.stats.tasks_waiting,wq.stats.total_tasks_dispatched,wq.stats.total_tasks_complete)
        print "Data: %i / %i kb sent/received" % (wq.stats.total_bytes_sent/1000, wq.stats.total_bytes_received/1024)

class PropertyMatch(FittingSimulation):
    
    """ Subclass of FittingSimulation for property matching."""
    
    def __init__(self,options,sim_opts,forcefield):
        """Instantiation of the subclass.

        We begin by instantiating the superclass here and also
        defining a number of core concepts for energy / force
        matching.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """
        
        # Initialize the SuperClass!
        super(PropertyMatch,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        
        ## The number of true atoms 
        self.natoms      = 0
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts,forcefield)

        #======================================#
        #          UNDER DEVELOPMENT           #
        #======================================#
        # Put stuff here that I'm not sure about. :)
            
    def indicate(self):
        print "Sim: %-15s E_err(kJ/mol)= %10.4f F_err(%%)= %10.4f" % (self.name, self.e_err, self.f_err*100)

    def get(self, mvals, AGrad=False, AHess=False, tempdir=None):
        """
        LPW 04-21-2012
        
        @todo Document me.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient, useless here
        @param[in] AHess Switch to turn on analytic Hessian, useless here
        @param[in] tempdir Temporary directory for running computation
        @return Answer Contribution to the objective function
        """
        if tempdir == None:
            tempdir = self.tempdir
        abstempdir = os.path.join(self.root,self.tempdir)
        Answer = {}
        cwd = os.getcwd()
        # Create the new force field!!
        pvals = self.FF.make(tempdir,mvals,self.usepvals)
        # Go into the temporary directory
        os.chdir(os.path.join(self.root,tempdir))

        # Launch a series of simulations
        for Temperature in [248, 273, 278, 283, 288, 293, 298, 323, 348, 373]:
            os.makedirs('%i' % Temperature)
            os.chdir('%i' % Temperature)
            self.execute(Temperature,os.getcwd())
            os.chdir('..')

        wq_wait(self.wq)
            
        Answer = {'X':BC, 'G':zeros(self.FF.np), 'H':zeros((self.FF.np,self.FF.np))}
        os.chdir(cwd)
        return Answer
