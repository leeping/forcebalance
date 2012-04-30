""" @package property_match Matching of experimental properties.

@author Lee-Ping Wang
@date 04/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool_dictionary, lp_dump, lp_load, printcool
from fitsim import FittingSimulation
import numpy as np
from molecule import Molecule
from re import match
import subprocess
from subprocess import PIPE
from lxml import etree

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
        # Go into the temporary directory
        os.chdir(os.path.join(self.root,tempdir))
        # Dump the force field to a pickle file
        with open(os.path.join(self.root,tempdir,'forcebalance.p'),'w') as f: lp_dump((self.FF,mvals))

        DensityRef = {235.5 : 968.8, 248.0 : 989.2,
                      260.5 : 997.1, 273.0 : 999.8,
                      285.5 : 999.5, 298.0 : 997.2,
                      323.0 : 988.3, 348.0 : 975.2,
                      373.0 : 958.7, 400.0 : 938.0}

        TempSeries = sorted([i for i in DensityRef])

        Denom = np.std(np.array([DensityRef[i] for i in TempSeries]))
        
        # Launch a series of simulations
        for Temperature in TempSeries:
            os.makedirs('%.1f' % Temperature)
            os.chdir('%.1f' % Temperature)
            self.execute(Temperature,os.getcwd())
            os.chdir('..')

        wq_wait(self.wq)

        DensityCalc = {}
        DensityErr = {}
        for Temperature in TempSeries:
            for line in open('./%.1f/npt.out' % Temperature):
                if 'Density: mean' in line:
                    DensityCalc[Temperature] = float(line.split()[2]) * 1000
                    DensityErr[Temperature] = float(line.split()[4]) * 1000
        
        DensityPrint = {T:"%.3f +- %.3f" % (DensityCalc[T],DensityErr[T]) for T in DensityCalc}

        Delta = np.array([DensityCalc[T] - DensityRef[T] for T in TempSeries]) / Denom
        Objective = np.mean(Delta*Delta)

        printcool_dictionary(DensityRef,title='Reference Densities',color=3)
        printcool_dictionary(DensityPrint,title='Calculated Densities',color=4)
        print "Deltas:", Delta
        print "Objective:", Objective

        # Pseudocode for (future) analytic gradients.
        # if AGrad:
        #     Grad = np.zeros(self.FF.np)
        #     for Temperature in TempSeries:
        #         SimTraj = Load_SimTraj('%.1f/dynamics.dcd' % Temperature)
        #         for frame in SimTraj:
        #             rho = SimTraj.Densities[frame]
        #             for parameter in FF:
        #                 dEdk = finite_difference_derivative(Compute_Energy(SimTraj[frame]))
        #                 Grad[parameter] += -1 * beta * rho * dEdk / nFrames
                    

        Answer = {'X':Objective, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np,self.FF.np))}
        os.chdir(cwd)
        return Answer

