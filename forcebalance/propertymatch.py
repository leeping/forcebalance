""" @package property_match Matching of experimental properties.  Under development.

@author Lee-Ping Wang
@date 04/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool_dictionary, lp_dump, lp_load, printcool, wq_wait
from fitsim import FittingSimulation
import numpy as np
from molecule import Molecule
from re import match
import subprocess
from subprocess import PIPE
from lxml import etree

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
        print "Placeholder for the density error and stuff. :)"
        #print "Sim: %-15s E_err(kJ/mol)= %10.4f F_err(%%)= %10.4f" % (self.name, self.e_err, self.f_err*100)

    def get(self, mvals, AGrad=True, AHess=True, tempdir=None):
        
        """
        Fitting of experimental properties.  This is the current major
        direction of development for ForceBalance.  Basically, fitting
        the QM energies / forces alone does not always give us the
        best simulation behavior.  In many cases it makes more sense
        to try and reproduce some experimentally known data as well.

        In order to reproduce experimentally known data, we need to
        run a simulation and compare the simulation result to
        experiment.  The main challenge here is that the simulations
        are computationally intensive (i.e. they require energy and
        force evaluations), and furthermore the results are noisy.  We
        need to run the simulations automatically and remotely
        (i.e. on clusters) and a good way to calculate the derivatives
        of the simulation results with respect to the parameter values.

        This function contains some experimentally known values of the
        density and enthalpy of vaporization (Hvap) of liquid water.
        It launches the density and Hvap calculations on the cluster,
        and gathers the results / derivatives.  The actual calculation
        of results / derivatives is done in a separate file.

        After the results come back, they are gathered together to form
        an objective function.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @param[in] tempdir Temporary directory for running computation
        @return Answer Contribution to the objective function
        
        """

        print "get has been called with mvals = ", mvals

        if tempdir == None:
            tempdir = self.tempdir
        abstempdir = os.path.join(self.root,self.tempdir)
        Answer = {}
        cwd = os.getcwd()
        # Go into the temporary directory
        os.chdir(os.path.join(self.root,tempdir))
        # Dump the force field to a pickle file
        with open(os.path.join(self.root,tempdir,'forcebalance.p'),'w') as f: lp_dump((self.FF,mvals,self.h),f)

        RhoRef = {235.5 : 968.8, 248.0 : 989.2,
                      260.5 : 997.1, 273.0 : 999.8,
                      285.5 : 999.5, 298.0 : 997.2,
                      323.0 : 988.3, 348.0 : 975.2,
                      373.0 : 958.7, 400.0 : 938.0}

        # Rough estimates of density simulation uncertainties (relative)
        Uncerts = {235.5 : 9.0, 248.0 : 7.0,
                   260.5 : 5.0, 273.0 : 4.0,
                   285.5 : 3.0, 298.0 : 3.0,
                   323.0 : 3.0, 348.0 : 3.0,
                   373.0 : 3.0, 400.0 : 4.0}

        TempSeries = sorted([i for i in RhoRef])
        Denom = np.std(np.array([RhoRef[i] for i in TempSeries])) ** 2

        # Build an array of weights.
        weight_array = np.array([Uncerts[i] for i in TempSeries])
        weight_array = 1.0 / weight_array
        weight_array /= sum(weight_array)
        weight_array = list(weight_array)

        Weights = {}
        for T, W in zip(TempSeries,weight_array):
            Weights[T] = W
        
        # Launch a series of simulations
        for T in TempSeries:
            if not os.path.exists('%.1f' % T):
                os.makedirs('%.1f' % T)
            os.chdir('%.1f' % T)
            self.execute(T,os.getcwd())
            os.chdir('..')

        # Wait for simulations to finish
        wq_wait(self.wq)

        RhoCalc = {}
        RhoStd = {}
        
        Objective = 0.0
        Gradient = np.zeros(self.FF.np, dtype=float)
        Hessian = np.zeros((self.FF.np,self.FF.np),dtype=float)
        
        for T in TempSeries:
            RhoCalc[T], RhoStd[T], G, Hd = lp_load(open('./%.1f/npt_result.p' % T))
            DRho = RhoCalc[T] - RhoRef[T]
            Objective += Weights[T] * DRho ** 2 / Denom
            if AGrad:
                Gradient += 2.0 * Weights[T] * DRho * G
            if AHess:
                print np.outer(G, G)
                print DRho * np.diag(Hd)
                #Hessian += 2.0 * Weights[T] * (np.outer(G, G) + DRho * np.diag(Hd))
                Hessian += 2.0 * Weights[T] * (np.outer(G, G))# + DRho * np.diag(Hd))
            
            # for line in open('./%.1f/npt.out' % T):
            #     if 'Rho: mean' in line:
            #         RhoCalc[T] = float(line.split()[2]) * 1000
            #         RhoStd[T] = float(line.split()[4]) * 1000
        
        RhoPrint = {T:"%.3f +- %.3f" % (RhoCalc[T],RhoStd[T]) for T in RhoCalc}
        printcool_dictionary(RhoRef,title='Reference Densities',color=3)
        printcool_dictionary(RhoPrint,title='Calculated Densities',color=4)
        Delta = np.array([RhoCalc[T] - RhoRef[T] for T in TempSeries]) / Denom
        print "Deltas:", Delta
        print "Objective:", Objective
        
        Answer = {'X':Objective, 'G':Gradient, 'H':Hessian}
        print Answer
        os.chdir(cwd)
        return Answer

