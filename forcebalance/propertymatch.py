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
        print "Sim: %-15s E_err(kJ/mol)= %10.4f F_err(%%)= %10.4f" % (self.name, self.e_err, self.f_err*100)

    def get(self, mvals, AGrad=False, AHess=False, tempdir=None):
        
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
        with open(os.path.join(self.root,tempdir,'forcebalance.p'),'w') as f: lp_dump((self.FF,mvals),f)

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

        Answer = {'X':Objective, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np,self.FF.np))}
        os.chdir(cwd)
        return Answer

