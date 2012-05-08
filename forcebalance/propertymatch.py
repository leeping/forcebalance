""" @package property_match Matching of experimental properties.  Under development.

@author Lee-Ping Wang
@date 04/2012
"""

import os
import shutil
from nifty import *
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

    def objective_term(self,temps, exp, calc, std, grad, fitorder, name="Quantity"):
        # Assuming all uncertainties equal for now. :P
        Uncerts = {}
        for i in exp:
            Uncerts[i] = 1.0
        # Build an array of weights.  (Cumbersome)
        weight_array = np.array([Uncerts[i] for i in temps])
        weight_array = 1.0 / weight_array
        weight_array /= sum(weight_array)
        weight_array = list(weight_array)
        Weights = {}
        for T, W in zip(temps,weight_array):
            Weights[T] = W
        # The denominator
        Denom = np.std(np.array([exp[i] for i in temps])) ** 2
        # Now we have an array of rescaled temperatures from zero to one.
        tarray = temps - min(temps)
        tarray /= max(temps)
        # Make a Vandermonde matrix
        xmat   = np.mat(np.vander(tarray,fitorder+1)[:,::-1])
        yarray = np.array([calc[T] for T in temps])
        Beta, Hats, Yfit = get_least_squares(xmat, yarray)
        # Curve-fitted densities
        cfit = {T : r for T, r in zip(temps,Yfit)}
        # Switch to fit the curve-fitted densities :P
        FitFitted = 1
        GradYMat = np.mat([grad[T] for T in temps])
        GradZMat = Hats * GradYMat

        Objective = 0.0
        Gradient = np.zeros(self.FF.np, dtype=float)
        Hessian = np.zeros((self.FF.np,self.FF.np),dtype=float)
        for i, T in enumerate(temps):
            if FitFitted:
                G = flat(GradZMat[i,:])
                DRho = cfit[T] - exp[T]
            else:
                G = grad[T]
                DRho = calc[T] - exp[T]
            ThisObj = Weights[T] * DRho ** 2 / Denom
            ThisGrad = 2.0 * Weights[T] * DRho * G / Denom
            bar = printcool("%s at %.1f :%s Objective = % .3f, Gradient:" % (name, T, ' Smoothed' if FitFitted else '', ThisObj))
            self.FF.print_map(vals=ThisGrad)
            Objective += ThisObj
            Gradient += ThisGrad
            # The second derivatives are inaccurate; we estimate the objective function Hessian using first derivatives.
            # If we had a good Hessian estimate the formula would be: 2.0 * Weights[T] * (np.outer(G, G) + DRho * np.diag(Hd))
            Hessian += 2.0 * Weights[T] * (np.outer(G, G)) / Denom
        print bar

        Delta = np.array([calc[T] - exp[T] for T in temps])
        Defit = np.array([cfit[T] - exp[T] for T in temps])
        delt = {T : r for T, r in zip(temps,Delta)}
        dfit = {T : r for T, r in zip(temps,Defit)}
        print_out = {'    %8.3f' % T:"%9.3f   %9.3f +- %-7.3f % 7.3f  %9.3f  % 7.3f" % (exp[T],calc[T],std[T],delt[T],cfit[T],dfit[T]) for T in calc}
        return Objective, Gradient, Hessian, print_out

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

        # Reference densities from the CRC Handbook. :P
        Rho_exp = {273.2 : 999.84300, 274.2 : 999.90170, 275.4 : 999.94910, 
                   276.6 : 999.97220, 277.8 : 999.97190, 279.2 : 999.94310, 
                   280.7 : 999.87970, 282.3 : 999.77640, 284.1 : 999.61820, 
                   285.9 : 999.41760, 287.9 : 999.14730, 290.1 : 998.79510, 
                   292.4 : 998.36860, 294.9 : 997.84070, 297.6 : 997.19920, 
                   300.4 : 996.45990, 303.5 : 995.55780, 306.9 : 994.47280, 
                   310.5 : 993.22070, 314.3 : 991.75010, 318.5 : 990.04400, 
                   322.9 : 988.13200, 327.7 : 985.90000, 332.9 : 983.31300, 
                   338.4 : 980.41000, 344.4 : 977.04400, 350.8 : 973.24400, 
                   357.8 : 968.85000, 365.2 : 963.94000, 373.2 : 958.37000}

        Hvap_exp = {273.2 : 45.04377000, 274.2 : 45.00227882, 275.4 : 44.95242692, 
                    276.6 : 44.90250684, 277.8 : 44.85251860, 279.2 : 44.79411281, 
                    280.7 : 44.73143222, 282.3 : 44.66445551, 284.1 : 44.58896185, 
                    285.9 : 44.51331480, 287.9 : 44.42908262, 290.1 : 44.33620851, 
                    292.4 : 44.23886785, 294.9 : 44.13277874, 297.6 : 44.01787016, 
                    300.4 : 43.89834117, 303.5 : 43.76557256, 306.9 : 43.61943226, 
                    310.5 : 43.46409895, 314.3 : 43.29947040, 318.5 : 43.11671718, 
                    322.9 : 42.92436572, 327.7 : 42.71348245, 332.9 : 42.48379470, 
                    338.4 : 42.23946269, 344.4 : 41.97128539, 350.8 : 41.68335109, 
                    357.8 : 41.36620261, 365.2 : 41.02840899, 373.2 : 40.66031044}

        # Sorted list of temperatures.
        Temps = np.array(sorted([i for i in Rho_exp]))

        # Launch a series of simulations
        for T in Temps:
            if not os.path.exists('%.1f' % T):
                os.makedirs('%.1f' % T)
            os.chdir('%.1f' % T)
            self.execute(T,os.getcwd())
            os.chdir('..')

        # Wait for simulations to finish
        wq_wait(self.wq)

        # Gather the calculation data
        Results = {T : lp_load(open('./%.1f/npt_result.p' % T)) for T in Temps}
        Rho_calc = {T : Results[T][0] for T in Temps}
        Rho_std  = {T : Results[T][1] for T in Temps}
        Rho_grad = {T : Results[T][2] for T in Temps}
        Hvap_calc = {T : Results[T][3] for T in Temps}
        Hvap_std  = {T : Results[T][4] for T in Temps}
        Hvap_grad = {T : Results[T][5] for T in Temps}
        
        # Get contributions to the objective function
        X_Rho, G_Rho, H_Rho, RhoPrint = self.objective_term(Temps, Rho_exp, Rho_calc, Rho_std, Rho_grad, 3, name="Density")
        X_Hvap, G_Hvap, H_Hvap, HvapPrint = self.objective_term(Temps, Hvap_exp, Hvap_calc, Hvap_std, Hvap_grad, 2, name="H_vap")

        Objective = X_Rho + X_Hvap
        if AGrad:
            Gradient = G_Rho + G_Hvap
        if AHess:
            Hessian = H_Rho + H_Hvap
        
        printcool_dictionary(RhoPrint,title='Rho vs T:   Reference  Calculated +- Stdev    Delta    CurveFit   D(Cfit)',bold=True,color=4,keywidth=15)
        bar = printcool("Density objective function: % .3f, Derivative:" % X_Rho)
        self.FF.print_map(vals=G_Rho)
        print bar

        printcool_dictionary(HvapPrint,title='Hvap vs T:  Reference  Calculated +- Stdev    Delta    CurveFit   D(Cfit)',bold=True,color=3,keywidth=15)
        bar = printcool("H_vap objective function: % .3f, Derivative:" % X_Hvap)
        self.FF.print_map(vals=G_Hvap)
        print bar
        
        Answer = {'X':Objective, 'G':Gradient, 'H':Hessian}
        os.chdir(cwd)
        return Answer

