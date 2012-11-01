""" @package liquid Matching of liquid bulk properties.  Under development.

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
try:
    from lxml import etree
except: pass
from pymbar import pymbar
import itertools
from collections import defaultdict, namedtuple
from optimizer import Counter

def weight_info(W, T, N_k, verbose=True):
    C = []
    N = 0
    W += 1.0e-300
    I = np.exp(-1*np.sum((W*np.log(W))))
    for ns in N_k:
        C.append(sum(W[N:N+ns]))
        N += ns
    C = np.array(C)
    if verbose:
        print "MBAR Results for Temperature % .1f, Box, Contributions:" % T
        print C
        print "InfoContent: % .1f snapshots (%.2f %%)" % (I, 100*I/len(W))
    return C

NPT_Trajectory = namedtuple('NPT_Trajectory', ['fnm', 'Rhos', 'pVs', 'Energies', 'Grads', 'mEnergies', 'mGrads', 'Rho_errs', 'Hvap_errs'])

class Liquid(FittingSimulation):
    
    """ Subclass of FittingSimulation for liquid property matching."""
    
    def __init__(self,options,sim_opts,forcefield):
        """Instantiation of the subclass.

        We begin by instantiating the superclass here and also
        defining a number of core concepts for energy / force
        matching.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """
        
        # Initialize the SuperClass!
        super(Liquid,self).__init__(options,sim_opts,forcefield)
        # Fractional weight of the density
        self.set_option(sim_opts,'w_rho','W_Rho')
        # Fractional weight of the enthalpy of vaporization
        self.set_option(sim_opts,'w_hvap','W_Hvap')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        
        ## The number of true atoms 
        self.natoms      = 0
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)
        ## Saved force field mvals for all iterations
        self.SavedMVal = {}
        ## Saved trajectories for all iterations and all temperatures :)
        self.SavedTraj = defaultdict(dict)
        ## Evaluated energies for all trajectories (i.e. all iterations and all temperatures), using all mvals
        self.MBarEnergy = defaultdict(lambda:defaultdict(dict))
        #======================================#
        #          UNDER DEVELOPMENT           #
        #======================================#
        # Put stuff here that I'm not sure about. :)
        np.set_printoptions(precision=4, linewidth=100)
            
    def indicate(self):
        return

    def objective_term(self,temps, exp, calc, std, grad, fitorder, name="Quantity", verbose = True, FitFitted = False, FitCoefs = False):
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
        if len(temps) == 1:
            Denom = 1.0
        
        # Now we have an array of rescaled temperatures from zero to one.
        tarray = temps - min(temps)
        tarray /= max(temps)
        # Make a Vandermonde matrix
        xmat   = np.mat(np.vander(tarray,fitorder+1)[:,::-1])
        yarray = np.array([calc[T] for T in temps])
        Beta, Hats, Yfit, MPPI = get_least_squares(xmat, yarray)
        # Curve-fitted values
        cfit = {T : r for T, r in zip(temps,Yfit)}
        GradYMat = np.mat([grad[T] for T in temps])
        GradZMat = Hats * GradYMat
        # Derivatives of least-squares coefficients
        Objective = 0.0
        Gradient = np.zeros(self.FF.np, dtype=float)
        Hessian = np.zeros((self.FF.np,self.FF.np),dtype=float)
        if FitCoefs:
            # === This piece of code doesn't work yet, for some reason. ===
            # For enthalpy of vaporization, we would like to add the derivative of the least squares coefficients to our fit.
            # Center the temperature values around room temperature and normalize.
            tarray = (temps - 298.0)
            tarray /= (max(tarray) - min(tarray))
            yarray_exp = np.array([exp[T] for T in temps])
            xmat   = np.mat(np.vander(tarray,fitorder+1)[:,::-1])
            # Perform the least-squares again.
            Beta_calc, Hats_calc, Yfit_calc, MPPI_calc = get_least_squares(xmat, yarray)
            Beta_exp, Hats_exp, Yfit_exp, MPPI_exp = get_least_squares(xmat, yarray_exp)
            Beta_calc = flat(Beta_calc)
            Beta_exp = flat(Beta_exp)
            # Assume that we're only interested in the zeroth and first-order coefficients.
            GradCoef = MPPI_calc * GradYMat
            # Fractional tolerances.
            # Shooting for error of 0.1 kJ/mol (0.2% error) in absolute Hvap
            # Shooting for 5% error in slope
            Denoms = [2e-3, 5e-2]
            for i in range(2):
                G = flat(GradCoef[i,:])
                Delta = (Beta_calc[i] - Beta_exp[i]) / Beta_exp[i]
                ThisObj = Delta ** 2 / Denoms[i] ** 2
                ThisGrad = 2.0 * Delta * G / Denoms[i] ** 2
                # Gauss-Newton approximation to the Hessian.
                Hessian += 2.0 * (np.outer(G, G)) / Denoms[i] ** 2
                bar = printcool("%s Least-Squares Coefficient %i : Calculated = % .3f, Experimental = % .3f, Objective = % .3f, Gradient of Coefficient:" % (name, i, Beta_calc[i], Beta_exp[i], ThisObj))
                self.FF.print_map(vals=G)
                Objective += ThisObj
                print Gradient.shape
                print ThisGrad.shape
                Gradient += ThisGrad
                
        else:
            for i, T in enumerate(temps):
                if FitFitted:
                    G = flat(GradZMat[i,:])
                    Delta = cfit[T] - exp[T]
                else:
                    G = grad[T]
                    Delta = calc[T] - exp[T]
                # print "T = ", T
                # print "Weight = ", Weights[T]
                # print "Delta = ", Delta
                # print "G = ", G
                ThisObj = Weights[T] * Delta ** 2 / Denom
                ThisGrad = 2.0 * Weights[T] * Delta * G / Denom
                if verbose:
                    bar = printcool("%s at %.1f : Calculated = % .3f, Experimental = % .3f, Objective = % .3f, Gradient:" % (name, T, cfit[T] if FitFitted else calc[T], exp[T], ThisObj))
                    self.FF.print_map(vals=G)
                    print "Denominator = % .3f Stdev(expt) = %.3f" % (Denom, np.std(np.array([exp[i] for i in temps])))
                else:
                    print "%s at %.1f contributes % .3f to the objective function" % (name, T, ThisObj)
                Objective += ThisObj
                Gradient += ThisGrad
                # Gauss-Newton approximation to the Hessian.
                Hessian += 2.0 * Weights[T] * (np.outer(G, G)) / Denom
            
        # Do this one thousand times!
        # Random_Objectives = []
        # for I in range(10000):
        #     Objective_ = 0.0
        #     for i, T in enumerate(temps):
        #         if FitFitted:
        #             Delta_ = cfit_[T] - exp[T]
        #         else:
        #             Delta_ = calc[T] - exp[T] + np.random.normal(0,std[T])
        #         Objective_ += Weights[T] * Delta_ ** 2 / Denom
        #     Random_Objectives.append(Objective_)
        # print "The mean and 95% upper/lower confidence bounds of the objective function are:", sorted(Random_Objectives)[5000], sorted(Random_Objectives)[9750], sorted(Random_Objectives)[250]

        Delta = np.array([calc[T] - exp[T] for T in temps])
        Defit = np.array([cfit[T] - exp[T] for T in temps])
        delt = {T : r for T, r in zip(temps,Delta)}
        dfit = {T : r for T, r in zip(temps,Defit)}
        print_out = {'    %8.3f' % T:"%9.3f   %9.3f +- %-7.3f % 7.3f  %9.3f  % 7.3f" % (exp[T],calc[T],std[T],delt[T],cfit[T],dfit[T]) for T in calc}
        return Objective, Gradient, Hessian, print_out

    def get(self, mvals, AGrad=True, AHess=True):
        
        """
        Fitting of liquid bulk properties.  This is the current major
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
        @return Answer Contribution to the objective function
        
        """

        Answer = {}
        # Dump the force field to a pickle file
        with open('forcebalance.p','w') as f: lp_dump((self.FF,mvals,self.h,AGrad),f)

        # Reference densities from the CRC Handbook, interpolated. :P
        # Rho_exp = {273.2 : 999.84300, 274.2 : 999.90170, 275.4 : 999.94910, 
        #            276.6 : 999.97220, 277.8 : 999.97190, 279.2 : 999.94310, 
        #            280.7 : 999.87970, 282.3 : 999.77640, 284.1 : 999.61820, 
        #            285.9 : 999.41760, 287.9 : 999.14730, 290.1 : 998.79510, 
        #            292.4 : 998.36860, 294.9 : 997.84070, 297.6 : 997.19920, 
        #            300.4 : 996.45990, 303.5 : 995.55780, 306.9 : 994.47280, 
        #            310.5 : 993.22070, 314.3 : 991.75010, 318.5 : 990.04400, 
        #            322.9 : 988.13200, 327.7 : 985.90000, 332.9 : 983.31300, 
        #            338.4 : 980.41000, 344.4 : 977.04400, 350.8 : 973.24400, 
        #            357.8 : 968.85000, 365.2 : 963.94000, 373.2 : 958.37000}

        # Better reference densities including supercooled temperatures. :)
        Rho_exp = {243.2 : 983.854137445, 247.2 : 988.60299822,  250.8 : 991.824502494, 
                   253.9 : 993.994466297, 256.8 : 995.618742244, 259.5 : 996.836459755, 
                   262.0 : 997.74652127,  264.3 : 998.421562116, 266.6 : 998.957824968, 
                   268.7 : 999.337763488, 270.9 : 999.633348744, 273.2 : 999.83952, 
                   275.1 : 999.936526572, 277.2 : 999.971994126, 279.4 : 999.933616453, 
                   281.6 : 999.822945901, 283.9 : 999.634813488, 286.4 : 999.351547484, 
                   289.0 : 998.975273971, 291.8 : 998.482682091, 294.9 : 997.838201247, 
                   298.2 : 997.044895418, 302.3 : 995.915450218, 306.8 : 994.505082827, 
                   312.2 : 992.594323895, 318.6 : 990.045341555, 326.7 : 986.414237859, 
                   337.3 : 981.041345405, 351.8 : 972.665015969, 373.2 : 958.363657052}


        # Hvap_exp = {273.2 : 45.04377000, 274.2 : 45.00227882, 275.4 : 44.95242692, 
        #             276.6 : 44.90250684, 277.8 : 44.85251860, 279.2 : 44.79411281, 
        #             280.7 : 44.73143222, 282.3 : 44.66445551, 284.1 : 44.58896185, 
        #             285.9 : 44.51331480, 287.9 : 44.42908262, 290.1 : 44.33620851, 
        #             292.4 : 44.23886785, 294.9 : 44.13277874, 297.6 : 44.01787016, 
        #             300.4 : 43.89834117, 303.5 : 43.76557256, 306.9 : 43.61943226, 
        #             310.5 : 43.46409895, 314.3 : 43.29947040, 318.5 : 43.11671718, 
        #             322.9 : 42.92436572, 327.7 : 42.71348245, 332.9 : 42.48379470, 
        #             338.4 : 42.23946269, 344.4 : 41.97128539, 350.8 : 41.68335109, 
        #             357.8 : 41.36620261, 365.2 : 41.02840899, 373.2 : 40.66031044}
        Hvap_exp = {243.2 : 46.3954889576, 247.2 : 46.1962620917, 250.8 : 46.026660115, 
                    253.9 : 45.8854856909, 256.8 : 45.7562378857, 259.5 : 45.6376566716, 
                    262.0 : 45.5289631106, 264.3 : 45.4296676383, 266.6 : 45.3308849086, 
                    268.7 : 45.2410382648, 270.9 : 45.14718848,   273.2 : 45.05426457, 
                    275.1 : 44.97303237,   277.2 : 44.883316  ,   279.4 : 44.78938954, 
                    281.6 : 44.69551282,   283.9 : 44.59740693,   286.4 : 44.49079567, 
                    289.0 : 44.37992667,   291.8 : 44.2605107 ,   294.9 : 44.12824356, 
                    298.2 : 43.98733503,   302.3 : 43.81204172,   306.8 : 43.61925833, 
                    312.2 : 43.38721755,   318.6 : 43.11094613,   326.7 : 42.75881714, 
                    337.3 : 42.29274032,   351.8 : 41.64286045,   373.2 : 40.64987322}

        #Rho_exp = {297.6 : 997.19920, }
        #Hvap_exp = {297.6 : 44.01787016, }
        
        # This is just the pV part
        # Hvap_exp = {273.2 : -0.394460540333, 274.2 : -0.394437383223, 275.4 : -0.394418685939, 
        #             276.6 : -0.394409574614, 277.8 : -0.394409692941, 279.2 : -0.394421052586, 
        #             280.7 : -0.39444606189,  282.3 : -0.394486817281, 284.1 : -0.394549248932, 
        #             285.9 : -0.394628441633, 287.9 : -0.394735200734, 290.1 : -0.394874394186, 
        #             292.4 : -0.395043083314, 294.9 : -0.39525207784,  297.6 : -0.395506344197, 
        #             300.4 : -0.395799780832, 303.5 : -0.396158424984, 306.9 : -0.396590645846, 
        #             310.5 : -0.397090606376, 314.3 : -0.397679425521, 318.5 : -0.398364729273, 
        #             322.9 : -0.399135550745, 327.7 : -0.400039162216, 332.9 : -0.401091625991, 
        #             338.4 : -0.402279260746, 344.4 : -0.403665147146, 350.8 : -0.405241244773, 
        #             357.8 : -0.407079124765, 365.2 : -0.409152654759, 373.2 : -0.411530630162}

        # Sorted list of temperatures.
        Temps = np.array(sorted([i for i in Rho_exp]))

        # Launch a series of simulations
        for T in Temps:
            if not os.path.exists('%.1f' % T):
                os.makedirs('%.1f' % T)
            os.chdir('%.1f' % T)
            self.npt_simulation(T)
            os.chdir('..')

        # Wait for simulations to finish
        wq_wait(self.wq)

        # Uncomment in case I screwed up ...
        #print "Extract the directory contents now please"
        #raw_input()

        # Gather the calculation data
        Results = {t : lp_load(open('./%.1f/npt_result.p' % T)) for t, T in enumerate(Temps)}

        Rhos, pVs, Energies, Grads, mEnergies, mGrads, Rho_errs, Hvap_errs = ([Results[t][i] for t in range(len(Temps))] for i in range(8))

        R  = np.array(list(itertools.chain(*list(Rhos))))
        PV = np.array(list(itertools.chain(*list(pVs))))
        E  = np.array(list(itertools.chain(*list(Energies))))
        G  = np.hstack(tuple(Grads))
        mE = np.array(list(itertools.chain(*list(mEnergies))))
        mG = np.hstack(tuple(mGrads))

        Rho_calc = {}
        Rho_grad = {}
        Rho_std  = {}
        Hvap_calc = {}
        Hvap_grad = {}
        Hvap_std  = {}

        Sims = len(Temps)
        Shots = len(Energies[0])
        mShots = len(mEnergies[0])
        N_k = np.ones(Sims)*Shots
        mN_k = np.ones(Sims)*mShots
        # Use the value of the energy for snapshot t from simulation k at potential m
        U_kln = np.zeros([Sims,Sims,Shots], dtype = np.float64)
        mU_kln = np.zeros([Sims,Sims,mShots], dtype = np.float64)
        ## This fills out a 'square' in the matrix with 30 trajectories and 30 temperatures
        for m, T in enumerate(Temps):
            beta = 1. / (kb * T)
            for k in range(len(Temps)):
                U_kln[k, m, :]   = Energies[k]
                U_kln[k, m,: ]  *= beta
                mU_kln[k, m, :]  = mEnergies[k]
                mU_kln[k, m, :] *= beta
        print "Running MBAR analysis..."
        mbar = pymbar.MBAR(U_kln, N_k, verbose=False, relative_tolerance=5.0e-8)
        mmbar = pymbar.MBAR(mU_kln, mN_k, verbose=False, relative_tolerance=5.0e-8)
        W1 = mbar.getWeights()
        mW1 = mmbar.getWeights()
        print "Done"

        def random_temperature_trial():
            rrnd = []
            for r, dr in zip(Rhos, Rho_errs):
                rrnd.append(r + dr*np.random.randn(1)[0])
            R_ = np.array(list(itertools.chain(*rrnd)))
            for i, T in enumerate(Temps):
                # The weights that we want are the last ones.
                W = flat(W1[:,i])
                Rho_calc[T]   = np.dot(W,R_)

            # Get contributions to the objective function
            X, _, __, ___ = self.objective_term(Temps, Rho_exp, Rho_calc, Rho_std, Rho_grad, 3, name="Density", verbose=False)
            return X

        for i, T in enumerate(Temps):
            # The weights that we want are the last ones.
            W = flat(W1[:,i])
            C = weight_info(W, T, N_k, verbose=False)
            mW = flat(mW1[:,i])
            Gbar = flat(np.mat(G)*col(W))
            mGbar = flat(np.mat(mG)*col(mW))
            mBeta = -1/kb/T
            Rho_calc[T]   = np.dot(W,R)
            Rho_grad[T]   = mBeta*(flat(np.mat(G)*col(W*R)) - np.dot(W,R)*Gbar)
            Hvap_calc[T]  = np.dot(mW,mE) - np.dot(W,E)/216 + kb*T - np.dot(W, PV)
            Hvap_grad[T]  = mGbar + mBeta*(flat(np.mat(mG)*col(mW*mE)) - np.dot(mW,mE)*mGbar)
            Hvap_grad[T] -= (Gbar + mBeta*(flat(np.mat(G)*col(W*E)) - np.dot(W,E)*Gbar)) / 216
            ###
            # The pV terms are behaving strangely, there's a chance I got the derivative wrong
            # However, it contributes such a negligible amount to the gradient / Hessian that perhaps I'll just leave the derivative term out.
            # Hvap_grad[T] -= mBeta*(flat(np.mat(G)*col(W*PV)) - np.dot(W,PV)*Gbar)
            # Hvap_calc[T]  = - np.dot(W, PV)
            # Hvap_grad[T]  = -1*( mBeta*(flat(np.mat(G)*col(W*PV)) - np.dot(W,PV)*Gbar))
            ###
            Rho_std[T]    = np.sqrt(sum(C**2 * np.array(Rho_errs)**2))
            Hvap_std[T]   = np.sqrt(sum(C**2 * np.array(Hvap_errs)**2))

            #print "C = ", C
            #print "Rho_errs = ", Rho_errs

        #print "The -PV contribution to the gradient is: (Hopefully it is right)"
        #print -1 * (mBeta*(flat(np.mat(G)*col(W*PV)) - np.dot(W,PV)*Gbar))

        # Get contributions to the objective function
        X_Rho, G_Rho, H_Rho, RhoPrint = self.objective_term(Temps, Rho_exp, Rho_calc, Rho_std, Rho_grad, 3, name="Density", verbose=False)
        X_Hvap, G_Hvap, H_Hvap, HvapPrint = self.objective_term(Temps, Hvap_exp, Hvap_calc, Hvap_std, Hvap_grad, 2, name="H_vap", verbose=False)

        Gradient = np.zeros(self.FF.np, dtype=float)
        Hessian = np.zeros((self.FF.np,self.FF.np),dtype=float)

        w_1 = self.W_Rho / (self.W_Rho + self.W_Hvap)
        w_2 = self.W_Hvap / (self.W_Rho + self.W_Hvap)

        Objective    = w_1 * X_Rho + w_2 * X_Hvap
        if AGrad:
            Gradient = w_1 * G_Rho + w_2 * G_Hvap
        if AHess:
            Hessian  = w_1 * H_Rho + w_2 * H_Hvap

        printcool_dictionary(RhoPrint,title='Rho vs T:   Reference  Calculated +- Stdev    Delta    CurveFit   D(Cfit)',bold=True,color=4,keywidth=15)
        bar = printcool("Density objective function: % .3f, Derivative:" % X_Rho)
        self.FF.print_map(vals=G_Rho)
        print bar

        printcool_dictionary(HvapPrint,title='Hvap vs T:  Reference  Calculated +- Stdev    Delta    CurveFit   D(Cfit)',bold=True,color=3,keywidth=15)
        bar = printcool("H_vap objective function: % .3f, Derivative:" % X_Hvap)
        self.FF.print_map(vals=G_Hvap)
        print bar

        #for i in range(1000):
        #    print "Density objective function, random trial %i : " % i,
        #    print random_temperature_trial()
        
        Answer = {'X':Objective, 'G':Gradient, 'H':Hessian}
        return Answer

