""" @package liquid Matching of liquid bulk properties.  Under development.

@author Lee-Ping Wang
@date 04/2012
"""

import os
import shutil
from nifty import *
from target import Target
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

# NPT_Trajectory = namedtuple('NPT_Trajectory', ['fnm', 'Rhos', 'pVs', 'Energies', 'Grads', 'mEnergies', 'mGrads', 'Rho_errs', 'Hvap_errs'])

class Liquid(Target):
    
    """ Subclass of Target for liquid property matching."""
    
    def __init__(self,options,tgt_opts,forcefield):
        """Instantiation of the subclass.

        We begin by instantiating the superclass here and also
        defining a number of core concepts for energy / force
        matching.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """
        
        # Initialize the SuperClass!
        super(Liquid,self).__init__(options,tgt_opts,forcefield)
        # Fractional weight of the density
        self.set_option(tgt_opts,'w_rho','W_Rho')
        # Fractional weight of the enthalpy of vaporization
        self.set_option(tgt_opts,'w_hvap','W_Hvap')
        # Fractional weight of the thermal expansion coefficient
        self.set_option(tgt_opts,'w_alpha','W_Alpha')
        # Fractional weight of the isothermal compressibility
        self.set_option(tgt_opts,'w_kappa','W_Kappa')
        # Fractional weight of the isobaric heat capacity
        self.set_option(tgt_opts,'w_cp','W_Cp')
        # Optionally pause on the zeroth step
        self.set_option(tgt_opts,'manual','manual')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,tgt_opts)
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

    def objective_term(self,temps, exp, calc, err, grad, fitorder, name="Quantity", verbose = True, FitFitted = False, Denom=None, Weights=None):
        # Have the option to assume all uncertainties equal.
        Uncerts = {}
        InverseErrorWeights = False
        for i in exp:
            if InverseErrorWeights:
                Uncerts[i] = err[i]
            else:
                Uncerts[i] = 1.0
        if Weights == None:
            # Build an array of weights.  (Cumbersome)
            weight_array = np.array([Uncerts[i]**-2 for i in temps])
            weight_array /= sum(weight_array)
            weight_array = list(weight_array)
            Weights = {}
            for T, W in zip(temps,weight_array):
                Weights[T] = W
        else:
            Sum = sum(Weights.values())
            for i in Weights:
                Weights[i] /= Sum
            print "Weights have been renormalized to", sum(Weights.values())
        # Use least-squares or hyperbolic (experimental) objective.
        LeastSquares = True
        # The denominator
        if Denom == None:
            Denom = np.std(np.array([exp[i] for i in temps]))
            if len(temps) == 1:
                Denom = 1.0
        print "Physical quantity %s uses denominator = % .4f" % (name, Denom)
        if not LeastSquares:
            # If using a hyperbolic functional form
            # we still want the contribution to the 
            # objective function to be the same when
            # Delta = Denom.
            Denom /= 3 ** 0.5
        
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
        Objs = {}
        GradMap = []
        for i, T in enumerate(temps):
            if FitFitted:
                G = flat(GradZMat[i,:])
                Delta = cfit[T] - exp[T]
            else:
                G = grad[T]
                Delta = calc[T] - exp[T]
            if LeastSquares:
                # Least-squares objective function.
                ThisObj = Weights[T] * Delta ** 2 / Denom**2
                Objs[T] = ThisObj
                ThisGrad = 2.0 * Weights[T] * Delta * G / Denom**2
                GradMap.append(G)
                Objective += ThisObj
                Gradient += ThisGrad
                # Gauss-Newton approximation to the Hessian.
                Hessian += 2.0 * Weights[T] * (np.outer(G, G)) / Denom**2
            else:
                # L1-like objective function.
                D = Denom
                S = Delta**2 + D**2
                # if np.abs(Delta) < D:
                #     print "%s, temp % .1f : abs(calc-exp) = % .1e (< % .1e)" % (name,T,np.abs(Delta),D)
                # else:
                #     print "%s, temp % .1f : abs(calc-exp) = % .1e ;" % (name,T,np.abs(Delta)),
                #     print "recommend trust <", Delta/np.abs(G)
                ThisObj  = Weights[T] * (S**0.5-D) / Denom
                ThisGrad = Weights[T] * (Delta/S**0.5) * G / Denom
                ThisHess = Weights[T] * (1/S**0.5-Delta**2/S**1.5) * np.outer(G,G) / Denom
                print "%s, T = % .1f" % (name, T)
                print "Objective : "
                print ThisObj
                print "Gradient : "
                print ThisGrad
                print "Hessian : "
                print ThisHess
                Objs[T] = ThisObj
                GradMap.append(G)
                Objective += ThisObj
                Gradient += ThisGrad
                Hessian += ThisHess
        GradMapPrint = [' '.join(["#Temperature"] + self.FF.plist)]
        for T, g in zip(temps,GradMap):
            GradMapPrint.append(["%9.3f" % T] + ["% 9.3e" % i for i in g])
        o = open('gradient_%s.dat' % name,'w')
        for line in GradMapPrint:
            print >> o, ' '.join(line)
        o.close()
            
        Delta = np.array([calc[T] - exp[T] for T in temps])
        Defit = np.array([cfit[T] - exp[T] for T in temps])
        delt = {T : r for T, r in zip(temps,Delta)}
        dfit = {T : r for T, r in zip(temps,Defit)}
        #print_out = {'    %8.3f' % T:"%9.3f    %9.3f +- %-7.3f % 7.3f  %9.3f  % 7.3f" % (exp[T],calc[T],err[T],delt[T],cfit[T],dfit[T]) for T in calc}
        print_out = {'    %8.3f' % T:"%9.3f    %9.3f +- %-7.3f % 7.3f % 9.5f % 9.5f" % (exp[T],calc[T],err[T],delt[T],Weights[T],Objs[T]) for T in calc}
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

        Rho_wt = {243.2 : 0.005964, 247.2 : 0.007338, 250.8 : 0.008774, 253.9 : 0.010175, 256.8 : 0.011630, 
                  259.5 : 0.013116, 262.0 : 0.014606, 264.3 : 0.016076, 266.6 : 0.017641, 268.7 : 0.019153, 
                  270.9 : 0.020820, 273.2 : 0.022652, 275.1 : 0.024232, 277.2 : 0.026046, 279.4 : 0.028016, 
                  281.6 : 0.030053, 283.9 : 0.032248, 286.4 : 0.034697, 289.0 : 0.037303, 291.8 : 0.040156, 
                  294.9 : 0.043346, 298.2 : 0.046742, 302.3 : 0.050897, 306.8 : 0.055277, 312.2 : 0.060120, 
                  318.6 : 0.065013, 326.7 : 0.069442, 337.3 : 0.071582, 351.8 : 0.067340, 373.2 : 0.049548}
        
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

        Hvap_wt = {243.2 : 0.035362, 247.2 : 0.035538, 250.8 : 0.035653, 253.9 : 0.035718, 256.8 : 0.035752, 
                   259.5 : 0.035759, 262.0 : 0.035744, 264.3 : 0.035713, 266.6 : 0.035664, 268.7 : 0.035605, 
                   270.9 : 0.035528, 273.2 : 0.035431, 275.1 : 0.035339, 277.2 : 0.035224, 279.4 : 0.035088, 
                   281.6 : 0.034938, 283.9 : 0.034765, 286.4 : 0.034560, 289.0 : 0.034327, 291.8 : 0.034055, 
                   294.9 : 0.033728, 298.2 : 0.033351, 302.3 : 0.032844, 306.8 : 0.032240, 312.2 : 0.031454, 
                   318.6 : 0.030443, 326.7 : 0.029055, 337.3 : 0.027090, 351.8 : 0.024214, 373.2 : 0.019818}

        # Thermal expansion coefficients in units of 1e-4 K-1
        Alpha_exp = {243.2 :-13.945, 247.2 :-10.261, 250.8 : -7.859, 253.9 : -6.246, 256.8 : -5.009, 
                     259.5 : -4.034, 262.0 : -3.252, 264.3 : -2.614, 266.6 : -2.041, 268.7 : -1.565, 
                     270.9 : -1.109, 273.2 : -0.671, 275.1 : -0.336, 277.2 :  0.011, 279.4 :  0.350, 
                     281.6 :  0.667, 283.9 :  0.979, 286.4 :  1.297, 289.0 :  1.608, 291.8 :  1.923, 
                     294.9 :  2.250, 298.2 :  2.577, 302.3 :  2.957, 306.8 :  3.346, 312.2 :  3.780, 
                     318.6 :  4.257, 326.7 :  4.815, 337.3 :  5.487, 351.8 :  6.335, 373.2 :  7.504}
        
        Alpha_wt = {243.2 : 0.007502, 247.2 : 0.008895, 250.8 : 0.010310, 253.9 : 0.011657, 256.8 : 0.013030, 
                    259.5 : 0.014407, 262.0 : 0.015769, 264.3 : 0.017097, 266.6 : 0.018495, 268.7 : 0.019834, 
                    270.9 : 0.021299, 273.2 : 0.022897, 275.1 : 0.024267, 277.2 : 0.025832, 279.4 : 0.027525, 
                    281.6 : 0.029271, 283.9 : 0.031148, 286.4 : 0.033243, 289.0 : 0.035473, 291.8 : 0.037923, 
                    294.9 : 0.040679, 298.2 : 0.043643, 302.3 : 0.047328, 306.8 : 0.051320, 312.2 : 0.055934, 
                    318.6 : 0.060981, 326.7 : 0.066391, 337.3 : 0.071219, 351.8 : 0.072704, 373.2 : 0.063927}

        # Isothermal compressibilities in units of 1e-6 bar-1
        Kappa_exp = {243.2 : 76.292, 247.2 : 70.319, 250.8 : 66.225, 253.9 : 63.344, 256.8 : 61.040, 
                     259.5 : 59.161, 262.0 : 57.605, 264.3 : 56.304, 266.6 : 55.109, 268.7 : 54.100, 
                     270.9 : 53.115, 273.2 : 52.156, 275.1 : 51.411, 277.2 : 50.635, 279.4 : 49.868, 
                     281.6 : 49.143, 283.9 : 48.428, 286.4 : 47.694, 289.0 : 46.975, 291.8 : 46.245, 
                     294.9 : 45.645, 298.2 : 45.242, 302.3 : 44.841, 306.8 : 44.516, 312.2 : 44.268, 
                     318.6 : 44.151, 326.7 : 44.246, 337.3 : 44.733, 351.8 : 45.993, 373.2 : 49.027}

        Kappa_wt = {243.2 : 0.004143, 247.2 : 0.005334, 250.8 : 0.006631, 253.9 : 0.007939, 256.8 : 0.009337, 
                    259.5 : 0.010801, 262.0 : 0.012302, 264.3 : 0.013813, 266.6 : 0.015450, 268.7 : 0.017058, 
                    270.9 : 0.018858, 273.2 : 0.020866, 275.1 : 0.022621, 277.2 : 0.024659, 279.4 : 0.026899, 
                    281.6 : 0.029242, 283.9 : 0.031793, 286.4 : 0.034669, 289.0 : 0.037757, 291.8 : 0.041168, 
                    294.9 : 0.045010, 298.2 : 0.049123, 302.3 : 0.054171, 306.8 : 0.059481, 312.2 : 0.065288, 
                    318.6 : 0.070967, 326.7 : 0.075630, 337.3 : 0.076579, 351.8 : 0.068402, 373.2 : 0.044008}

        Kappa_wt1 = {243.2 : 0.0, 247.2 : 1.0, 250.8 : 1.0, 253.9 : 1.0, 256.8 : 1.0, 
                     259.5 : 1.0, 262.0 : 1.0, 264.3 : 1.0, 266.6 : 1.0, 268.7 : 1.0, 
                     270.9 : 1.0, 273.2 : 1.0, 275.1 : 1.0, 277.2 : 1.0, 279.4 : 1.0, 
                     281.6 : 1.0, 283.9 : 1.0, 286.4 : 1.0, 289.0 : 1.0, 291.8 : 1.0, 
                     294.9 : 1.0, 298.2 : 1.0, 302.3 : 1.0, 306.8 : 1.0, 312.2 : 1.0, 
                     318.6 : 1.0, 326.7 : 1.0, 337.3 : 1.0, 351.8 : 1.0, 373.2 : 1.0}

        # Isobaric heat capacities in units of cal mol-1 K-1
        Cp_exp = {243.2 : 20.292, 247.2 : 19.458, 250.8 : 18.988, 253.9 : 18.721, 256.8 : 18.549, 
                  259.5 : 18.435, 262.0 : 18.356, 264.3 : 18.299, 266.6 : 18.253, 268.7 : 18.218, 
                  270.9 : 18.186, 273.2 : 18.157, 275.1 : 18.136, 277.2 : 18.115, 279.4 : 18.095, 
                  281.6 : 18.077, 283.9 : 18.061, 286.4 : 18.045, 289.0 : 18.032, 291.8 : 18.020, 
                  294.9 : 18.010, 298.2 : 18.003, 302.3 : 17.997, 306.8 : 17.995, 312.2 : 17.996, 
                  318.6 : 18.000, 326.7 : 18.009, 337.3 : 18.027, 351.8 : 18.066, 373.2 : 18.152}
        
        Cp_wt = {243.2 : 0.009993, 247.2 : 0.011432, 250.8 : 0.012851, 253.9 : 0.014169, 256.8 : 0.015483, 
                 259.5 : 0.016777, 262.0 : 0.018037, 264.3 : 0.019247, 266.6 : 0.020505, 268.7 : 0.021696, 
                 270.9 : 0.022985, 273.2 : 0.024376, 275.1 : 0.025558, 277.2 : 0.026897, 279.4 : 0.028335, 
                 281.6 : 0.029806, 283.9 : 0.031376, 286.4 : 0.033118, 289.0 : 0.034961, 291.8 : 0.036977, 
                 294.9 : 0.039237, 298.2 : 0.041662, 302.3 : 0.044680, 306.8 : 0.047964, 312.2 : 0.051806, 
                 318.6 : 0.056117, 326.7 : 0.061004, 337.3 : 0.066045, 351.8 : 0.069696, 373.2 : 0.067209}

        # Sorted list of temperatures.
        Temps = np.array(sorted([i for i in Rho_exp]))

        if Counter() == 0 and self.manual:
            warn_press_key("Now's our chance to fill the temp directory up with data!")

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

        Rhos, Vols, Hs, pVs, Energies, Grads, mEnergies, mGrads, Rho_errs, Hvap_errs, Alpha_errs, Kappa_errs, Cp_errs = ([Results[t][i] for t in range(len(Temps))] for i in range(13))

        R  = np.array(list(itertools.chain(*list(Rhos))))
        V  = np.array(list(itertools.chain(*list(Vols))))
        H  = np.array(list(itertools.chain(*list(Hs))))
        PV = np.array(list(itertools.chain(*list(pVs))))
        E  = np.array(list(itertools.chain(*list(Energies))))
        G  = np.hstack(tuple(Grads))
        mE = np.array(list(itertools.chain(*list(mEnergies))))
        mG = np.hstack(tuple(mGrads))
        NMol = 216 # Number of molecules

        Rho_calc = {}
        Rho_grad = {}
        Rho_std  = {}
        Hvap_calc = {}
        Hvap_grad = {}
        Hvap_std  = {}
        Alpha_calc = {}
        Alpha_grad = {}
        Alpha_std  = {}
        Kappa_calc = {}
        Kappa_grad = {}
        Kappa_std  = {}
        Cp_calc = {}
        Cp_grad = {}
        Cp_std  = {}

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
                # The correct Boltzmann factors include PV.
                U_kln[k, m, :]   = Energies[k] + pVs[k]
                U_kln[k, m,: ]  *= beta
                mU_kln[k, m, :]  = mEnergies[k]
                mU_kln[k, m, :] *= beta
        print "Running MBAR analysis..."
        mbar = pymbar.MBAR(U_kln, N_k, verbose=False, relative_tolerance=5.0e-8)
        W1 = mbar.getWeights()
        if np.abs(np.std(mEnergies)) > 1e-6:
            mmbar = pymbar.MBAR(mU_kln, mN_k, verbose=False, relative_tolerance=5.0e-8)
            mW1 = mmbar.getWeights()
        else:
            mW1 = np.ones((Sims*mShots,Sims),dtype=float)
            mW1 /= Sims*mShots
        for i, T in enumerate(Temps):
            # The weights that we want are the last ones.
            W = flat(W1[:,i])
            C = weight_info(W, T, N_k, verbose=False)
            mW = flat(mW1[:,i])
            Gbar = flat(np.mat(G)*col(W))
            mGbar = flat(np.mat(mG)*col(mW))
            mBeta = -1/kb/T
            Beta  = 1/kb/T
            kT    = kb*T
            # Define some things to make the analytic derivatives easier.
            def avg(vec):
                return np.dot(W,vec)
            def covde(vec):
                return flat(np.mat(G)*col(W*vec)) - avg(vec)*Gbar
            ## Density.
            Rho_calc[T]   = np.dot(W,R)
            Rho_grad[T]   = mBeta*(flat(np.mat(G)*col(W*R)) - np.dot(W,R)*Gbar)
            ## Enthalpy of vaporization.
            Hvap_calc[T]  = np.dot(mW,mE) - np.dot(W,E)/NMol + kb*T - np.dot(W, PV)/NMol
            Hvap_grad[T]  = mGbar + mBeta*(flat(np.mat(mG)*col(mW*mE)) - np.dot(mW,mE)*mGbar)
            Hvap_grad[T] -= (Gbar + mBeta*(flat(np.mat(G)*col(W*E)) - np.dot(W,E)*Gbar)) / NMol
            Hvap_grad[T] -= (mBeta*(flat(np.mat(G)*col(W*PV)) - np.dot(W,PV)*Gbar)) / NMol
            ## Thermal expansion coefficient.
            Alpha_calc[T] = 1e4 * (avg(H*V)-avg(H)*avg(V))/avg(V)/(kT*T)
            GAlpha1 = mBeta * covde(H*V) / avg(V)
            GAlpha2 = Beta * avg(H*V) * covde(V) / avg(V)**2
            #GAlpha3 = flat(np.mat(G)*col(V))/N/avg(V) - Gbar
            GAlpha3 = flat(np.mat(G)*col(W*V))/avg(V) - Gbar
            GAlpha4 = Beta * covde(H)
            Alpha_grad[T] = 1e4 * (GAlpha1 + GAlpha2 + GAlpha3 + GAlpha4)/(kT*T)
            ## Isothermal compressibility.
            bar_unit = 0.06022141793 * 1e6
            Kappa_calc[T] = bar_unit / kT * (avg(V**2)-avg(V)**2)/avg(V)
            GKappa1 = -1 * Beta**2 * avg(V) * covde(V**2) / avg(V)**2
            GKappa2 = +1 * Beta**2 * avg(V**2) * covde(V) / avg(V)**2
            GKappa3 = +1 * Beta**2 * covde(V)
            Kappa_grad[T] = bar_unit*(GKappa1 + GKappa2 + GKappa3)
            ## Isobaric heat capacity.
            Cp_calc[T] = 1000/(4.184*NMol*kT*T) * (avg(H**2) - avg(H)**2)
            GCp1 = 2*covde(H) * 1000 / 4.184 / (NMol*kT*T)
            GCp2 = mBeta*covde(H**2) * 1000 / 4.184 / (NMol*kT*T)
            GCp3 = 2*Beta*avg(H)*covde(H) * 1000 / 4.184 / (NMol*kT*T)
            Cp_grad[T] = GCp1 + GCp2 + GCp3
            ## Estimation of errors.
            Rho_std[T]    = np.sqrt(sum(C**2 * np.array(Rho_errs)**2))
            Hvap_std[T]   = np.sqrt(sum(C**2 * np.array(Hvap_errs)**2))
            Alpha_std[T]   = np.sqrt(sum(C**2 * np.array(Alpha_errs)**2)) * 1e4
            Kappa_std[T]   = np.sqrt(sum(C**2 * np.array(Kappa_errs)**2)) * 1e6
            Cp_std[T]   = np.sqrt(sum(C**2 * np.array(Cp_errs)**2))

        # Get contributions to the objective function
        X_Rho, G_Rho, H_Rho, RhoPrint = self.objective_term(Temps, Rho_exp, Rho_calc, Rho_std, Rho_grad, 3, name="Density", verbose=False)
        X_Hvap, G_Hvap, H_Hvap, HvapPrint = self.objective_term(Temps, Hvap_exp, Hvap_calc, Hvap_std, Hvap_grad, 2, name="H_vap", verbose=False)
        X_Alpha, G_Alpha, H_Alpha, AlphaPrint = self.objective_term(Temps, Alpha_exp, Alpha_calc, Alpha_std, Alpha_grad, 2, name="Thermal Expansion", verbose=False, Denom=1.0)
        X_Kappa, G_Kappa, H_Kappa, KappaPrint = self.objective_term(Temps, Kappa_exp, Kappa_calc, Kappa_std, Kappa_grad, 2, name="Compressibility", verbose=False, Denom=10.0, Weights=Kappa_wt1)
        X_Cp, G_Cp, H_Cp, CpPrint = self.objective_term(Temps, Cp_exp, Cp_calc, Cp_std, Cp_grad, 2, name="Heat Capacity", verbose=False, Denom=1.0)

        Gradient = np.zeros(self.FF.np, dtype=float)
        Hessian = np.zeros((self.FF.np,self.FF.np),dtype=float)

        w_tot = self.W_Rho + self.W_Hvap + self.W_Alpha + self.W_Kappa + self.W_Cp
        w_1 = self.W_Rho / w_tot
        w_2 = self.W_Hvap / w_tot
        w_3 = self.W_Alpha / w_tot
        w_4 = self.W_Kappa / w_tot
        w_5 = self.W_Cp / w_tot

        Objective    = w_1 * X_Rho + w_2 * X_Hvap + w_3 * X_Alpha + w_4 * X_Kappa + w_5 * X_Cp
        if AGrad:
            Gradient = w_1 * G_Rho + w_2 * G_Hvap + w_3 * G_Alpha + w_4 * G_Kappa + w_5 * G_Cp
        if AHess:
            Hessian  = w_1 * H_Rho + w_2 * H_Hvap + w_3 * H_Alpha + w_4 * H_Kappa + w_5 * H_Cp

        printcool_dictionary(RhoPrint, title='Density vs T (kg m^-3) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Density objective function: % .3f, Derivative:" % X_Rho)
        self.FF.print_map(vals=G_Rho)
        print bar

        printcool_dictionary(HvapPrint, title='Enthalpy of Vaporization vs T (kJ mol^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("H_vap objective function: % .3f, Derivative:" % X_Hvap)
        self.FF.print_map(vals=G_Hvap)
        print bar

        printcool_dictionary(AlphaPrint,title='Thermal Expansion Coefficient vs T (10^-4 K^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Thermal Expansion objective function: % .3f, Derivative:" % X_Alpha)
        self.FF.print_map(vals=G_Alpha)
        print bar

        printcool_dictionary(KappaPrint,title='Isothermal Compressibility vs T (10^-6 bar^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Compressibility objective function: % .3f, Derivative:" % X_Kappa)
        self.FF.print_map(vals=G_Kappa)
        print bar

        printcool_dictionary(CpPrint,   title='Isobaric Heat Capacity vs T (cal mol^-1 K^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Heat Capacity objective function: % .3f, Derivative:" % X_Cp)
        self.FF.print_map(vals=G_Cp)
        print bar

        Answer = {'X':Objective, 'G':Gradient, 'H':Hessian}
        return Answer

