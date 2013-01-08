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
        print "MBAR Results for Temperature % .2f, Box, Contributions:" % T
        print C
        print "InfoContent: % .2f snapshots (%.2f %%)" % (I, 100*I/len(W))
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
        # Don't target the average enthalpy of vaporization and allow it to freely float (experimental)
        self.set_option(tgt_opts,'hvap_subaverage','hvap_subaverage')
        
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

    def objective_term(self,temps, exp, calc, err, grad, fitorder, name="Quantity", verbose = True, FitFitted = False, Denom=None, Weights=None, SubAverage=False):
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
        tarray /= (max(temps)-min(temps))
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
        avgCalc = 0.0
        avgExp  = 0.0
        avgGrad = np.zeros(self.FF.np, dtype=float)
        for i, T in enumerate(temps):
            avgCalc += Weights[T]*calc[T]
            avgExp  += Weights[T]*exp[T]
            avgGrad += Weights[T]*grad[T]
        for i, T in enumerate(temps):
            if FitFitted:
                G = flat(GradZMat[i,:])
                Delta = cfit[T] - exp[T]
            elif SubAverage:
                G = grad[T]-avgGrad
                Delta = calc[T] - exp[T] - avgCalc + avgExp
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
                #     print "%s, temp % .2f : abs(calc-exp) = % .2e (< % .2e)" % (name,T,np.abs(Delta),D)
                # else:
                #     print "%s, temp % .2f : abs(calc-exp) = % .2e ;" % (name,T,np.abs(Delta)),
                #     print "recommend trust <", Delta/np.abs(G)
                ThisObj  = Weights[T] * (S**0.5-D) / Denom
                ThisGrad = Weights[T] * (Delta/S**0.5) * G / Denom
                ThisHess = Weights[T] * (1/S**0.5-Delta**2/S**1.5) * np.outer(G,G) / Denom
                print "%s, T = % .2f" % (name, T)
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

        data = "new"

        if data == "tiny":
            Rho_exp = {297.0 : 997.333, 297.1 : 997.309, 297.2 : 997.284, 297.3 : 997.259, 297.4 : 997.235, 297.5 : 997.210,
                       297.6 : 997.184, 297.7 : 997.159, 297.8 : 997.134, 297.9 : 997.109, 298.0 : 997.083, 298.1 : 997.058,
                       298.2 : 997.032, 298.3 : 997.006, 298.4 : 996.980, 298.5 : 996.955, 298.6 : 996.929, 298.7 : 996.902,
                       298.8 : 996.876, 298.9 : 996.850, 299.0 : 996.823, 299.1 : 996.797, 299.2 : 996.770, 299.3 : 996.744,
                       299.4 : 996.717, 299.5 : 996.690, 299.6 : 996.663, 299.7 : 996.636, 299.8 : 996.609, 299.9 : 996.582, }
            Hvap_exp = {297.0 : 44.039, 297.1 : 44.034, 297.2 : 44.030, 297.3 : 44.026, 297.4 : 44.022, 297.5 : 44.017,
                        297.6 : 44.013, 297.7 : 44.009, 297.8 : 44.004, 297.9 : 44.000, 298.0 : 43.996, 298.1 : 43.992,
                        298.2 : 43.987, 298.3 : 43.983, 298.4 : 43.979, 298.5 : 43.974, 298.6 : 43.970, 298.7 : 43.966,
                        298.8 : 43.962, 298.9 : 43.957, 299.0 : 43.953, 299.1 : 43.949, 299.2 : 43.944, 299.3 : 43.940,
                        299.4 : 43.936, 299.5 : 43.932, 299.6 : 43.927, 299.7 : 43.923, 299.8 : 43.919, 299.9 : 43.915, }
            Alpha_exp = {297.0 : 2.460, 297.1 : 2.470, 297.2 : 2.480, 297.3 : 2.490, 297.4 : 2.499, 297.5 : 2.509,
                         297.6 : 2.519, 297.7 : 2.529, 297.8 : 2.538, 297.9 : 2.548, 298.0 : 2.558, 298.1 : 2.567,
                         298.2 : 2.577, 298.3 : 2.586, 298.4 : 2.596, 298.5 : 2.606, 298.6 : 2.615, 298.7 : 2.625,
                         298.8 : 2.634, 298.9 : 2.644, 299.0 : 2.653, 299.1 : 2.663, 299.2 : 2.672, 299.3 : 2.681,
                         299.4 : 2.691, 299.5 : 2.700, 299.6 : 2.710, 299.7 : 2.719, 299.8 : 2.728, 299.9 : 2.738, }
            Kappa_exp = {297.0 : 45.380, 297.1 : 45.368, 297.2 : 45.356, 297.3 : 45.344, 297.4 : 45.333, 297.5 : 45.321,
                         297.6 : 45.309, 297.7 : 45.298, 297.8 : 45.287, 297.9 : 45.275, 298.0 : 45.264, 298.1 : 45.253,
                         298.2 : 45.242, 298.3 : 45.231, 298.4 : 45.220, 298.5 : 45.209, 298.6 : 45.198, 298.7 : 45.187,
                         298.8 : 45.176, 298.9 : 45.166, 299.0 : 45.155, 299.1 : 45.144, 299.2 : 45.134, 299.3 : 45.124,
                         299.4 : 45.113, 299.5 : 45.103, 299.6 : 45.093, 299.7 : 45.083, 299.8 : 45.073, 299.9 : 45.063, }
            Cp_exp = {297.0 : 18.005, 297.1 : 18.005, 297.2 : 18.005, 297.3 : 18.004, 297.4 : 18.004, 297.5 : 18.004,
                      297.6 : 18.004, 297.7 : 18.004, 297.8 : 18.003, 297.9 : 18.003, 298.0 : 18.003, 298.1 : 18.003,
                      298.2 : 18.003, 298.3 : 18.003, 298.4 : 18.002, 298.5 : 18.002, 298.6 : 18.002, 298.7 : 18.002,
                      298.8 : 18.002, 298.9 : 18.001, 299.0 : 18.001, 299.1 : 18.001, 299.2 : 18.001, 299.3 : 18.001,
                      299.4 : 18.001, 299.5 : 18.001, 299.6 : 18.000, 299.7 : 18.000, 299.8 : 18.000, 299.9 : 18.000, }
            Weights = None
        elif data == "new":
            Rho_exp = {249.15 : 990.497, 253.15 : 993.547, 257.15 : 995.816, 261.15 : 997.476, 265.15 : 998.647, 269.15 : 999.414, 273.15 : 999.840, 277.15 : 999.972, 
                       281.15 : 999.848, 285.15 : 999.497, 289.15 : 998.943, 293.15 : 998.204, 298.15 : 997.045, 301.15 : 996.234, 305.15 : 995.026, 309.15 : 993.684, 
                       313.15 : 992.216, 317.15 : 990.628, 321.15 : 988.927, 325.15 : 987.119, 329.15 : 985.208, 333.15 : 983.199, 337.15 : 981.095, 341.15 : 978.900, 
                       345.15 : 976.617, 349.15 : 974.249, 353.15 : 971.798, 357.15 : 969.266, 361.15 : 966.655, 365.15 : 963.966, 369.15 : 961.202, 373.15 : 958.364, }
            Hvap_exp = {249.15 : 46.103, 253.15 : 45.919, 257.15 : 45.740, 261.15 : 45.565, 265.15 : 45.393, 269.15 : 45.223, 273.15 : 45.053, 277.15 : 44.884, 
                        281.15 : 44.714, 285.15 : 44.544, 289.15 : 44.374, 293.15 : 44.203, 298.15 : 43.989, 301.15 : 43.861, 305.15 : 43.689, 309.15 : 43.517, 
                        313.15 : 43.345, 317.15 : 43.172, 321.15 : 42.999, 325.15 : 42.826, 329.15 : 42.652, 333.15 : 42.476, 337.15 : 42.300, 341.15 : 42.123, 
                        345.15 : 41.943, 349.15 : 41.762, 353.15 : 41.580, 357.15 : 41.396, 361.15 : 41.210, 365.15 : 41.024, 369.15 : 40.837, 373.15 : 40.652, }
            Alpha_exp = {249.15 : -8.877, 253.15 : -6.606, 257.15 : -4.874, 261.15 : -3.506, 265.15 : -2.395, 269.15 : -1.469, 273.15 : -0.680, 277.15 : 0.003, 
                         281.15 : 0.604, 285.15 : 1.141, 289.15 : 1.626, 293.15 : 2.068, 298.15 : 2.572, 301.15 : 2.853, 305.15 : 3.206, 309.15 : 3.539, 
                         313.15 : 3.853, 317.15 : 4.152, 321.15 : 4.438, 325.15 : 4.712, 329.15 : 4.975, 333.15 : 5.231, 337.15 : 5.478, 341.15 : 5.719, 
                         345.15 : 5.954, 349.15 : 6.185, 353.15 : 6.411, 357.15 : 6.633, 361.15 : 6.853, 365.15 : 7.071, 369.15 : 7.287, 373.15 : 7.501, }
            Kappa_exp = {249.15 : 67.985, 253.15 : 63.996, 257.15 : 60.783, 261.15 : 58.116, 265.15 : 55.851, 269.15 : 53.893, 273.15 : 52.176, 277.15 : 50.653, 
                         281.15 : 49.288, 285.15 : 48.056, 289.15 : 46.934, 293.15 : 45.892, 298.15 : 45.247, 301.15 : 44.943, 305.15 : 44.622, 309.15 : 44.390, 
                         313.15 : 44.239, 317.15 : 44.162, 321.15 : 44.153, 325.15 : 44.209, 329.15 : 44.324, 333.15 : 44.496, 337.15 : 44.723, 341.15 : 45.003, 
                         345.15 : 45.333, 349.15 : 45.714, 353.15 : 46.143, 357.15 : 46.621, 361.15 : 47.148, 365.15 : 47.722, 369.15 : 48.346, 373.15 : 49.019, }
            Cp_exp = {249.15 : 19.177, 253.15 : 18.777, 257.15 : 18.532, 261.15 : 18.380, 265.15 : 18.280, 269.15 : 18.210, 273.15 : 18.157, 277.15 : 18.115, 
                      281.15 : 18.080, 285.15 : 18.052, 289.15 : 18.030, 293.15 : 18.015, 298.15 : 18.002, 301.15 : 17.998, 305.15 : 17.995, 309.15 : 17.995, 
                      313.15 : 17.996, 317.15 : 17.999, 321.15 : 18.002, 325.15 : 18.006, 329.15 : 18.011, 333.15 : 18.018, 337.15 : 18.026, 341.15 : 18.035, 
                      345.15 : 18.046, 349.15 : 18.058, 353.15 : 18.071, 357.15 : 18.086, 361.15 : 18.101, 365.15 : 18.117, 369.15 : 18.134, 373.15 : 18.151, }
            # The lowest temperature point is too noisy so we cut it out.
            Weights = {249.15 : 0.0, 253.15 : 1.0, 257.15 : 1.0, 261.15 : 1.0, 265.15 : 1.0, 269.15 : 1.0, 273.15 : 1.0, 277.15 : 1.0, 
                       281.15 : 1.0, 285.15 : 1.0, 289.15 : 1.0, 293.15 : 1.0, 298.15 : 1.0, 301.15 : 1.0, 305.15 : 1.0, 309.15 : 1.0, 
                       313.15 : 1.0, 317.15 : 1.0, 321.15 : 1.0, 325.15 : 1.0, 329.15 : 1.0, 333.15 : 1.0, 337.15 : 1.0, 341.15 : 1.0, 
                       345.15 : 1.0, 349.15 : 1.0, 353.15 : 1.0, 357.15 : 1.0, 361.15 : 1.0, 365.15 : 1.0, 369.15 : 1.0, 373.15 : 1.0, }
            # Rho_exp = {253.15 : 993.547, 257.16 : 995.821, 261.16 : 997.480, 265.16 : 998.650, 269.16 : 999.415, 273.15 : 999.840, 
            #            277.15 : 999.972, 281.14 : 999.849, 285.14 : 999.499, 289.14 : 998.945, 293.15 : 998.204, 298.15 : 997.045, 
            #            301.20 : 996.219, 305.25 : 994.994, 309.30 : 993.631, 313.37 : 992.131, 317.46 : 990.500, 321.57 : 988.742, 
            #            325.70 : 986.862, 329.86 : 984.859, 334.04 : 982.739, 338.24 : 980.506, 342.48 : 978.151, 346.75 : 975.680, 
            #            351.05 : 973.095, 355.39 : 970.390, 359.76 : 967.571, 364.18 : 964.625, 368.64 : 961.559, 373.15 : 958.364, }
            # Hvap_exp = {253.15 : 45.919, 257.16 : 45.739, 261.16 : 45.565, 265.16 : 45.393, 269.16 : 45.222, 273.15 : 45.053, 
            #             277.15 : 44.884, 281.14 : 44.715, 285.14 : 44.545, 289.14 : 44.374, 293.15 : 44.203, 298.15 : 43.989, 
            #             301.20 : 43.859, 305.25 : 43.685, 309.30 : 43.511, 313.37 : 43.335, 317.46 : 43.159, 321.57 : 42.981, 
            #             325.70 : 42.802, 329.86 : 42.621, 334.04 : 42.437, 338.24 : 42.252, 342.48 : 42.063, 346.75 : 41.871, 
            #             351.05 : 41.676, 355.39 : 41.477, 359.76 : 41.275, 364.18 : 41.069, 368.64 : 40.861, 373.15 : 40.652, }
            # Alpha_exp = {253.15 : -6.606, 257.16 : -4.870, 261.16 : -3.503, 265.16 : -2.392, 269.16 : -1.467, 273.15 : -0.680, 
            #              277.15 : 0.003, 281.14 : 0.603, 285.14 : 1.140, 289.14 : 1.624, 293.15 : 2.068, 298.15 : 2.572, 
            #              301.20 : 2.858, 305.25 : 3.215, 309.30 : 3.551, 313.37 : 3.870, 317.46 : 4.175, 321.57 : 4.467, 
            #              325.70 : 4.748, 329.86 : 5.021, 334.04 : 5.286, 338.24 : 5.544, 342.48 : 5.798, 346.75 : 6.047, 
            #              351.05 : 6.293, 355.39 : 6.536, 359.76 : 6.777, 364.18 : 7.018, 368.64 : 7.259, 373.15 : 7.501, }
            # Kappa_exp = {253.15 : 63.996, 257.16 : 60.776, 261.16 : 58.110, 265.16 : 55.845, 269.16 : 53.888, 273.15 : 52.176, 
            #              277.15 : 50.653, 281.14 : 49.291, 285.14 : 48.059, 289.14 : 46.937, 293.15 : 45.892, 298.15 : 45.247, 
            #              301.20 : 44.938, 305.25 : 44.615, 309.30 : 44.383, 313.37 : 44.233, 317.46 : 44.159, 321.57 : 44.156, 
            #              325.70 : 44.221, 329.86 : 44.351, 334.04 : 44.542, 338.24 : 44.794, 342.48 : 45.107, 346.75 : 45.480, 
            #              351.05 : 45.912, 355.39 : 46.405, 359.76 : 46.959, 364.18 : 47.578, 368.64 : 48.264, 373.15 : 49.019, }
            # Cp_exp = {253.15 : 18.777, 257.16 : 18.532, 261.16 : 18.379, 265.16 : 18.280, 269.16 : 18.210, 273.15 : 18.157, 
            #           277.15 : 18.115, 281.14 : 18.080, 285.14 : 18.052, 289.14 : 18.030, 293.15 : 18.015, 298.15 : 18.002, 
            #           301.20 : 17.998, 305.25 : 17.995, 309.30 : 17.995, 313.37 : 17.996, 317.46 : 17.999, 321.57 : 18.002, 
            #           325.70 : 18.007, 329.86 : 18.013, 334.04 : 18.020, 338.24 : 18.028, 342.48 : 18.038, 346.75 : 18.050, 
            #           351.05 : 18.064, 355.39 : 18.079, 359.76 : 18.096, 364.18 : 18.113, 368.64 : 18.131, 373.15 : 18.151, }
        elif data == "old":
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
            Weights = None

        # Sorted list of temperatures.
        Temps = np.array(sorted([i for i in Rho_exp]))

        if Counter() == 0 and self.manual:
            warn_press_key("Now's our chance to fill the temp directory up with data!")

        # Launch a series of simulations
        for T in Temps:
            if not os.path.exists('%.2f' % T):
                os.makedirs('%.2f' % T)
            os.chdir('%.2f' % T)
            self.npt_simulation(T)
            os.chdir('..')

        # Wait for simulations to finish
        wq_wait(self.wq)

        # Gather the calculation data
        Results = {t : lp_load(open('./%.2f/npt_result.p' % T)) for t, T in enumerate(Temps)}

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
        X_Rho, G_Rho, H_Rho, RhoPrint = self.objective_term(Temps, Rho_exp, Rho_calc, Rho_std, Rho_grad, 3, name="Density", verbose=False, Denom=5.0, Weights=Weights)
        X_Hvap, G_Hvap, H_Hvap, HvapPrint = self.objective_term(Temps, Hvap_exp, Hvap_calc, Hvap_std, Hvap_grad, 2, name="H_vap", verbose=False, Denom=2.0, Weights=Weights, SubAverage=self.hvap_subaverage)
        X_Alpha, G_Alpha, H_Alpha, AlphaPrint = self.objective_term(Temps, Alpha_exp, Alpha_calc, Alpha_std, Alpha_grad, 2, name="Thermal Expansion", verbose=False, Denom=1.0, Weights=Weights)
        X_Kappa, G_Kappa, H_Kappa, KappaPrint = self.objective_term(Temps, Kappa_exp, Kappa_calc, Kappa_std, Kappa_grad, 2, name="Compressibility", verbose=False, Denom=10.0, Weights=Weights)
        X_Cp, G_Cp, H_Cp, CpPrint = self.objective_term(Temps, Cp_exp, Cp_calc, Cp_std, Cp_grad, 2, name="Heat Capacity", verbose=False, Denom=1.0, Weights=Weights)

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

        PrintDict = OrderedDict()
        Title = "Condensed Phase Properties:\n %-20s %40s" % ("Property Name", "Residual x Weight = Contribution")
        printcool_dictionary(RhoPrint, title='Density vs T (kg m^-3) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Density objective function: % .3f, Derivative:" % X_Rho)
        self.FF.print_map(vals=G_Rho)
        print bar
        PrintDict['Density'] = "% 10.5f % 8.3f % 14.5e" % (X_Rho, w_1, X_Rho*w_1)

        printcool_dictionary(HvapPrint, title='Enthalpy of Vaporization vs T (kJ mol^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("H_vap objective function: % .3f, Derivative:" % X_Hvap)
        self.FF.print_map(vals=G_Hvap)
        print bar
        PrintDict['Enthalpy of Vaporization'] = "% 10.5f % 8.3f % 14.5e" % (X_Hvap, w_2, X_Hvap*w_2)

        printcool_dictionary(AlphaPrint,title='Thermal Expansion Coefficient vs T (10^-4 K^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Thermal Expansion objective function: % .3f, Derivative:" % X_Alpha)
        self.FF.print_map(vals=G_Alpha)
        print bar
        PrintDict['Thermal Expansion Coefficient'] = "% 10.5f % 8.3f % 14.5e" % (X_Alpha, w_3, X_Alpha*w_3)

        printcool_dictionary(KappaPrint,title='Isothermal Compressibility vs T (10^-6 bar^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Compressibility objective function: % .3f, Derivative:" % X_Kappa)
        self.FF.print_map(vals=G_Kappa)
        print bar
        PrintDict['Isothermal Compressibility'] = "% 10.5f % 8.3f % 14.5e" % (X_Kappa, w_4, X_Kappa*w_4)

        printcool_dictionary(CpPrint,   title='Isobaric Heat Capacity vs T (cal mol^-1 K^-1) \nTemperature  Reference  Calculated +- Stdev     Delta    Weight    Term   ',bold=True,color=3,keywidth=15)
        bar = printcool("Heat Capacity objective function: % .3f, Derivative:" % X_Cp)
        self.FF.print_map(vals=G_Cp)
        print bar
        PrintDict['Isobaric Heat Capacity'] = "% 10.5f % 8.3f % 14.5e" % (X_Cp, w_5, X_Cp*w_5)

        PrintDict['Total'] = "% 10s % 8s % 14.5e" % ("","",Objective)

        printcool_dictionary(PrintDict,color=4,title=Title,keywidth=31)

        Answer = {'X':Objective, 'G':Gradient, 'H':Hessian}
        return Answer

