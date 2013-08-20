""" @package forcebalance.optimizer Optimization algorithms.

My current implementation is to have a single optimizer class with several methods
contained inside.

@author Lee-Ping Wang
@date 12/2011

"""

import os, pickle, re, sys
import numpy as np
from copy import deepcopy
from numpy.linalg import eig, norm, solve
import forcebalance
from forcebalance.nifty import col, flat, row, printcool, printcool_dictionary, pvec1d, pmat2d, warn_press_key, invert_svd
from forcebalance.finite_difference import f1d7p, f1d5p, fdwrap
from collections import OrderedDict
import random
from forcebalance.output import getLogger, DEBUG
logger = getLogger(__name__)

# Global variable corresponding to the iteration number.
ITERATION_NUMBER = 0
# Global variable corresponding to whether the optimization took a good step.
GOODSTEP = 0

def Counter():
    global ITERATION_NUMBER
    return ITERATION_NUMBER

def GoodStep():
    global GOODSTEP
    return GOODSTEP

class Optimizer(forcebalance.BaseClass):
    """ Optimizer class.  Contains several methods for numerical optimization.

    For various reasons, the optimizer depends on the force field and fitting
    targets (i.e. we cannot treat it as a fully independent numerical optimizer).
    The dependency is rather weak which suggests that I can remove it someday.
    """
        
    def __init__(self,options,Objective,FF):
        """ Create an Optimizer object.
        
        The optimizer depends on both the FF and the fitting targets so there
        is a chain of dependencies: FF --> FitSim --> Optimizer, and FF --> Optimizer
        
        Here's what we do:
        - Take options from the parser
        - Pass in the objective function, force field, all fitting targets

        """
        super(Optimizer, self).__init__(options)

        ## A list of all the things we can ask the optimizer to do.
        self.OptTab    = {'NEWTONRAPHSON'     : self.NewtonRaphson, 
                          'NEWTON'            : self.NewtonRaphson, 
                          'NR'                : self.NewtonRaphson, 
                          'BFGS'              : self.BFGS,
                          'POWELL'            : self.Powell,
                          'SIMPLEX'           : self.Simplex,
                          'ANNEAL'            : self.Anneal,
                          'GENETIC'           : self.GeneticAlgorithm,
                          'CONJUGATEGRADIENT' : self.ConjugateGradient,
                          'SCAN_MVALS'        : self.ScanMVals,
                          'SCAN_PVALS'        : self.ScanPVals,
                          'SINGLE'            : self.SinglePoint,
                          'GRADIENT'          : self.Gradient,
                          'HESSIAN'           : self.Hessian,
                          'FDCHECKG'          : self.FDCheckG,
                          'FDCHECKH'          : self.FDCheckH
                          }
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        ## The root directory
        self.set_option(options,'root','root')
        ## The job type
        self.set_option(options,'jobtype','jobtype')
        ## Initial step size trust radius
        self.set_option(options,'trust0','trust0')
        ## Minimum trust radius (for noisy objective functions)
        self.set_option(options,'mintrust','mintrust')
        ## Lower bound on Hessian eigenvalue (below this, we add in steepest descent)
        self.set_option(options,'eig_lowerbound','eps')
        ## Guess value for Brent
        self.set_option(options,'lm_guess','lmg')
        ## Step size for numerical finite difference
        self.set_option(options,'finite_difference_h','h')
        ## Number of steps to average over
        self.set_option(options,'objective_history','hist')
        ## Function value convergence threshold
        self.set_option(options,'convergence_objective','conv_obj')
        ## Step size convergence threshold
        self.set_option(options,'convergence_step','conv_stp')
        ## Gradient convergence threshold
        self.set_option(options,'convergence_gradient','conv_grd')
        ## Maximum number of optimization steps
        self.set_option(options,'maxstep','maxstep')
        ## For scan[mp]vals: The parameter index to scan over
        self.set_option(options,'scanindex_num','idxnum')
        ## For scan[mp]vals: The parameter name to scan over, it just looks up an index
        self.set_option(options,'scanindex_name','idxname')
        ## For scan[mp]vals: The values that are fed into the scanner
        self.set_option(options,'scan_vals','scan_vals')
        ## Name of the checkpoint file that we're reading in
        self.set_option(options,'readchk','rchk_fnm')
        ## Name of the checkpoint file that we're writing out
        self.set_option(options,'writechk','wchk_fnm')
        ## Whether to write the checkpoint file at every step
        self.set_option(options,'writechk_step','wchk_step')
        ## Adaptive trust radius adjustment factor
        self.set_option(options,'adaptive_factor','adapt_fac')
        ## Adaptive trust radius adjustment damping
        self.set_option(options,'adaptive_damping','adapt_damp')
        ## Whether to print gradient during each step of the optimization
        self.set_option(options,'print_gradient','print_grad')
        ## Whether to print Hessian during each step of the optimization
        self.set_option(options,'print_hessian','print_hess')
        ## Whether to print parameters during each step of the optimization
        self.set_option(options,'print_parameters','print_vals')
        ## Error tolerance (if objective function rises by less than this, then the optimizer will forge ahead!)
        self.set_option(options,'error_tolerance','err_tol')
        ## Search tolerance (The nonlinear search will stop if the change is below this threshold)
        self.set_option(options,'search_tolerance','search_tol')
        self.set_option(options,'read_mvals')
        self.set_option(options,'read_pvals')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The objective function (needs to pass in when I instantiate)
        self.Objective = Objective
        ## Whether the penalty function is hyperbolic
        self.bhyp      = Objective.Penalty.ptyp != 2
        ## The force field itself
        self.FF        = FF
        
        #======================================#
        #    Variables from the force field    #
        #======================================#
        ## The indices to be excluded from the Hessian update
        self.excision  = list(FF.excision)

        ## Number of parameters
        self.np        = FF.np
        ## The original parameter values
        if options['read_mvals'] != None:
            self.mvals0    = np.array(options['read_mvals'])
        elif options['read_pvals'] != None:
            self.mvals0    = FF.create_mvals(options['read_pvals'])
        else:
            self.mvals0    = np.zeros(self.FF.np)

        ## Print the optimizer options.
        printcool_dictionary(self.PrintOptionDict, title="Setup for optimizer")
        ## Load the checkpoint file.
        self.readchk()
        
    def Run(self):
        """ Call the appropriate optimizer.  This is the method we might want to call from an executable. """

        xk = self.OptTab[self.jobtype]()
        
        ## Sometimes the optimizer doesn't return anything (i.e. in the case of a single point calculation)
        ## In these situations, don't do anything
        if xk == None: return

        ## Check derivatives by finite difference after the optimization is over (for good measure)
        check_after = False
        if check_after:
            self.mvals0 = xk.copy()
            self.FDCheckG()

        ## Print out final answer
        final_print = True
        if final_print:
            bar = printcool("Final optimization parameters:\n Paste to input file to restart",bold=True,color=4)
            logger.info("read_mvals\n")
            self.FF.print_map(xk)
            logger.info("/read_mvals\n")
            bar = printcool("Final physical parameters:",bold=True,color=4)
            self.FF.print_map(self.FF.create_pvals(xk))
            logger.info(bar)
            self.FF.make(xk,False,'result')
            logger.info("\nThe final force field has been printed to the 'result' directory.\n")
            #bar = printcool("\x1b[1;45;93mCongratulations, ForceBalance has finished\x1b[0m\n\x1b[1;45;93mGive yourself a pat on the back!\x1b[0m")
            bar = printcool("Congratulations, ForceBalance has finished\nGive yourself a pat on the back!",ansi="1;44;93")

        ## Write out stuff to checkpoint file
        self.writechk()

        return xk
            
    def MainOptimizer(self,b_BFGS=0):
        """ The main ForceBalance adaptive trust-radius pseudo-Newton optimizer.  Tried and true in many situations. :)

        Usually this function is called with the BFGS or NewtonRaphson
        method.  The NewtonRaphson method is consistently the best
        method I have, because I always provide at least an
        approximate Hessian to the objective function.  The BFGS
        method is vestigial and currently does not work.

        BFGS is a pseudo-Newton method in the sense that it builds an
        approximate Hessian matrix from the gradient information in previous
        steps; Newton-Raphson requires the actual Hessian matrix.
        However, the algorithms are similar in that they both compute the
        step by inverting the Hessian and multiplying by the gradient.

        The method adaptively changes the step size.  If the step is
        sufficiently good (i.e. the objective function goes down by a
        large fraction of the predicted decrease), then the step size
        is increased; if the step is bad, then it rejects the step and
        tries again.

        The optimization is terminated after either a function value or
        step size tolerance is reached.

        @param[in] b_BFGS Switch to use BFGS (True) or Newton-Raphson (False)

        """
        if any(['liquid' in tgt.name.lower() for tgt in self.Objective.Targets]) and self.conv_obj < 1e-3:
            warn_press_key("Condensed phase targets detected - may not converge with current choice of convergence_objective (%.e)\nRecommended range is 1e-2 - 1e-1 for this option." % self.conv_obj)
        # Parameters for the adaptive trust radius
        a = self.adapt_fac  # Default value is 0.5, decrease to make more conservative.  Zero to turn off all adaptive.
        b = self.adapt_damp # Default value is 0.5, increase to make more conservative
        printcool( "Main Optimizer\n%s Mode%s" % ("BFGS" if b_BFGS else "Newton-Raphson", " (Static Radius)" if a == 0.0 else " (Adaptive Radius)"), ansi=1, bold=1)
        # First, set a bunch of starting values
        Ord         = 1 if b_BFGS else 2
        #Ord         = 2
        global ITERATION_NUMBER
        ITERATION_NUMBER = 0
        global GOODSTEP
        Best_Step = 1
        if all(i in self.chk for i in ['xk','X','G','H','ehist','x_best','xk_prev','trust']):
            logger.info("Reading initial objective, gradient, Hessian from checkpoint file\n")
            xk, X, G, H, ehist     = self.chk['xk'], self.chk['X'], self.chk['G'], self.chk['H'], self.chk['ehist']
            X_best, xk_prev, trust = self.chk['x_best'], self.chk['xk_prev'], self.chk['trust']
        else:
            xk       = self.mvals0.copy()
            logger.info('\n')
            data     = self.Objective.Full(xk,Ord,verbose=True)
            X, G, H  = data['X'], data['G'], data['H']
            ehist    = np.array([X])
            xk_prev  = xk.copy()
            trust    = abs(self.trust0)
            X_best   = X

        X_prev   = X
        G_prev   = G.copy()
        H_stor   = H.copy()
        ndx    = 0.0
        color  = "\x1b[1m"
        nxk = norm(xk)
        ngr = norm(G)

        Quality  = 0.0
        restep = False
        GOODSTEP = 1
        Ord         = 1 if b_BFGS else 2

        while 1: # Loop until convergence is reached.
            ## Put data into the checkpoint file
            self.chk = {'xk': xk, 'X' : X, 'G' : G, 'H': H, 'ehist': ehist,
                        'x_best': X_best,'xk_prev': xk_prev, 'trust': trust}
            if self.wchk_step:
                self.writechk()
            stdfront = len(ehist) > self.hist and np.std(np.sort(ehist)[:self.hist]) or (len(ehist) > 0 and np.std(ehist) or 0.0)
            stdfront *= 2
            logger.info("%6s%12s%12s%12s%14s%12s%12s\n" % ("Step", "  |k|  ","  |dk|  "," |grad| ","    -=X2=-  ","Delta(X2)", "StepQual"))
            logger.info("%6i%12.3e%12.3e%12.3e%s%14.5e\x1b[0m%12.3e% 11.3f\n\n" % (ITERATION_NUMBER, nxk, ndx, ngr, color, X, stdfront, Quality))
            # Check the convergence criteria
            if ngr < self.conv_grd:
                logger.info("Convergence criterion reached for gradient norm (%.2e)\n" % self.conv_grd)
                break
            if ITERATION_NUMBER == self.maxstep:
                logger.info("Maximum number of optimization steps reached (%i)\n" % ITERATION_NUMBER)
                break
            if ndx < self.conv_stp and ITERATION_NUMBER > 0 and not restep:
                logger.info("Convergence criterion reached in step size (%.2e)\n" % self.conv_stp)
                break
            if stdfront < self.conv_obj and len(ehist) > self.hist and not restep: # Factor of two is so [0,1] stdev is normalized to 1
                logger.info("Convergence criterion reached for objective function (%.2e)\n" % self.conv_obj)
                break
            if self.print_grad:
                bar = printcool("Total Gradient",color=4)
                self.FF.print_map(vals=G,precision=8)
                logger.info(bar)
            if self.print_hess:
                bar = printcool("Total Hessian",color=4)
                pmat2d(H,precision=8)
                logger.info(bar)
            for key, val in self.Objective.ObjDict.items():
                if Best_Step:
                    self.Objective.ObjDict_Last[key] = val
            restep = False
            dx, dX_expect, bump = self.step(xk, data, trust)
            old_pk = self.FF.create_pvals(xk)
            old_xk = xk.copy()
            # Increment the iteration counter.
            ITERATION_NUMBER += 1
            # Take a step in the parameter space.
            xk += dx
            if self.print_vals:
                pk = self.FF.create_pvals(xk)
                dp = pk - old_pk
                bar = printcool("Mathematical Parameters (Current + Step = Next)",color=5)
                self.FF.print_map(vals=["% .4e %s %.4e = % .4e" % (old_xk[i], '+' if dx[i] >= 0 else '-', abs(dx[i]), xk[i]) for i in range(len(xk))])
                logger.info(bar)
                bar = printcool("Physical Parameters (Current + Step = Next)",color=5)
                self.FF.print_map(vals=["% .4e %s %.4e = % .4e" % (old_pk[i], '+' if dp[i] >= 0 else '-', abs(dp[i]), pk[i]) for i in range(len(pk))])
                logger.info(bar)
            # Evaluate the objective function and its derivatives.
            data        = self.Objective.Full(xk,Ord,verbose=True)
            X, G, H = data['X'], data['G'], data['H']
            ndx = norm(dx)
            nxk = norm(xk)
            ngr = norm(G)
            drc = abs(flat(dx)).argmax()

            dX_actual = X - X_prev
            try:
                Quality = dX_actual / dX_expect
            except:
                logger.warning("Warning: Step size of zero detected (i.e. wrong direction).  Try reducing the finite_difference_h parameter\n")
                Quality = 1.0 # This is a step length of zero.

            if Quality <= 0.25 and X < (X_prev + self.err_tol) and self.trust0 > 0:
                # If the step quality is bad, then we should decrease the trust radius.
                trust = max(ndx*(1./(1+a)), self.mintrust)
                logger.info("Low quality step, reducing trust radius to % .4e\n" % trust)
            if Quality >= 0.75 and bump and self.trust0 > 0:
                # If the step quality is good, then we should increase the trust radius.
                # The 'a' factor is how much we should grow or shrink the trust radius each step
                # and the 'b' factor determines how closely we are tied down to the original value.
                # Recommend values 0.5 and 0.5
                trust += a*trust*np.exp(-b*(trust/self.trust0 - 1))
            if X > (X_prev + self.err_tol):
                Best_Step = 0
                # Toggle switch for rejection (experimenting with no rejection)
                Rejects = True
                GOODSTEP = 0
                Reevaluate = True
                trust = max(ndx*(1./(1+a)), self.mintrust)
                logger.info("Rejecting step and reducing trust radius to % .4e\n" % trust)
                if Rejects:
                    xk = xk_prev.copy()
                    if Reevaluate:
                        restep = True
                        color = "\x1b[91m"
                        logger.info("%6s%12s%12s%12s%14s%12s%12s\n" % ("Step", "  |k|  ","  |dk|  "," |grad| ","    -=X2=-  ","Delta(X2)", "StepQual"))
                        logger.info("%6i%12.3e%12.3e%12.3e%s%14.5e\x1b[0m%12.3e% 11.3f\n\n" % (ITERATION_NUMBER, nxk, ndx, ngr, color, X, stdfront, Quality))
                        printcool("Objective function rises!\nRe-evaluating at the previous point..",color=1)
                        ITERATION_NUMBER += 1
                        data        = self.Objective.Full(xk,Ord,verbose=True)
                        GOODSTEP = 1
                        X, G, H = data['X'], data['G'], data['H']
                        X_prev = X
                        dx *= 0
                        ndx = norm(dx)
                        nxk = norm(xk)
                        ngr = norm(G)
                        Quality = 0.0
                        color = "\x1b[0m"
                    else:
                        color = "\x1b[91m"
                        G = G_prev.copy()
                        H = H_stor.copy()
                        data = deepcopy(datastor)
                    continue
            else:
                GOODSTEP = 1
                if X > X_best:
                    Best_Step = 0
                    color = "\x1b[95m"
                else:
                    Best_Step = 1
                    color = "\x1b[92m"
                    X_best = X
                ehist = np.append(ehist, X)
            # Hessian update for BFGS.
            if b_BFGS:
                Hnew = H_stor.copy()
                Dx   = col(xk - xk_prev)
                Dy   = col(G  - G_prev)
                Mat1 = (Dy*Dy.T)/(Dy.T*Dx)[0,0]
                Mat2 = ((Hnew*Dx)*(Hnew*Dx).T)/(Dx.T*Hnew*Dx)[0,0]
                Hnew += Mat1-Mat2
                H = Hnew.copy()
                data['H'] = H.copy()

            datastor= deepcopy(data)
            G_prev  = G.copy()
            H_stor  = H.copy()
            xk_prev = xk.copy()
            X_prev  = X
            if len(self.FF.parmdestroy_this) > 0:
                self.FF.parmdestroy_save.append(self.FF.parmdestroy_this)
                self.FF.linedestroy_save.append(self.FF.linedestroy_this)
        
        bar = printcool("Final objective function value\nFull: % .6e  Un-penalized: % .6e" % (data['X'],data['X0']), '@', bold=True, color=2)
        return xk


    def step(self, xk, data, trust):
        """ Computes the next step in the parameter space.  There are lots of tricks here that I will document later.

        @param[in] G The gradient
        @param[in] H The Hessian
        @param[in] trust The trust radius
        
        """
        from scipy import optimize

        X, G, H = (data['X0'], data['G0'], data['H0']) if self.bhyp else (data['X'], data['G'], data['H'])
        H1 = H.copy()
        H1 = np.delete(H1, self.excision, axis=0)
        H1 = np.delete(H1, self.excision, axis=1)
        Eig = eig(H1)[0]            # Diagonalize Hessian
        Emin = min(Eig)
        if Emin < self.eps:         # Mix in SD step if Hessian minimum eigenvalue is negative
            # Experiment.
            Adj = max(self.eps, 0.01*abs(Emin)) - Emin
            logger.info("Hessian has a small or negative eigenvalue (%.1e), mixing in some steepest descent (%.1e) to correct this.\n" % (Emin, Adj))
            logger.info("Eigenvalues are:\n")   ###
            pvec1d(Eig)                ###
            H += Adj*np.eye(H.shape[0])

        if self.bhyp:
            G = np.delete(G, self.excision)
            H = np.delete(H, self.excision, axis=0)
            H = np.delete(H, self.excision, axis=1)
            xkd = np.delete(xk, self.excision)
            if self.Objective.Penalty.fmul != 0.0:
                warn_press_key("Using the multiplicative hyperbolic penalty is discouraged!")
            # This is the gradient and Hessian without the contributions from the hyperbolic constraint.
            Obj0 = {'X':X,'G':G,'H':H}
            class Hyper(object):
                def __init__(self, HL, Penalty):
                    self.H = HL.copy()
                    self.dx = 1e10 * np.ones(len(HL),dtype=float)
                    self.Val = 0
                    self.Grad = np.zeros(len(HL),dtype=float)
                    self.Hess = np.zeros((len(HL),len(HL)),dtype=float)
                    self.Penalty = Penalty
                def _compute(self, dx):
                    self.dx = dx.copy()
                    Tmp = np.mat(self.H)*col(dx)
                    Reg_Term   = self.Penalty.compute(xkd+flat(dx), Obj0)
                    self.Val   = (X + np.dot(dx, G) + 0.5*row(dx)*Tmp + Reg_Term[0] - data['X'])[0,0]
                    self.Grad  = flat(col(G) + Tmp) + Reg_Term[1]
                def compute_val(self, dx):
                    if norm(dx - self.dx) > 1e-8:
                        self._compute(dx)
                    return self.Val
                def compute_grad(self, dx):
                    if norm(dx - self.dx) > 1e-8:
                        self._compute(dx)
                    return self.Grad
                def compute_hess(self, dx):
                    if norm(dx - self.dx) > 1e-8:
                        self._compute(dx)
                    return self.Hess
            def hyper_solver(L):
                dx0 = np.zeros(len(xkd),dtype=float)
                #dx0 = np.delete(dx0, self.excision)
                # HL = H + (L-1)**2*np.diag(np.diag(H))
                # Attempt to use plain Levenberg
                HL = H + (L-1)**2*np.eye(len(H))

                HYP = Hyper(HL, self.Objective.Penalty)
                try:
                    Opt1 = optimize.fmin_bfgs(HYP.compute_val,dx0,fprime=HYP.compute_grad,gtol=1e-5,full_output=True,disp=0)
                except:
                    Opt1 = optimize.fmin(HYP.compute_val,dx0,full_output=True,disp=0)
                try:
                    Opt2 = optimize.fmin_bfgs(HYP.compute_val,-xkd,fprime=HYP.compute_grad,gtol=1e-5,full_output=True,disp=0)
                except:
                    Opt2 = optimize.fmin(HYP.compute_val,-xkd,full_output=True,disp=0)
                #Opt2 = optimize.fmin(HYP.compute_val,-xkd,full_output=True,disp=0)
                dx1, sol1 = Opt1[0], Opt1[1]
                dx2, sol2 = Opt2[0], Opt2[1]
                dxb, sol = (dx1, sol1) if sol1 <= sol2 else (dx2, sol2)
                for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
                    dxb = np.insert(dxb, i, 0)
                return dxb, sol
        else:
            # G0 and H0 are used for determining the expected function change.
            G0 = G.copy()
            H0 = H.copy()
            G = np.delete(G, self.excision)
            H = np.delete(H, self.excision, axis=0)
            H = np.delete(H, self.excision, axis=1)
            
            logger.debug("Inverting Hessian:\n")                 ###
            logger.debug(" G:\n")                                ###
            pvec1d(G,precision=5, loglevel=DEBUG)                ###
            logger.debug(" H:\n")                                ###
            pmat2d(H,precision=5, loglevel=DEBUG)                ###
            
            Hi = invert_svd(np.mat(H))
            dx = flat(-1 * Hi * col(G))
            
            logger.debug(" dx:\n")                               ###
            pvec1d(dx,precision=5, loglevel=DEBUG)                     ###
            # dxa = -solve(H, G)          # Take Newton Raphson Step ; use -1*G if want steepest descent.
            # dxa = flat(dxa)
            # print " dxa:"                              ###
            # pvec1d(dxa,precision=5)                    ###
            
            logger.info('\n')                                      ###
            for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
                dx = np.insert(dx, i, 0)
            def para_solver(L):
                # Levenberg-Marquardt
                # HT = H + (L-1)**2*np.diag(np.diag(H))
                # Attempt to use plain Levenberg
                HT = H + (L-1)**2*np.eye(len(H))
                logger.debug("Inverting Scaled Hessian:\n")                       ###
                logger.debug(" G:\n")                                             ###
                pvec1d(G,precision=5, loglevel=DEBUG)                                   ###
                logger.debug(" HT: (Scal = %.4f)\n" % (1+(L-1)**2))               ###
                pmat2d(HT,precision=5, loglevel=DEBUG)                                  ###
                Hi = invert_svd(np.mat(HT))
                dx = flat(-1 * Hi * col(G))
                logger.debug(" dx:\n")                                            ###
                pvec1d(dx,precision=5, loglevel=DEBUG)                                  ###
                # dxa = -solve(HT, G)
                # dxa = flat(dxa)
                # print " dxa:"                                           ###
                # pvec1d(dxa,precision=5)                                 ###
                # print                                                   ###
                sol = flat(0.5*row(dx)*np.mat(H)*col(dx))[0] + np.dot(dx,G)
                for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
                    dx = np.insert(dx, i, 0)
                return dx, sol
    
        def solver(L):
            return hyper_solver(L) if self.bhyp else para_solver(L)
    
        def trust_fun(L):
            N = norm(solver(L)[0])
            logger.debug("\rL = %.4e, Hessian diagonal addition = %.4e: found length %.4e, objective is %.4e\n" % (L, (L-1)**2, N, (N - trust)**2))
            return (N - trust)**2

        def search_fun(L):
            # Evaluate ONLY the objective function.  Most useful when
            # the objective is cheap, but the derivative is expensive.
            dx, sol = solver(L) # dx is how much the step changes from the previous step.
            # This is our trial step.
            xk_ = dx + xk
            Result = self.Objective.Full(xk_,0,verbose=False)['X'] - data['X']
            logger.info("Searching! Hessian diagonal addition = %.4e, L = % .4e, length %.4e, result %.4e\n" % ((L-1)**2,L,norm(dx),Result))
            return Result
        
        if self.trust0 > 0: # This is the trust region code.
            bump = False
            dx, expect = solver(1)
            dxnorm = norm(dx)
            if dxnorm > trust:
                bump = True
                # Tried a few optimizers here, seems like Brent works well.
                # Okay, the problem with Brent is that the tolerance is fractional.  
                # If the optimized value is zero, then it takes a lot of meaningless steps.
                LOpt = optimize.brent(trust_fun,brack=(self.lmg,self.lmg*4),tol=1e-6)
                ### Result = optimize.fmin_powell(trust_fun,3,xtol=self.search_tol,ftol=self.search_tol,full_output=1,disp=0)
                ### LOpt = Result[0]
                dx, expect = solver(LOpt)
                dxnorm = norm(dx)

                logger.info("\rLevenberg-Marquardt: %s step found (length %.3e), % .8f added to Hessian diagonal\n" % ('hyperbolic-regularized' if self.bhyp else 'Newton-Raphson', dxnorm, (LOpt-1)**2))
        else: # This is the nonlinear search code.
            # First obtain a step that is the same length as the provided trust radius.
            LOpt = optimize.brent(trust_fun,brack=(self.lmg,self.lmg*4),tol=1e-6)
            bump = False
            Result = optimize.brent(search_fun,brack=(LOpt,LOpt*4),tol=self.search_tol,full_output=1)
            ### optimize.fmin(search_fun,0,xtol=1e-8,ftol=data['X']*0.1,full_output=1,disp=0)
            ### Result = optimize.fmin_powell(search_fun,3,xtol=self.search_tol,ftol=self.search_tol,full_output=1,disp=0)
            dx, _ = solver(Result[0])
            expect = Result[1]

        ## Decide which parameters to redirect.
        ## Currently not used.
        if self.Objective.Penalty.ptyp in [3,4,5]:
            self.FF.make_redirect(dx+xk)

        return dx, expect, bump

    def NewtonRaphson(self):
        """ Optimize the force field parameters using the Newton-Raphson method (@see MainOptimizer) """
        return self.MainOptimizer(b_BFGS=0)

    def BFGS(self):
        """ Optimize the force field parameters using the BFGS method; currently the recommended choice (@see MainOptimizer) """
        return self.MainOptimizer(b_BFGS=1)

    def ScipyOptimizer(self,Algorithm="None"):
        """ Driver for SciPy optimizations.

        Using any of the SciPy optimizers requires that SciPy is installed.
        This method first defines several wrappers around the objective function that the SciPy
        optimizers can use.  Then it calls the algorith mitself.

        @param[in] Algorithm The optimization algorithm to use, for example 'powell', 'simplex' or 'anneal'

        """
        from scipy import optimize
        def xwrap(func,verbose=True):
            def my_func(mvals):
                if verbose: logger.info('\n')
                Answer = func(mvals,Order=0,verbose=verbose)['X']
                dx = (my_func.x_best - Answer) if my_func.x_best != None else 0.0
                if Answer < my_func.x_best or my_func.x_best == None:
                    color = "\x1b[92m"
                    my_func.x_best = Answer
                else:
                    color = "\x1b[91m"
                if verbose:
                    if self.print_vals:
                        logger.info("k=" + ' '.join(["% .4f" % i for i in mvals]) + '\n')
                    logger.info("X2= %s%12.3e\x1b[0m d(X2)= %12.3e\n" % (color,Answer,dx))
                if Answer != Answer:
                    return 1e10
                else:
                    return Answer
            my_func.x_best = None
            return my_func
        def gwrap(func,verbose=True):
            def my_gfunc(mvals):
                if verbose: logger.info('\n')
                Output = func(mvals,Order=1,verbose=verbose)
                Answer = Output['G']
                Objective = Output['X']
                dx = (my_gfunc.x_best - Objective) if my_gfunc.x_best != None else 0.0
                if Objective < my_gfunc.x_best or my_gfunc.x_best == None:
                    color = "\x1b[92m"
                    my_gfunc.x_best = Objective
                else:
                    color = "\x1b[91m"
                if verbose:
                    if self.print_vals:
                        logger.info("k=" + ' '.join(["% .4f" % i for i in mvals]) + '\n')
                    logger.info("|Grad|= %12.3e X2= %s%12.3e\x1b[0m d(X2)= %12.3e\n\n" % (norm(Answer),color,Objective,dx))
                return Answer
            my_gfunc.x_best = None
            return my_gfunc
        if Algorithm == "powell":
            printcool("Minimizing Objective Function using Powell's Method" , ansi=1, bold=1)
            return optimize.fmin_powell(xwrap(self.Objective.Full),self.mvals0,ftol=self.conv_obj,xtol=self.conv_stp,maxiter=self.maxstep)
        elif Algorithm == "simplex":
            printcool("Minimizing Objective Function using Simplex Method" , ansi=1, bold=1)
            return optimize.fmin(xwrap(self.Objective.Full),self.mvals0,ftol=self.conv_obj,xtol=self.conv_stp,maxiter=self.maxstep,maxfun=self.maxstep*10)
        elif Algorithm == "anneal":
            printcool("Minimizing Objective Function using Simulated Annealing" , ansi=1, bold=1)
            return optimize.anneal(xwrap(self.Objective.Full),self.mvals0,lower=-1*self.trust0*np.ones(self.np),upper=self.trust0*np.ones(self.np),schedule='boltzmann')
        elif Algorithm == "cg":
            printcool("Minimizing Objective Function using Conjugate Gradient" , ansi=1, bold=1)
            return optimize.fmin_cg(xwrap(self.Objective.Full,verbose=False),self.mvals0,fprime=gwrap(self.Objective.Full),gtol=self.conv_grd)

    def GeneticAlgorithm(self):
        
        """ 
        Genetic algorithm, under development. It currently works but a
        genetic algorithm is more like a concept; i.e. there is no
        single way to implement it.
        
        @todo Massive parallelization hasn't been implemented yet

        """
        def generate_fresh(rows, cols):
            new_guys = np.zeros((rows, cols))
            for i in range(rows):
                new_guys[i, int(cols*np.random.random())] = self.trust0 * np.random.randn()
            return new_guys
        
        def cross_over(chrom1, chrom2):
            crosspt = 1 + int((len(chrom1) - 1) * np.random.random())
            Ans1 = np.hstack((chrom1[:crosspt],chrom2[crosspt:]))
            Ans2 = np.hstack((chrom2[:crosspt],chrom1[crosspt:]))
            return Ans1, Ans2

        def mutate(chrom):
            mutpt = int(len(chrom) * np.random.random())
            chrom[mutpt] += self.trust0 * np.random.randn()
            return chrom

        def initial_generation():
            return np.vstack((self.mvals0.copy(),np.random.randn(PopSize, self.FF.np)*self.trust0)) / (self.FF.np ** 0.5)
            #return np.vstack((self.mvals0.copy(),generate_fresh(PopSize, self.np)))

        def calculate_fitness(pop):
            return [self.Objective.Full(i,Order=0,verbose=False)['X'] for i in pop]

        def sort_by_fitness(fits):
            return np.sort(fits), np.argsort(fits)

        def generate_new_population(sorted, pop):
            newpop = pop[sorted[1]]
            # Individuals in this range are kept
            a = range(KeepNum)
            logger.info("Keeping: " + str(a) + '\n')
            random.shuffle(a)
            for i in range(0, KeepNum, 2):
                logger.info("%i and %i reproducing to replace %i and %i\n" % (a[i],a[i+1],len(newpop)-i-2,len(newpop)-i-1))
                newpop[-i-1], newpop[-i-2] = cross_over(newpop[a[i]],newpop[a[i+1]])
            b = range(KeepNum, len(newpop))
            random.shuffle(b)
            for i in b[:MutNum]:
                logger.info("Randomly mutating %i\n" % i)
                newpop[i] = mutate(newpop[i])
            return newpop
            
        def xwrap(func,verbose=True):
            def my_func(mvals):
                if verbose: logger.info('\n')
                Answer = func(mvals,Order=0,verbose=verbose)['X']
                dx = (my_func.x_best - Answer) if my_func.x_best != None else 0.0
                if Answer < my_func.x_best or my_func.x_best == None:
                    color = "\x1b[92m"
                    my_func.x_best = Answer
                else:
                    color = "\x1b[91m"
                if verbose:
                    logger.info("k=" + ' '.join(["% .4f" % i for i in mvals]) + '\n')
                    logger.info("X2= %s%12.3e\x1b[0m d(X2)= %12.3e\n" % (color,Answer,dx))
                return Answer
            my_func.x_best = None
            return my_func

        PopSize = 120
        KeepThresh = 0.5
        MutProb = 0.1
        CrosProb = 0.5
        
        KeepNum = int(KeepThresh * PopSize)
        MutNum = int(MutProb * PopSize)
        CrosNum = int(CrosProb/2 * PopSize) * 2
        Population = initial_generation()
        Gen = 0

        Best = [[],[]]

        while True:
            #print Population
            Fits = calculate_fitness(Population)
            Sorted = sort_by_fitness(Fits)
            logger.info(str(Sorted))
            Best[0].append(Sorted[0][0])
            Best[1].append(Population[Sorted[1][0]])
            logger.info(str(Best))
            if Gen == self.maxstep: break
            Population = generate_new_population(Sorted, Population)
            Gen += 1

        logger.info(str(Best))
        return Population[Sorted[1][0]]
        

    def Simplex(self):
        """ Use SciPy's built-in simplex algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        return self.ScipyOptimizer(Algorithm="simplex")

    def Powell(self):
        """ Use SciPy's built-in Powell direction-set algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        return self.ScipyOptimizer(Algorithm="powell")

    def Anneal(self):
        """ Use SciPy's built-in simulated annealing algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        return self.ScipyOptimizer(Algorithm="anneal")

    def ConjugateGradient(self):
        """ Use SciPy's built-in simulated annealing algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        return self.ScipyOptimizer(Algorithm="cg")

    def Scan_Values(self,MathPhys=1):
        """ Scan through parameter values.

        This option is activated using the inputs:

        @code
        scan[mp]vals
        scan_vals low:hi:nsteps
        scan_idxnum (number) -or-
        scan_idxname (name)
        @endcode
        
        This method goes to the specified parameter indices and scans through
        the supplied values, evaluating the objective function at every step.

        I hope this method will be useful for people who just want to look at
        changing one or two parameters and seeing how it affects the force
        field performance.

        @todo Maybe a multidimensional grid can be done.
        @param[in] MathPhys Switch to use mathematical (True) or physical (False) parameters.
        
        """
        # First make sure that the user entered the correct syntax.
        try:
            vals_in = [float(i) for i in self.scan_vals.split(":")]
        except:
            logger.error("Syntax error: in the input file please use scan_vals low:hi:nsteps\n")
            sys.exit(1)
        if len(vals_in) != 3:
            logger.error("Syntax error: in the input file please use scan_vals low:hi:nsteps\n")
            sys.exit(1)
        idx = [int(i) for i in self.idxnum]
        for j in self.idxname:
            idx += [self.FF.map[i] for i in self.FF.map if j in i]
        idx = set(idx)
        scanvals = np.linspace(vals_in[0],vals_in[1],vals_in[2])
        logger.info(str(vals_in) + '\n')
        logger.info(str(scanvals) + '\n')
        for pidx in idx:
            if MathPhys:
                logger.info("Scanning parameter %i (%s) in the mathematical space\n" % (pidx,self.FF.plist[pidx]))
                vals = self.mvals0.copy()
            else:
                logger.info("Scanning parameter %i (%s) in the physical space\n" % (pidx,self.FF.plist[pidx]))
                self.FF.use_pvals = True
                vals = self.FF.pvals0.copy()
            for i in scanvals:
                vals[pidx] = i
                data        = self.Objective.Full(vals,Order=0)
                logger.info("Value = % .4e Objective = % .4e\n" % (i, data['X']))

    def ScanMVals(self):
        """ Scan through the mathematical parameter space. @see Optimizer::ScanValues """
        self.Scan_Values(1)

    def ScanPVals(self):
        """ Scan through the physical parameter space. @see Optimizer::ScanValues """
        self.Scan_Values(0)

    def SinglePoint(self):
        """ A single-point objective function computation. """
        data        = self.Objective.Full(self.mvals0,Order=0,verbose=True)
        logger.info("The objective function is:" + str(data['X']) + '\n')

    def Gradient(self):
        """ A single-point gradient computation. """
        data        = self.Objective.Full(self.mvals0,Order=1)
        logger.info(str(data['X']) + '\n')
        logger.info(str(data['G']) + '\n')
        logger.info(str(data['H']) + '\n')

    def Hessian(self):
        """ A single-point Hessian computation. """
        data        = self.Objective.Full(self.mvals0,Order=2)
        logger.info(str(data['X']) + '\n')
        logger.info(str(data['G']) + '\n')
        logger.info(str(data['H']) + '\n')

    def FDCheckG(self):
        """ Finite-difference checker for the objective function gradient.

        For each element in the gradient, use a five-point finite difference
        stencil to compute a finite-difference derivative, and compare it to
        the analytic result.

        """

        Adata        = self.Objective.Full(self.mvals0,Order=1)['G']
        Fdata        = np.zeros(self.FF.np,dtype=float)
        printcool("Checking first derivatives by finite difference!\n%-8s%-20s%13s%13s%13s%13s" \
                  % ("Index", "Parameter ID","Analytic","Numerical","Difference","Fractional"),bold=1,color=5)
        for i in range(self.FF.np):
            Fdata[i] = f1d7p(fdwrap(self.Objective.Full,self.mvals0,i,'X',Order=0),self.h)
            Denom = max(abs(Adata[i]),abs(Fdata[i]))
            Denom = Denom > 1e-8 and Denom or 1e-8
            D = Adata[i] - Fdata[i]
            Q = (Adata[i] - Fdata[i])/Denom
            cD = abs(D) > 0.5 and "\x1b[1;91m" or (abs(D) > 1e-2 and "\x1b[91m" or (abs(D) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            cQ = abs(Q) > 0.5 and "\x1b[1;91m" or (abs(Q) > 1e-2 and "\x1b[91m" or (abs(Q) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            logger.info("\r    %-8i%-20s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m\n" \
                  % (i, self.FF.plist[i][:20], Adata[i], Fdata[i], cD, D, cQ, Q))

    def FDCheckH(self):
        """ Finite-difference checker for the objective function Hessian.

        For each element in the Hessian, use a five-point stencil in both
        parameter indices to compute a finite-difference derivative, and
        compare it to the analytic result.

        This is meant to be a foolproof checker, so it is pretty slow.  We
        could write a faster checker if we assumed we had accurate first
        derivatives, but it's better to not make that assumption.

        The second derivative is computed by double-wrapping the objective
        function via the 'wrap2' function.

        """
        Adata        = self.Objective.Full(self.mvals0,Order=2)['H']
        Fdata        = np.zeros((self.FF.np,self.FF.np),dtype=float)
        printcool("Checking second derivatives by finite difference!\n%-8s%-20s%-20s%13s%13s%13s%13s" \
                  % ("Index", "Parameter1 ID", "Parameter2 ID", "Analytic","Numerical","Difference","Fractional"),bold=1,color=5)

        # Whee, our double-wrapped finite difference second derivative!
        def wrap2(mvals0,pidxi,pidxj):
            def func1(arg):
                mvals = list(mvals0)
                mvals[pidxj] += arg
                return f1d5p(fdwrap(self.Objective.Full,mvals,pidxi,'X',Order=0),self.h)
            return func1
        
        for i in range(self.FF.np):
            for j in range(i,self.FF.np):
                Fdata[i,j] = f1d5p(wrap2(self.mvals0,i,j),self.h)
                Denom = max(abs(Adata[i,j]),abs(Fdata[i,j]))
                Denom = Denom > 1e-8 and Denom or 1e-8
                D = Adata[i,j] - Fdata[i,j]
                Q = (Adata[i,j] - Fdata[i,j])/Denom
                cD = abs(D) > 0.5 and "\x1b[1;91m" or (abs(D) > 1e-2 and "\x1b[91m" or (abs(D) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
                cQ = abs(Q) > 0.5 and "\x1b[1;91m" or (abs(Q) > 1e-2 and "\x1b[91m" or (abs(Q) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
                logger.info("\r    %-8i%-20s%-20s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m\n" \
                      % (i, self.FF.plist[i][:20], self.FF.plist[j][:20], Adata[i,j], Fdata[i,j], cD, D, cQ, Q))

    def readchk(self):
        """ Read the checkpoint file for the main optimizer. """
        self.chk = {}
        if self.rchk_fnm != None:
            absfnm = os.path.join(self.root,self.rchk_fnm)
            if os.path.exists(absfnm):
                self.chk = pickle.load(open(absfnm))
            else:
                logger.info("\x1b[40m\x1b[1;92mWARNING:\x1b[0m read_chk is set to True, but checkpoint file not loaded (wrong filename or doesn't exist?)\n")
        return self.chk

    def writechk(self):
        """ Write the checkpoint file for the main optimizer. """
        if self.wchk_fnm != None:
            logger.info("Writing the checkpoint file %s\n" % self.wchk_fnm)
            with open(os.path.join(self.root,self.wchk_fnm),'w') as f: pickle.dump(self.chk,f)
        
