""" @package optimizer Optimization algorithms.

My current implementation is to have a single optimizer class with several methods
contained inside.

@author Lee-Ping Wang
@date 12/2011

"""

import os, pickle, re, sys
import numpy as np
from numpy.linalg import eig, norm, solve
from nifty import col, flat, row, printcool, printcool_dictionary, pmat2d, warn_press_key
from finite_difference import f1d7p, f1d5p, fdwrap
import random

# Global variable corresponding to the iteration number.  This allows the
# Main Optimizer to 
ITERATION_NUMBER = None

def Counter():
    global ITERATION_NUMBER
    return ITERATION_NUMBER

class Optimizer(object):
    """ Optimizer class.  Contains several methods for numerical optimization.

    For various reasons, the optimizer depends on the force field and fitting
    simulations (i.e. we cannot treat it as a fully independent numerical optimizer).
    The dependency is rather weak which suggests that I can remove it someday.
    """
    
    def __init__(self,options,Objective,FF):
        """ Instantiation of the optimizer.

        The optimizer depends on both the FF and the fitting simulations so there
        is a chain of dependencies: FF --> FitSim --> Optimizer, and FF --> Optimizer

        Here's what we do:
        - Take options from the parser
        - Pass in the objective function, force field, all fitting simulations

        """
        
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
        self.root      = options['root']
        ## The job type
        self.jobtype   = options['jobtype']
        ## Initial step size trust radius
        self.trust0    = options['trust0']
        ## Minimum trust radius (for noisy objective functions)
        self.mintrust  = options['mintrust']
        ## Lower bound on Hessian eigenvalue (below this, we add in steepest descent)
        self.eps       = options['eig_lowerbound']
        ## Step size for numerical finite difference
        self.h         = options['finite_difference_h']
        ## Function value convergence threshold
        self.conv_obj  = options['convergence_objective']
        ## Step size convergence threshold
        self.conv_stp  = options['convergence_step']
        ## Gradient convergence threshold
        self.conv_grd  = options['convergence_gradient']
        ## Maximum number of optimization steps
        self.maxstep   = options['maxstep']
        ## For scan[mp]vals: The parameter index to scan over
        self.idxnum    = options['scanindex_num']
        ## For scan[mp]vals: The parameter name to scan over, it just looks up an index
        self.idxname   = options['scanindex_name']
        ## For scan[mp]vals: The values that are fed into the scanner
        self.scan_vals = options['scan_vals']
        ## Name of the checkpoint file that we're reading in
        self.rchk_fnm  = options['readchk']
        ## Name of the checkpoint file that we're writing out
        self.wchk_fnm  = options['writechk']
        ## Whether to write the checkpoint file at every step
        self.wchk_step = options['writechk_step']
        ## Adaptive trust radius adjustment factor
        self.adapt_fac  = options['adaptive_factor']
        ## Adaptive trust radius adjustment damping
        self.adapt_damp = options['adaptive_damping']
        ## Whether to print gradient during each step of the optimization
        self.print_grad = options['print_gradient']
        ## Whether to print Hessian during each step of the optimization
        self.print_hess = options['print_hessian']
        ## Whether to print parameters during each step of the optimization
        self.print_vals = options['print_parameters']
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The objective function (needs to pass in when I instantiate)
        self.Objective = Objective
        ## Whether the penalty function is hyperbolic
        self.bhyp      = Objective.Penalty.ptyp == 1
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
            self.mvals0    = np.zeros(self.np)

        ## Print the optimizer options.
        printcool_dictionary(options, title="Setup for optimizer")
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
            bar = printcool("Final parameter values\n Paste to input file to restart\n Choose pvals or mvals",bold=True,color=4)
            print "read_pvals"
            self.FF.print_map(self.FF.create_pvals(xk))
            print "/read_pvals"
            print "read_mvals"
            self.FF.print_map(xk)
            print "/read_mvals"
            print bar
            self.FF.make(xk,False,'result')

        ## Write out stuff to checkpoint file
        self.writechk()
            
    def MainOptimizer(self,b_BFGS=0):
        """ The main ForceBalance adaptive trust-radius pseudo-Newton optimizer.  Tried and true in many situations. :)

        Usually this function is called with the BFGS or NewtonRaphson
        method.  I've found the BFGS method to be most efficient,
        especially when we don't have access to the expensive analytic
        second derivatives of the objective function.  If we are
        computing derivatives by of the objective function by finite
        difference (which we often do), then the diagonal elements of
        the second derivative can also be obtained by taking a central
        difference.

        BFGS is a pseudo-Newton method in the sense that it builds an
        approximate Hessian matrix from the gradient information in previous
        steps; true Newton-Raphson needs all of the second derivatives.
        However, the algorithms are similar in that they both compute the
        step by inverting the Hessian and multiplying by the gradient.

        As this method iterates toward convergence, it computes BFGS updates
        of the Hessian matrix and adjusts the step size.  If the step
        is good (i.e. the objective function goes down), then the step
        size is increased; if the step is bad, then it steps back to the
        original point and tries again with a smaller step size.

        The optimization is terminated after either a function value or
        step size tolerance is reached.

        @param[in] b_BFGS Switch to use BFGS (True) or Newton-Raphson (False)

        """
        # Parameters for the adaptive trust radius
        a = self.adapt_fac  # Default value is 0.5, decrease to make more conservative.  Zero to turn off all adaptive.
        b = self.adapt_damp # Default value is 0.5, increase to make more conservative
        printcool( "Main Optimizer\n%s Mode%s" % ("BFGS" if b_BFGS else "Newton-Raphson", " (Static Radius)" if a == 0.0 else " (Adaptive Radius)"), color=7, bold=1)
        # First, set a bunch of starting values
        Ord         = 1 if b_BFGS else 2
        global ITERATION_NUMBER
        ITERATION_NUMBER = 0
        if all(i in self.chk for i in ['xk','X','G','H','ehist','x_best','xk_prev','trust']):
            print "Reading initial objective, gradient, Hessian from checkpoint file"
            xk, X, G, H, ehist     = self.chk['xk'], self.chk['X'], self.chk['G'], self.chk['H'], self.chk['ehist']
            X_best, xk_prev, trust = self.chk['x_best'], self.chk['xk_prev'], self.chk['trust']
        else:
            xk       = self.mvals0.copy()
            print
            data     = self.Objective.Full(xk,Ord,verbose=True) # Try to get a Hessian no matter what on the first step.
            X, G, H  = data['X'], data['G'], data['H']
            ehist    = np.array([X])
            xk_prev  = xk.copy()
            trust    = self.trust0
            X_best   = X

        X_prev   = X
        G_prev   = G.copy()
        H_stor   = H.copy()
        stepn  = 0
        ndx    = 0.0
        color  = "\x1b[97m"
        nxk = norm(xk)
        ngr = norm(G)

        wolfe_c1 = 1e-4
        wolfe_c2 = 0.9
        Quality  = 1.0

        while 1: # Loop until convergence is reached.
            ITERATION_NUMBER += 1
            ## Put data into the checkpoint file
            self.chk = {'xk': xk, 'X' : X, 'G' : G, 'H': H, 'ehist': ehist,
                        'x_best': X_best,'xk_prev': xk_prev, 'trust': trust}
            if self.wchk_step:
                self.writechk()
            stdfront = len(ehist) > 10 and np.std(np.sort(ehist)[:10]) or (len(ehist) > 0 and np.std(ehist) or 0.0)
            print "%6s%12s%12s%12s%14s%12s%12s" % ("Step", "  |k|  ","  |dk|  "," |grad| ","    -=X2=-  ","Stdev(X2)", "StepQual")
            print "%6i%12.3e%12.3e%12.3e%s%14.5e\x1b[0m%12.3e% 11.3f\n" % (stepn, nxk, ndx, ngr, color, X, stdfront, Quality)
            # Check the convergence criteria
            if ngr < self.conv_grd:
                print "Convergence criterion reached for gradient norm (%.2e)" % self.conv_grd
                break
            if stepn == self.maxstep:
                print "Maximum number of optimization steps reached (%i)" % stepn
                break
            if ndx < self.conv_stp and stepn > 0:
                print "Convergence criterion reached in step size (%.2e)" % self.conv_stp
                break
            if stdfront < self.conv_obj and len(ehist) > 10:
                print "Convergence criterion reached for objective function (%.2e)" % self.conv_obj
                break
            if self.print_grad:
                bar = printcool("Total Gradient",color=6)
                self.FF.print_map(vals=G)
                print bar
            if self.print_hess:
                bar = printcool("Total Hessian",color=6)
                pmat2d(H)
                print bar
            dx, dX_expect, bump = self.step(xk, data, trust)
            # if self.bhyp:
            #     dx, dX_expect, bump = self.step_hyperbolic(xk, data, trust)
            # else:
            #     dx, dX_expect, bump = self.step_normal(G, H, trust)
            old_pk = self.FF.create_pvals(xk)
            old_xk = xk.copy()
            # Take a step in the parameter space.
            xk += dx
            if self.print_vals:
                pk = self.FF.create_pvals(xk)
                dp = pk - old_pk
                bar = printcool("Mathematical Parameters (Current + Step = Next)",color=3)
                self.FF.print_map(vals=["% .4e %s %.4e = % .4e" % (old_xk[i], '+' if dx[i] >= 0 else '-', abs(dx[i]), xk[i]) for i in range(len(xk))])
                print bar
                bar = printcool("Physical Parameters (Current + Step = Next)",color=3)
                self.FF.print_map(vals=["% .4e %s %.4e = % .4e" % (old_pk[i], '+' if dp[i] >= 0 else '-', abs(dp[i]), pk[i]) for i in range(len(pk))])
                print bar
            # Evaluate the objective function and its derivatives.
            data        = self.Objective.Full(xk,Ord,verbose=True)
            stepn += 1
            X, G, H = data['X'], data['G'], data['H']
            ndx = norm(dx)
            nxk = norm(xk)
            ngr = norm(G)
            drc = abs(flat(dx)).argmax()

            dX_actual = X - X_prev
            Quality = dX_actual / dX_expect

            if Quality <= 0.25:
                # If the step quality is bad, then we should decrease the trust radius.
                trust = max(ndx*(1./(1+a)), self.mintrust)
            elif Quality >= 0.75 and bump:
                # If the step quality is good, then we should increase the trust radius.  Capeesh?
                # The 'a' factor is how much we should grow or shrink the trust radius each step
                # and the 'b' factor determines how closely we are tied down to the original value.
                # Recommend values 0.5 and 0.5
                trust += a*trust*np.exp(-b*(trust/self.trust0 - 1))
            if X > X_prev:
                color = "\x1b[91m"
                # Toggle switch for rejection (experimenting with no rejection)
                Rejects = True
                if Rejects:
                    xk = xk_prev.copy()
                    G = G_prev.copy()
                    H = H_stor.copy()
                    continue
            else:
                color = "\x1b[92m"
                X_best = X
                ehist = np.append(ehist, X)
            # Hessian update.
            if b_BFGS:
                Hnew = H_stor.copy()
                Dx   = col(xk - xk_prev)
                Dy   = col(G  - G_prev)
                Mat1 = (Dy*Dy.T)/(Dy.T*Dx)[0,0]
                Mat2 = ((Hnew*Dx)*(Hnew*Dx).T)/(Dx.T*Hnew*Dx)[0,0]
                Hnew += Mat1-Mat2
                H = Hnew.copy()
                
            G_prev  = G.copy()
            H_stor  = H.copy()
            xk_prev = xk.copy()
            X_prev  = X
        
        bar = printcool("Final objective function value\nFull: % .6e  Un-penalized: % .6e" % (data['X'],data['X0']), '@', bold=True, color=2)
        return xk


    def step(self, xk, data, trust):
        """ Computes the Newton-Raphson or BFGS step.

        The step is given by the inverse of the Hessian times the gradient.
        There are some extra considerations here:

        First, certain eigenvalues of the Hessian may be negative.  Then the
        NR optimization will take us to a saddle point and not a true minimum.
        In these instances, we mix in some steepest descent by adding a multiple
        of the identity matrix to the Hessian.

        Second, certain eigenvalues may be very small.  If the Hessian is close to
        being singular, then we also add in some steepest descent.

        Third, certain components of the gradient / Hessian are strictly zero, or they
        are excluded from our optimization.  These components are explicitly deleted
        when we do the Hessian inversion, and reinserted as a zero in the step.

        Fourth, we rescale the step size back to the trust radius.

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
            print "Hessian has a small or negative eigenvalue (%.1e), mixing in some steepest descent (%.1e) to correct this." % (Emin, self.eps - Emin)
            H += (self.eps - Emin)*np.eye(H.shape[0])

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
            def hyper_solver(L):
                dx0 = np.zeros(len(xkd),dtype=float)
                #dx0 = np.delete(dx0, self.excision)
                HL = H + L**2*np.diag(np.diag(H))
                HYP = Hyper(HL, self.Objective.Penalty)
                Opt1 = optimize.fmin_bfgs(HYP.compute_val,dx0,fprime=HYP.compute_grad,gtol=1e-5,full_output=True,disp=0)
                Opt2 = optimize.fmin_bfgs(HYP.compute_val,-xkd,fprime=HYP.compute_grad,gtol=1e-5,full_output=True,disp=0)
                #Opt1 = optimize.fmin(HYP.compute_val,dx0,full_output=True,disp=0)
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
            dx = -solve(H, G)          # Take Newton Raphson Step ; use -1*G if want steepest descent.
            dx = flat(dx)
            for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
                dx = np.insert(dx, i, 0)
            def para_solver(L):
                HT = H + L**2*np.diag(np.diag(H))
                dx = -solve(HT, G)
                sol = flat(0.5*row(dx)*np.mat(H)*col(dx))[0] + np.dot(dx,G)
                for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
                    dx = np.insert(dx, i, 0)
                return dx, sol
    
        def solver(L):
            return hyper_solver(L) if self.bhyp else para_solver(L)
    
        def trust_fun(L):
            N = norm(solver(L)[0])
            print "\rHessian diagonal scaling = %.1e: found length %.1e" % (1+L**2,N),
            return (N - trust)**2

        bump = False
        dx, expect = solver(0)
        dxnorm = norm(dx)
        if dxnorm > trust:
            bump = True
            # Tried a few optimizers here, seems like Brent works well.
            LOpt = optimize.brent(trust_fun,brack=(0.0,3.0),tol=1e-4)
            dx, expect = solver(LOpt)
            dxnorm = norm(dx)
            print "\rLevenberg-Marquardt: %s step found (length %.3e), Hessian diagonal is scaled by % .3f" % ('hyperbolic-regularized' if self.bhyp else 'Newton-Raphson', dxnorm, 1+LOpt**2)
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
                if verbose: print
                Answer = func(mvals,Order=0,verbose=verbose)['X']
                dx = (my_func.x_best - Answer) if my_func.x_best != None else 0.0
                if Answer < my_func.x_best or my_func.x_best == None:
                    color = "\x1b[92m"
                    my_func.x_best = Answer
                else:
                    color = "\x1b[91m"
                if verbose:
                    if self.print_vals:
                        print "k=", ' '.join(["% .4f" % i for i in mvals])
                    print "X2= %s%12.3e\x1b[0m d(X2)= %12.3e" % (color,Answer,dx)
                if Answer != Answer:
                    return 1e10
                else:
                    return Answer
            my_func.x_best = None
            return my_func
        def gwrap(func,verbose=True):
            def my_gfunc(mvals):
                if verbose: print
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
                        print "k=", ' '.join(["% .4f" % i for i in mvals])
                    print "|Grad|= %12.3e X2= %s%12.3e\x1b[0m d(X2)= %12.3e" % (norm(Answer),color,Objective,dx)
                    print
                return Answer
            my_gfunc.x_best = None
            return my_gfunc
        if Algorithm == "powell":
            printcool("Minimizing Objective Function using Powell's Method" , color=7, bold=1)
            return optimize.fmin_powell(xwrap(self.Objective.Full),self.mvals0,ftol=self.conv_obj,xtol=self.conv_stp,maxiter=self.maxstep)
        elif Algorithm == "simplex":
            printcool("Minimizing Objective Function using Simplex Method" , color=7, bold=1)
            return optimize.fmin(xwrap(self.Objective.Full),self.mvals0,ftol=self.conv_obj,xtol=self.conv_stp,maxiter=self.maxstep,maxfun=self.maxstep*10)
        elif Algorithm == "anneal":
            printcool("Minimizing Objective Function using Simulated Annealing" , color=7, bold=1)
            return optimize.anneal(xwrap(self.Objective.Full),self.mvals0,lower=-1*self.trust0*np.ones(self.np),upper=self.trust0*np.ones(self.np),schedule='boltzmann')
        elif Algorithm == "cg":
            printcool("Minimizing Objective Function using Conjugate Gradient" , color=7, bold=1)
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
            return np.vstack((self.mvals0.copy(),np.random.randn(PopSize, self.np)*self.trust0)) / (self.np ** 0.5)
            #return np.vstack((self.mvals0.copy(),generate_fresh(PopSize, self.np)))

        def calculate_fitness(pop):
            return [self.Objective.Full(i,Order=0,verbose=False)['X'] for i in pop]

        def sort_by_fitness(fits):
            return np.sort(fits), np.argsort(fits)

        def generate_new_population(sorted, pop):
            newpop = pop[sorted[1]]
            # Individuals in this range are kept
            a = range(KeepNum)
            print "Keeping:", a
            random.shuffle(a)
            for i in range(0, KeepNum, 2):
                print "%i and %i reproducing to replace %i and %i" % (a[i],a[i+1],len(newpop)-i-2,len(newpop)-i-1)
                newpop[-i-1], newpop[-i-2] = cross_over(newpop[a[i]],newpop[a[i+1]])
            b = range(KeepNum, len(newpop))
            random.shuffle(b)
            for i in b[:MutNum]:
                print "Randomly mutating %i" % i
                newpop[i] = mutate(newpop[i])
            return newpop
            
        def xwrap(func,verbose=True):
            def my_func(mvals):
                if verbose: print
                Answer = func(mvals,Order=0,verbose=verbose)['X']
                dx = (my_func.x_best - Answer) if my_func.x_best != None else 0.0
                if Answer < my_func.x_best or my_func.x_best == None:
                    color = "\x1b[92m"
                    my_func.x_best = Answer
                else:
                    color = "\x1b[91m"
                if verbose:
                    print "k=", ' '.join(["% .4f" % i for i in mvals])
                    print "X2= %s%12.3e\x1b[0m d(X2)= %12.3e" % (color,Answer,dx)
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
            print Sorted
            Best[0].append(Sorted[0][0])
            Best[1].append(Population[Sorted[1][0]])
            print Best
            if Gen == self.maxstep: break
            Population = generate_new_population(Sorted, Population)
            Gen += 1

        print Best
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
            print "Syntax error: in the input file please use scan_vals low:hi:nsteps"
            sys.exit(1)
        if len(vals_in) != 3:
            print "Syntax error: in the input file please use scan_vals low:hi:nsteps"
            sys.exit(1)
        idx = [int(i) for i in self.idxnum]
        for j in self.idxname:
            idx += [self.FF.map[i] for i in self.FF.map if j in i]
        idx = set(idx)
        scanvals = np.linspace(vals_in[0],vals_in[1],vals_in[2])
        print vals_in
        print scanvals
        for pidx in idx:
            if MathPhys:
                print "Scanning parameter %i (%s) in the mathematical space" % (pidx,self.FF.plist[pidx])
                vals = self.mvals0.copy()
            else:
                print "Scanning parameter %i (%s) in the physical space" % (pidx,self.FF.plist[pidx])
                for Sim in self.Objective.Simulations:
                    Sim.usepvals = True
                vals = self.FF.pvals0.copy()
            for i in scanvals:
                vals[pidx] = i
                data        = self.Objective.Full(vals,Order=0)
                print "Value = % .4e Objective = % .4e" % (i, data['X'])

    def ScanMVals(self):
        """ Scan through the mathematical parameter space. @see Optimizer::ScanValues """
        self.Scan_Values(1)

    def ScanPVals(self):
        """ Scan through the physical parameter space. @see Optimizer::ScanValues """
        self.Scan_Values(0)

    def SinglePoint(self):
        """ A single-point objective function computation. """
        data        = self.Objective.Full(self.mvals0,Order=0,verbose=True)
        print "The objective function is:", data['X']

    def Gradient(self):
        """ A single-point gradient computation. """
        data        = self.Objective.Full(self.mvals0,Order=1)
        print data['X']
        print data['G']
        print data['H']

    def Hessian(self):
        """ A single-point Hessian computation. """
        data        = self.Objective.Full(self.mvals0,Order=2)
        print data['X']
        print data['G']
        print data['H']

    def FDCheckG(self):
        """ Finite-difference checker for the objective function gradient.

        For each element in the gradient, use a five-point finite difference
        stencil to compute a finite-difference derivative, and compare it to
        the analytic result.

        """

        Adata        = self.Objective.Full(self.mvals0,Order=1)['G']
        Fdata        = np.zeros(self.np,dtype=float)
        printcool("Checking first derivatives by finite difference!\n%-8s%-20s%13s%13s%13s%13s" \
                  % ("Index", "Parameter ID","Analytic","Numerical","Difference","Fractional"),bold=1,color=5)
        for i in range(self.np):
            Fdata[i] = f1d7p(fdwrap(self.Objective.Full,self.mvals0,i,'X',Order=0),self.h)
            Denom = max(abs(Adata[i]),abs(Fdata[i]))
            Denom = Denom > 1e-8 and Denom or 1e-8
            D = Adata[i] - Fdata[i]
            Q = (Adata[i] - Fdata[i])/Denom
            cD = abs(D) > 0.5 and "\x1b[1;91m" or (abs(D) > 1e-2 and "\x1b[91m" or (abs(D) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            cQ = abs(Q) > 0.5 and "\x1b[1;91m" or (abs(Q) > 1e-2 and "\x1b[91m" or (abs(Q) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            print "\r    %-8i%-20s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m" \
                  % (i, self.FF.plist[i][:20], Adata[i], Fdata[i], cD, D, cQ, Q)

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
        Fdata        = np.zeros((self.np,self.np),dtype=float)
        printcool("Checking second derivatives by finite difference!\n%-8s%-20s%-20s%13s%13s%13s%13s" \
                  % ("Index", "Parameter1 ID", "Parameter2 ID", "Analytic","Numerical","Difference","Fractional"),bold=1,color=5)

        # Whee, our double-wrapped finite difference second derivative!
        def wrap2(mvals0,pidxi,pidxj):
            def func1(arg):
                mvals = list(mvals0)
                mvals[pidxj] += arg
                return f1d5p(fdwrap(self.Objective.Full,mvals,pidxi,'X',Order=0),self.h)
            return func1
        
        for i in range(self.np):
            for j in range(i,self.np):
                Fdata[i,j] = f1d5p(wrap2(self.mvals0,i,j),self.h)
                Denom = max(abs(Adata[i,j]),abs(Fdata[i,j]))
                Denom = Denom > 1e-8 and Denom or 1e-8
                D = Adata[i,j] - Fdata[i,j]
                Q = (Adata[i,j] - Fdata[i,j])/Denom
                cD = abs(D) > 0.5 and "\x1b[1;91m" or (abs(D) > 1e-2 and "\x1b[91m" or (abs(D) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
                cQ = abs(Q) > 0.5 and "\x1b[1;91m" or (abs(Q) > 1e-2 and "\x1b[91m" or (abs(Q) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
                print "\r    %-8i%-20s%-20s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m" \
                      % (i, self.FF.plist[i][:20], self.FF.plist[j][:20], Adata[i,j], Fdata[i,j], cD, D, cQ, Q)

    def readchk(self):
        """ Read the checkpoint file for the main optimizer. """
        self.chk = {}
        if self.rchk_fnm != None:
            absfnm = os.path.join(self.root,self.rchk_fnm)
            if os.path.exists(absfnm):
                self.chk = pickle.load(open(absfnm))
            else:
                print "\x1b[1;93mWARNING:\x1b[0m read_chk is set to True, but checkpoint file not loaded (wrong filename or doesn't exist?)"
        return self.chk

    def writechk(self):
        """ Write the checkpoint file for the main optimizer. """
        if self.wchk_fnm != None:
            print "Writing the checkpoint file %s" % self.wchk_fnm
            with open(os.path.join(self.root,self.wchk_fnm),'w') as f: pickle.dump(self.chk,f)
        
