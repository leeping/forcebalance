""" @package optimizer Optimization algorithms.

My current implementation is to have a single optimizer class with several methods
contained inside.

@todo I might want to sample over different force fields and store past parameters
@todo Pickle-loading is helpful mainly for non-initial parameter values, for reproducibility
@todo Read in parameters from input file, that would be nice

@author Lee-Ping Wang
@date 12/2011

"""

import os
import sys
import re
from numpy import append, array, delete, exp, eye, insert, linspace, mat, sort, std, zeros
from numpy.linalg import eig, norm, solve
from nifty import col, flat, row, printcool
from finite_difference import f1d5p, fdwrap

class Optimizer(object):
    """ Optimizer class.  Contains several methods for numerical optimization.

    For various reasons, the optimizer depends on the force field and fitting
    simulations (i.e. we cannot treat it as a fully independent numerical optimizer).
    The dependency is rather weak which suggests that I can remove it someday.
    """
    
    def __init__(self,options,Objective,FF,Simulations):
        """ Instantiation of the optimizer.

        The optimizer depends on both the FF and the fitting simulations so there
        is a chain of dependencies: FF --> FitSim --> Optimizer, and FF --> Optimizer

        Here's what we do:
        - Take options from the parser
        - Pass in the objective function, force field, all fitting simulations

        """
        #======================================#
        # Options that are given by the parser #
        #======================================#
        ## Initial step size trust radius
        self.trust0    = options['trust0']
        ## Lower bound on Hessian eigenvalue (below this, we add in steepest descent)
        self.eps       = options['eig_lowerbound']
        ## Step size for numerical finite difference
        self.h         = options['finite_difference_h']
        ## Function value convergence threshold
        self.conv_obj  = options['convergence_objective']
        ## Step size convergence threshold
        self.conv_stp  = options['convergence_step']
        ## Maximum number of optimization steps
        self.maxstep   = options['maxstep']
        ## For scan[mp]vals: The parameter index to scan over
        self.idxnum    = options['scanindex_num']
        ## For scan[mp]vals: The parameter name to scan over, it just looks up an index
        self.idxname   = options['scanindex_name']
        ## For scan[mp]vals: The values that are fed into the scanner
        self.scan_vals = options['scan_vals']
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The objective function (needs to pass in when I instantiate)
        self.Objective = Objective
        ## The fitting simulations
        self.Sims      = Simulations
        ## The force field itself
        self.FF        = FF
        ## A list of all the things we can ask the optimizer to do.
        self.OptTab    = {'NEWTONRAPHSON' : self.NewtonRaphson, 
                          'BFGS'          : self.BFGS,
                          'POWELL'        : self.Powell,
                          'SIMPLEX'       : self.Simplex,
                          'ANNEAL'        : self.Anneal,
                          'SCAN_MVALS'    : self.ScanMVals,
                          'SCAN_PVALS'    : self.ScanPVals,
                          'SINGLE'        : self.SinglePoint,
                          'GRADIENT'      : self.Gradient,
                          'HESSIAN'       : self.Hessian,
                          'FDCHECKG'      : self.FDCheckG,
                          'FDCHECKH'      : self.FDCheckH
                          }
        
        #======================================#
        #    Variables from the force field    #
        #======================================#
        ## The indices to be excluded from the Hessian update
        self.excision  = list(FF.excision)
        ## The original parameter values
        self.mvals0    = FF.mvals0.copy()
        ## Number of parameters
        self.np        = FF.np
        
        
    def MainOptimizer(self,b_BFGS=0):
        """ The main ForceBalance trust-radius optimizer.

        Usually this function is called with the BFGS or NewtonRaphson
        method.  I've found the BFGS method to be most efficient,
        especially when we don't have access to the expensive analytic
        second derivatives of the objective function.  If we are computing
        derivatives by finite difference (which we often do), then the
        diagonal elements of the second derivative can also be obtained
        by taking a central difference.

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
        # First, set a bunch of starting values
        xk_prev     = self.mvals0.copy()
        xk          = self.mvals0.copy()
        trust       = self.trust0
        Ord         = b_BFGS and 1 or 2
        data        = self.Objective(xk,Ord)
        ehist       = array([])
        X, G, H = data['X'], data['G'], data['H']
        X_best  = X
        G_prev = G.copy()
        H_stor = H.copy()
        stepn  = 0
        while 1: # Loop until convergence is reached.
            # Take a step in the parameter space.
            dx, over = self.step(G, H, trust)
            xk += dx
            # Evaluate the objective function and its derivatives.
            data        = self.Objective(xk,Ord)
            X, G, H = data['X'], data['G'], data['H']
            # if pkg.efweight > 0.99:
            #     dFc, patomax = cartesian_dforce(pkg)
            ndx = norm(dx)
            nxk = norm(xk)
            if X > X_best:
                goodstep = False
                color = "\x1b[91m"
                # Decrease the trust radius and take a step back, if the step was bad.
                trust = ndx*0.667
                xk = xk_prev.copy()
                # Hmm, we don't want to take a step at the "old" xk but with the "new" G and H, now do we?
                G = G_prev.copy()
                H = H_stor.copy()
            else:
                goodstep = True
                color = "\x1b[92m"
                # Adjust the trust radius using this funky formula
                # Very conservative.  Multiplier is 1.5 and decreases to 1.1 when trust=4trust0
                trust += over and 0.5*trust*exp(-0.5*(trust/self.trust0 - 1)) or 0
                X_best = X
                # So here's the deal.  I'm getting steepest-descent badness in some of the parameters (e.g. virtual site positions)
                # The solution would be to build up a BFGS quasi-Hessian but only in that parameter block, since we have exact second
                # derivatives for everything else.  I will leave the cross-terms alone.
                Hnew = mat(H.copy())
                if b_BFGS:
                    Hnew = H_stor.copy()
                # for i in range(xk.shape[0]):
                #     for j in range(xk.shape[0]):
                #         if b_BFGS:
                #             Hnew[i,j] = H_stor[i,j]
                Dx   = col(xk - xk_prev)
                Dy   = col(G  - G_prev)
                Mat1 = (Dy*Dy.T)/(Dy.T*Dx)[0,0]
                Mat2 = ((Hnew*Dx)*(Hnew*Dx).T)/(Dx.T*Hnew*Dx)[0,0]
                Hnew += Mat1-Mat2
                # After that bit of trickery, Hnew is now updated with BFGS stuff.
                # We now want to put the BFGS-gradients into the Hessian that will be used to take the step.
                if b_BFGS:
                    H = Hnew.copy()
                # for i in range(xk.shape[0]):
                #     for j in range(xk.shape[0]):
                #         if b_BFGS:
                #             H[i,j] = Hnew[i,j]
                G_prev = G.copy()
                H_stor = H.copy()
                # End BFGS stuff
                xk_prev = xk.copy()
                ehist = append(ehist, X)
            stdfront = len(ehist) > 10 and std(sort(ehist)[:10]) or (len(ehist) > 0 and std(ehist) or 0.0)
            drc = abs(flat(dx)).argmax()
            stepn += 1
            print " %12.3e%12.3e%s%14.5e \x1b[0m           " % (stdfront, ndx, color, X)
            if ndx < self.conv_stp:
                print "Convergence criterion reached in step size (%.2e)" % self.conv_stp
                break
            elif stdfront < self.conv_obj and len(ehist) > 10:
                print "Convergence criterion reached for objective function (%.2e)" % self.conv_obj
                break
            elif stepn == self.maxstep:
                print "Maximum number of optimization steps reached (%i)" % stepn
            
    def step(self, G, H, trust):
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
        G = delete(G, self.excision)
        H = delete(H, self.excision, axis=0)
        H = delete(H, self.excision, axis=1)
        Eig = eig(H)[0]            # Diagonalize Hessian
        Emin = min(Eig)
        if Emin < 0:               # Mix in SD step if Hessian minimum eigenvalue is negative
            H += (2*abs(Emin) + self.eps)*eye(H.shape[0])
        elif abs(Emin) < self.eps: # Do the same if Hessian is close to singular
            H += (  abs(Emin) + self.eps)*eye(H.shape[0])
        dx = -solve(H, G)          # Take Newton Raphson Step ; use -1*G if want steepest descent.
        dx = flat(dx)
        for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
            dx = insert(dx, i, 0)
        dxnorm = norm(dx)          # Length of step
        over = False
        if dxnorm > trust:
            over = True
            dx *= trust / dxnorm   # Normalize step length (Trust region)
        return dx, over

    def NewtonRaphson(self):
        """ Optimize the force field parameters using the Newton-Raphson method (@see MainOptimizer) """
        self.MainOptimizer(b_BFGS=0)

    def BFGS(self):
        """ Optimize the force field parameters using the BFGS method; currently the recommended choice (@see MainOptimizer) """
        self.MainOptimizer(b_BFGS=1)

    def ScipyOptimizer(self,Algorithm="None"):
        """ Driver for SciPy optimizations.

        Using any of the SciPy optimizers requires that SciPy is installed.
        This method first defines several wrappers around the objective function that the SciPy
        optimizers can use.  Then it calls the algorith mitself.

        @param[in] Algorithm The optimization algorithm to use, for example 'powell', 'simplex' or 'anneal'

        """
        def xwrap(func):
            def my_func(mvals):
                print ' '.join(["% .4f" % i for i in mvals])
                Answer = func(mvals,Order=0)['X']
                print Answer
                return Answer
            return my_func
        def gwrap(func):
            def my_gfunc(mvals):
                print ' '.join(["% .4f" % i for i in mvals])
                Answer = func(mvals,Order=1)['G']
                print Answer
                return Answer
            return my_gfunc
        if Algorithm == "powell":
            from scipy.optimize import fmin_powell
            fmin_powell(swrap(self.Objective),self.mvals0)
        elif Algorithm == "simplex":
            from scipy.optimize import fmin
            fmin(swrap(self.Objective),self.mvals0)
        elif Algorithm == "anneal":
            from scipy.optimize import anneal
            anneal(swrap(self.Objective),self.mvals0,lower=0,upper=2)

    def Simplex(self):
        """ Use SciPy's built-in simplex algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        self.ScipyOptimizer(Algorithm="simplex")

    def Powell(self):
        """ Use SciPy's built-in Powell direction-set algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        self.ScipyOptimizer(Algorithm="powell")

    def Anneal(self):
        """ Use SciPy's built-in simulated annealing algorithm to optimize the parameters. @see Optimizer::ScipyOptimizer """
        self.ScipyOptimizer(Algorithm="anneal")

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
        scanvals = linspace(vals_in[0],vals_in[1],vals_in[2]+1)
        for pidx in idx:
            if MathPhys:
                print "Scanning parameter %i (%s) in the mathematical space" % (pidx,self.FF.plist[pidx])
                vals = self.mvals0.copy()
            else:
                print "Scanning parameter %i (%s) in the physical space" % (pidx,self.FF.plist[pidx])
                for Sim in self.Sims:
                    Sim.usepvals = True
                vals = self.FF.pvals0.copy()
            for i in scanvals:
                vals[pidx] = i
                data        = self.Objective(vals,Order=0)
                print self.FF.pvals
                print "Value = % .4e Objective = % .4e" % (i, data['X'])

    def ScanMVals(self):
        """ Scan through the mathematical parameter space. @see Optimizer::ScanValues """
        self.Scan_Values(1)

    def ScanPVals(self):
        """ Scan through the physical parameter space. @see Optimizer::ScanValues """
        self.Scan_Values(0)

    def SinglePoint(self):
        """ A single-point objective function computation. """
        data        = self.Objective(self.mvals0,Order=0)
        print data['X']

    def Gradient(self):
        """ A single-point gradient computation. """
        data        = self.Objective(self.mvals0,Order=1)
        print data['X']
        print data['G']
        print data['H']

    def Hessian(self):
        """ A single-point Hessian computation. """
        data        = self.Objective(self.mvals0,Order=2)
        print data['X']
        print data['G']
        print data['H']

    def FDCheckG(self):
        """ Finite-difference checker for the objective function gradient.

        For each element in the gradient, use a five-point finite difference
        stencil to compute a finite-difference derivative, and compare it to
        the analytic result.

        """

        Adata        = self.Objective(self.mvals0,Order=1)['G']
        Fdata        = zeros(self.np,dtype=float)
        printcool("Checking first derivatives by finite difference!\n%-8s%-35s%13s%13s%13s%13s" \
                  % ("Index", "Parameter ID","Analytic","Numerical","Difference","Fractional"),color=5)
        for i in range(self.np):
            Fdata[i] = f1d5p(fdwrap(self.Objective,self.mvals0,i,'X',Order=0),self.h)
            Denom = max(abs(Adata[i]),abs(Fdata[i]))
            Denom = Denom > 1e-8 and Denom or 1e-8
            D = Adata[i] - Fdata[i]
            Q = (Adata[i] - Fdata[i])/Denom
            cD = abs(D) > 0.5 and "\x1b[1;91m" or (abs(D) > 1e-2 and "\x1b[91m" or (abs(D) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            cQ = abs(Q) > 0.5 and "\x1b[1;91m" or (abs(Q) > 1e-2 and "\x1b[91m" or (abs(Q) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            print "    %-8i%35s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m" \
                  % (i, self.FF.plist[i][:35], Adata[i], Fdata[i], cD, D, cQ, Q)

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
        Adata        = self.Objective(self.mvals0,Order=2)['H']
        Fdata        = zeros((self.np,self.np),dtype=float)
        printcool("Checking second derivatives by finite difference!\n%-8s%-35s%13s%13s%13s%13s" \
                  % ("Index", "Parameter ID","Analytic","Numerical","Difference","Fractional"),color=5)

        # Whee, our double-wrapped finite difference second derivative!
        def wrap2(mvals0,pidxi,pidxj):
            def func1(arg):
                mvals = list(mvals0)
                mvals[pidxj] += arg
                return f1d5p(fdwrap(self.Objective,mvals,pidxi,'X',Order=0),self.h)
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
                print "    %-8i%-20s%-20s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m" \
                      % (i, self.FF.plist[i][:20], self.FF.plist[j][:20], Adata[i,j], Fdata[i,j], cD, D, cQ, Q)
