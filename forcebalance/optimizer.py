""" @package optimizer Optimization algorithms.

My current implementation is to have a single optimizer class with several methods
contained inside.

@author Lee-Ping Wang
@date 12/2011

"""

import os, pickle, re, sys
import numpy as np
from numpy.linalg import eig, norm, solve
from nifty import col, flat, row, printcool, printcool_dictionary
from finite_difference import f1d7p, f1d5p, fdwrap
import random

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
        
        ## A list of all the things we can ask the optimizer to do.
        self.OptTab    = {'NEWTONRAPHSON'     : self.NewtonRaphson, 
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
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The objective function (needs to pass in when I instantiate)
        self.Objective = Objective
        ## The fitting simulations
        self.Sims      = Simulations
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
            self.FF.make('result',xk,False)

        ## Write out stuff to checkpoint file
        self.writechk()
            
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
        printcool( "Main Optimizer\n%s Mode" % ("BFGS" if b_BFGS else "Newton-Raphson"), color=7, bold=1)
        # First, set a bunch of starting values
        #Ord         = 1 if b_BFGS else 2
        ## @todo I need to accommodate the use case:
        ## BFGS update but with Hessian elements still. :P
        Ord         = 2
        if all(i in self.chk for i in ['xk','X','G','H','ehist','x_best','xk_prev','trust']):
            print "Reading initial objective, gradient, Hessian from checkpoint file"
            xk, X, G, H, ehist     = self.chk['xk'], self.chk['X'], self.chk['G'], self.chk['H'], self.chk['ehist']
            X_best, xk_prev, trust = self.chk['x_best'], self.chk['xk_prev'], self.chk['trust']
        else:
            xk       = self.mvals0.copy()
            print
            data     = self.Objective(xk,Ord,verbose=True)
            X, G, H  = data['X'], data['G'], data['H']
            ehist    = np.array([X])
            xk_prev  = xk.copy()
            trust    = self.trust0
            X_best   = X

        G_prev   = G.copy()
        H_stor   = H.copy()
        stepn  = 0
        ndx    = 0.0
        color  = "\x1b[97m"
        while 1: # Loop until convergence is reached.
            ## Put data into the checkpoint file
            self.chk = {'xk': xk, 'X' : X, 'G' : G, 'H': H, 'ehist': ehist,
                        'x_best': X_best,'xk_prev': xk_prev, 'trust': trust}
            if self.wchk_step:
                self.writechk()
            nxk = norm(xk)
            ngr = norm(G)
            stdfront = len(ehist) > 10 and np.std(np.sort(ehist)[:10]) or (len(ehist) > 0 and np.std(ehist) or 0.0)
            print "\n%6s%12s%12s%12s%14s%12s" % ("Step", "  |k|  ","  |dk|  "," |grad| ","    -=X2=-  ","Stdev(X2)")
            print "%6i%12.3e%12.3e%12.3e%s%14.5e\x1b[0m%12.3e" % (stepn, nxk, ndx, ngr, color, X, stdfront)
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
            # Take a step in the parameter space.
            print "Taking a step"
            print G, H
            dx, over = self.step(G, H, trust)
            print dx
            xk += dx
            # Evaluate the objective function and its derivatives.
            print
            data        = self.Objective(xk,Ord,verbose=True)
            X, G, H = data['X'], data['G'], data['H']
            # if pkg.efweight > 0.99:
            #     dFc, patomax = cartesian_dforce(pkg)
            ndx = norm(dx)
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
                trust += over and 0.5*trust*np.exp(-0.5*(trust/self.trust0 - 1)) or 0
                X_best = X
                # So here's the deal.  I'm getting steepest-descent badness in some of the parameters (e.g. virtual site positions)
                # The solution would be to build up a BFGS quasi-Hessian but only in that parameter block, since we have exact second
                # derivatives for everything else.  I will leave the cross-terms alone.
                Hnew = np.mat(H.copy())
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
                ehist = np.append(ehist, X)
            drc = abs(flat(dx)).argmax()
            stepn += 1
            
        return xk

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
        G = np.delete(G, self.excision)
        H = np.delete(H, self.excision, axis=0)
        H = np.delete(H, self.excision, axis=1)
        Eig = eig(H)[0]            # Diagonalize Hessian
        Emin = min(Eig)
        if Emin < self.eps:        # Mix in SD step if Hessian minimum eigenvalue is negative
            H += (self.eps - Emin)*np.eye(H.shape[0])
        dx = -solve(H, G)          # Take Newton Raphson Step ; use -1*G if want steepest descent.
        dx = flat(dx)
        for i in self.excision:    # Reinsert deleted coordinates - don't take a step in those directions
            dx = np.insert(dx, i, 0)
        dxnorm = norm(dx)          # Length of step
        over = False
        if dxnorm > trust:
            over = True
            dx *= trust / dxnorm   # Normalize step length (Trust region)
        return dx, over

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
                    print "k=", ' '.join(["% .4f" % i for i in mvals])
                    print "|Grad|= %12.3e X2= %s%12.3e\x1b[0m d(X2)= %12.3e" % (norm(Answer),color,Objective,dx)
                    print
                return Answer
            my_gfunc.x_best = None
            return my_gfunc
        if Algorithm == "powell":
            printcool("Minimizing Objective Function using Powell's Method" , color=7, bold=1)
            return optimize.fmin_powell(xwrap(self.Objective),self.mvals0,ftol=self.conv_obj,xtol=self.conv_stp,maxiter=self.maxstep)
        elif Algorithm == "simplex":
            printcool("Minimizing Objective Function using Simplex Method" , color=7, bold=1)
            return optimize.fmin(xwrap(self.Objective),self.mvals0,ftol=self.conv_obj,xtol=self.conv_stp,maxiter=self.maxstep,maxfun=self.maxstep*10)
        elif Algorithm == "anneal":
            printcool("Minimizing Objective Function using Simulated Annealing" , color=7, bold=1)
            return optimize.anneal(xwrap(self.Objective),self.mvals0,lower=-1*self.trust0*np.ones(self.np),upper=self.trust0*np.ones(self.np),schedule='boltzmann')
        elif Algorithm == "cg":
            printcool("Minimizing Objective Function using Conjugate Gradient" , color=7, bold=1)
            return optimize.fmin_cg(xwrap(self.Objective,verbose=False),self.mvals0,fprime=gwrap(self.Objective),gtol=self.conv_grd)

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
            return [self.Objective(i,Order=0,verbose=False)['X'] for i in pop]

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
                for Sim in self.Sims:
                    Sim.usepvals = True
                vals = self.FF.pvals0.copy()
            for i in scanvals:
                vals[pidx] = i
                data        = self.Objective(vals,Order=0)
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
        Fdata        = np.zeros(self.np,dtype=float)
        printcool("Checking first derivatives by finite difference!\n%-8s%-20s%13s%13s%13s%13s" \
                  % ("Index", "Parameter ID","Analytic","Numerical","Difference","Fractional"),bold=1,color=5)
        for i in range(self.np):
            Fdata[i] = f1d7p(fdwrap(self.Objective,self.mvals0,i,'X',Order=0),self.h)
            Denom = max(abs(Adata[i]),abs(Fdata[i]))
            Denom = Denom > 1e-8 and Denom or 1e-8
            D = Adata[i] - Fdata[i]
            Q = (Adata[i] - Fdata[i])/Denom
            cD = abs(D) > 0.5 and "\x1b[1;91m" or (abs(D) > 1e-2 and "\x1b[91m" or (abs(D) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            cQ = abs(Q) > 0.5 and "\x1b[1;91m" or (abs(Q) > 1e-2 and "\x1b[91m" or (abs(Q) > 1e-5 and "\x1b[93m" or "\x1b[92m"))
            print "    %-8i%-20s% 13.4e% 13.4e%s% 13.4e%s% 13.4e\x1b[0m" \
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
        Adata        = self.Objective(self.mvals0,Order=2)['H']
        Fdata        = np.zeros((self.np,self.np),dtype=float)
        printcool("Checking second derivatives by finite difference!\n%-8s%-20s%-20s%13s%13s%13s%13s" \
                  % ("Index", "Parameter1 ID", "Parameter2 ID", "Analytic","Numerical","Difference","Fractional"),bold=1,color=5)

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
        
