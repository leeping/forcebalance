"""@package forcebalance.objective

ForceBalance objective function."""

import sys
import inspect
#from implemented import Implemented_Targets
import numpy as np
from collections import defaultdict, OrderedDict
import forcebalance
from forcebalance.finite_difference import in_fd
from forcebalance.nifty import printcool_dictionary, createWorkQueue, getWorkQueue, wq_wait
import datetime
import traceback
from forcebalance.output import getLogger
logger = getLogger(__name__)

try:
    from forcebalance.gmxio import AbInitio_GMX, BindingEnergy_GMX, Liquid_GMX, Lipid_GMX, Interaction_GMX, Moments_GMX, Vibration_GMX, Thermo_GMX
except:
    logger.warning(traceback.format_exc())
    logger.warning("Gromacs module import failed\n")

try:
    from forcebalance.tinkerio import AbInitio_TINKER, Vibration_TINKER, BindingEnergy_TINKER, Moments_TINKER, Interaction_TINKER, Liquid_TINKER
except:
    logger.warning(traceback.format_exc())
    logger.warning("Tinker module import failed\n")

try:
    from forcebalance.openmmio import AbInitio_OpenMM, Liquid_OpenMM, Interaction_OpenMM, BindingEnergy_OpenMM, Moments_OpenMM, Hydration_OpenMM
except:
    logger.warning(traceback.format_exc())
    logger.warning("OpenMM module import failed; check OpenMM package\n")

try:
    from forcebalance.abinitio_internal import AbInitio_Internal
except:
    logger.warning(traceback.format_exc())
    logger.warning("Internal energy fitting module import failed\n")

try:
    from forcebalance.counterpoise import Counterpoise
except:
    logger.warning(traceback.format_exc())
    logger.warning("Counterpoise module import failed\n")

try:
    from forcebalance.amberio import AbInitio_AMBER, Interaction_AMBER, Vibration_AMBER
except:
    logger.warning(traceback.format_exc())
    logger.warning("Amber module import failed\n")

try:
    from forcebalance.psi4io import THCDF_Psi4, RDVR3_Psi4
except:
    logger.warning(traceback.format_exc())
    logger.warning("PSI4 module import failed\n")

try:
    from forcebalance.target import RemoteTarget
except:
    logger.warning(traceback.format_exc())
    logger.warning("Remote Target import failed\n")

## The table of implemented Targets
Implemented_Targets = {
    'ABINITIO_GMX':AbInitio_GMX,
    'ABINITIO_TINKER':AbInitio_TINKER,
    'ABINITIO_OPENMM':AbInitio_OpenMM,
    'ABINITIO_AMBER':AbInitio_AMBER,
    'ABINITIO_INTERNAL':AbInitio_Internal,
    'VIBRATION_TINKER':Vibration_TINKER,
    'VIBRATION_GMX':Vibration_GMX,
    'VIBRATION_AMBER':Vibration_AMBER,
    'THERMO_GMX':Thermo_GMX,
    'LIQUID_OPENMM':Liquid_OpenMM,
    'LIQUID_TINKER':Liquid_TINKER, 
    'LIQUID_GMX':Liquid_GMX, 
    'LIPID_GMX':Lipid_GMX, 
    'COUNTERPOISE':Counterpoise,
    'THCDF_PSI4':THCDF_Psi4,
    'RDVR3_PSI4':RDVR3_Psi4,
    'INTERACTION_AMBER':Interaction_AMBER,
    'INTERACTION_GMX':Interaction_GMX,
    'INTERACTION_TINKER':Interaction_TINKER,
    'INTERACTION_OPENMM':Interaction_OpenMM,
    'BINDINGENERGY_TINKER':BindingEnergy_TINKER,
    'BINDINGENERGY_GMX':BindingEnergy_GMX,
    'BINDINGENERGY_OPENMM':BindingEnergy_OpenMM,
    'MOMENTS_TINKER':Moments_TINKER,
    'MOMENTS_GMX':Moments_GMX,
    'MOMENTS_OPENMM':Moments_OpenMM,
    'HYDRATION_OPENMM':Hydration_OpenMM,
    'REMOTE_TARGET':RemoteTarget,
    }

## This is the canonical lettering that corresponds to : objective function, gradient, Hessian.
Letters = ['X','G','H']

class Objective(forcebalance.BaseClass):
    """ Objective function.
    
    The objective function is a combination of contributions from the different
    fitting targets.  Basically, it loops through the targets, gets their 
    contributions to the objective function and then sums all of them
    (although more elaborate schemes are conceivable).  The return value is the
    same data type as calling the target itself: a dictionary containing
    the objective function, the gradient and the Hessian.

    The penalty function is also computed here; it keeps the parameters from straying
    too far from their initial values.

    @param[in] mvals The mathematical parameters that enter into computing the objective function
    @param[in] Order The requested order of differentiation
    """
    def __init__(self, options, tgt_opts, forcefield):

        super(Objective, self).__init__(options)
        self.set_option(options, 'penalty_type')
        self.set_option(options, 'penalty_additive')
        self.set_option(options, 'penalty_multiplicative')
        self.set_option(options, 'penalty_hyperbolic_b')
        self.set_option(options, 'penalty_alpha')
        self.set_option(options, 'normalize_weights')
        ## Work Queue Port (The specific target itself may or may not actually use this.)
        self.set_option(options, 'wq_port')
        ## Asynchronous objective function evaluation (i.e. execute Work Queue and local objective concurrently.)
        self.set_option(options, 'asynchronous')

        ## The list of fitting targets
        self.Targets = []
        for opts in tgt_opts:
            if opts['type'] not in Implemented_Targets:
                logger.error('The target type \x1b[1;91m%s\x1b[0m is not implemented!\n' % opts['type'])
                raise RuntimeError
            # Create a target object.  This is done by looking up the
            # Target class from the Implemented_Targets dictionary
            # using opts['type'] as the key.  The object is created by
            # passing (options, opts, forcefield) to the constructor.
            if opts["remote"] and self.wq_port != 0: Tgt = forcebalance.target.RemoteTarget(options, opts, forcefield)
            else: Tgt = Implemented_Targets[opts['type']](options,opts,forcefield)
            self.Targets.append(Tgt)
            printcool_dictionary(Tgt.PrintOptionDict,"Setup for target %s :" % Tgt.name)
        if len(set([Tgt.name for Tgt in self.Targets])) != len([Tgt.name for Tgt in self.Targets]):
            logger.error("The list of target names is not unique!\n")
            raise RuntimeError
        ## The force field (it seems to be everywhere)
        self.FF = forcefield
        ## Initialize the penalty function.
        self.Penalty = Penalty(self.penalty_type,forcefield,self.penalty_additive,
                               self.penalty_multiplicative,self.penalty_hyperbolic_b,
                               self.penalty_alpha)
        ## Obtain the denominator.
        if self.normalize_weights:
            self.WTot = np.sum([i.weight for i in self.Targets])
        else:
            self.WTot = 1.0
        self.ObjDict = OrderedDict()
        self.ObjDict_Last = OrderedDict()

        # Create the work queue here.
        if self.wq_port != 0:
            createWorkQueue(self.wq_port)
            logger.info('Work Queue is listening on %d\n' % self.wq_port)

        printcool_dictionary(self.PrintOptionDict, "Setup for objective function :")

        
    def Target_Terms(self, mvals, Order=0, verbose=False, customdir=None):
        ## This is the objective function; it's a dictionary containing the value, first and second derivatives
        Objective = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np,self.FF.np))}
        # Loop through the targets, stage the directories and submit the Work Queue processes.
        for Tgt in self.Targets:
            Tgt.stage(mvals, AGrad = Order >= 1, AHess = Order >= 2, customdir=customdir)
        if self.asynchronous:
            # Asynchronous evaluation of objective function and Work Queue tasks.
            # Create a list of the targets, and remove them from the list as they are finished.
            Need2Evaluate = self.Targets[:]
            # This ensures that the OrderedDict doesn't get out of order.
            for Tgt in self.Targets:
                self.ObjDict[Tgt.name] = None
            # Loop through the targets and compute the objective function for ones that are finished.
            while len(Need2Evaluate) > 0:
                for Tgt in Need2Evaluate:
                    if Tgt.wq_complete():
                        # List of functions that I can call.
                        Funcs   = [Tgt.get_X, Tgt.get_G, Tgt.get_H]
                        # Call the appropriate function
                        Ans = Funcs[Order](mvals, customdir=customdir)
                        # Print out the qualitative indicators
                        if verbose:
                            Tgt.meta_indicate(customdir=customdir)
                        # Note that no matter which order of function we call, we still increment the objective / gradient / Hessian the same way.
                        if not in_fd():
                            self.ObjDict[Tgt.name] = {'w' : Tgt.weight/self.WTot , 'x' : Ans['X']}
                        for i in range(3):
                            Objective[Letters[i]] += Ans[Letters[i]]*Tgt.weight/self.WTot
                        Need2Evaluate.remove(Tgt)
                        break
                    else:
                        pass
        else:
            wq = getWorkQueue()
            if wq is not None:
                wq_wait(wq)
            for Tgt in self.Targets:
                # The first call is always done at the midpoint.
                Tgt.bSave = True
                # List of functions that I can call.
                Funcs   = [Tgt.get_X, Tgt.get_G, Tgt.get_H]
                # Call the appropriate function
                Ans = Funcs[Order](mvals, customdir=customdir)
                # Print out the qualitative indicators
                if verbose:
                    Tgt.meta_indicate(customdir=customdir)
                # Note that no matter which order of function we call, we still increment the objective / gradient / Hessian the same way.
                if not in_fd():
                    self.ObjDict[Tgt.name] = {'w' : Tgt.weight/self.WTot , 'x' : Ans['X']}
                for i in range(3):
                    Objective[Letters[i]] += Ans[Letters[i]]*Tgt.weight/self.WTot
        # The target has evaluated at least once.
        for Tgt in self.Targets:
            Tgt.evaluated = True
        # Safeguard to make sure we don't have exact zeros on Hessian diagonal
        for i in range(self.FF.np):
            if Objective['H'][i,i] == 0.0:
                Objective['H'][i,i] = 1.0
        return Objective

    def Indicate(self):
        """ Print objective function contributions. """
        PrintDict = OrderedDict()
        Total = 0.0
        Change = False
        color = "\x1b[0m"
        for key, val in self.ObjDict.items():
            if key == 'Total' : continue
            color = "\x1b[94m"
            if key in self.ObjDict_Last:
                Change = True
                if self.ObjDict[key] <= self.ObjDict_Last[key]:
                    color = "\x1b[92m"
                elif self.ObjDict[key] > self.ObjDict_Last[key]:
                    color = "\x1b[91m"
            PrintDict[key] = "% 12.5f % 10.3f %s% 16.5e%s" % (val['x'],val['w'],color,val['x']*val['w'],"\x1b[0m")
            if Change:
                xnew = self.ObjDict[key]['x'] * self.ObjDict[key]['w']
                xold = self.ObjDict_Last[key]['x'] * self.ObjDict_Last[key]['w']
                PrintDict[key] += " ( %+10.3e )" % (xnew - xold)
            Total += val['x']*val['w']
        self.ObjDict['Total'] = Total
        if 'Total' in self.ObjDict_Last:
            Change = True
            if self.ObjDict['Total'] <= self.ObjDict_Last['Total']:
                color = "\x1b[92m"
            elif self.ObjDict['Total'] > self.ObjDict_Last['Total']:
                color = "\x1b[91m"
        PrintDict['Total'] = "% 12s % 10s %s% 16.5e%s" % ("","",color,Total,"\x1b[0m")
        if Change:
            xnew = self.ObjDict['Total']
            xold = self.ObjDict_Last['Total']
            PrintDict['Total'] += " ( %+10.3e )" % (xnew - xold)
            Title = "Objective Function Breakdown\n %-20s %55s" % ("Target Name", "Residual  x  Weight  =  Contribution (Current-Prev)")
        else:
            Title = "Objective Function Breakdown\n %-20s %40s" % ("Target Name", "Residual  x  Weight  =  Contribution")
        printcool_dictionary(PrintDict,color=4,title=Title)
        return

    def Full(self, vals, Order=0, verbose=False, customdir=None):
        Objective = self.Target_Terms(vals, Order, verbose, customdir)
        ## Compute the penalty function.
        if self.FF.use_pvals:
            Extra = self.Penalty.compute(self.FF.create_mvals(vals),Objective)
        else:
            Extra = self.Penalty.compute(vals,Objective)
        Objective['X0'] = Objective['X']
        Objective['G0'] = Objective['G'].copy()
        Objective['H0'] = Objective['H'].copy()
        if not in_fd():
            self.ObjDict['Regularization'] = {'w' : 1.0, 'x' : Extra[0]}
            if verbose:
                self.Indicate()
        for i in range(3):
            Objective[Letters[i]] += Extra[i]
        return Objective

class Penalty:
    """ Penalty functions for regularizing the force field optimizer.

    The purpose for this module is to improve the behavior of our optimizer;
    essentially, our problem is fraught with 'linear dependencies', a.k.a.
    directions in the parameter space that the objective function does not
    respond to.  This would happen if a parameter is just plain useless, or
    if there are two or more parameters that describe the same thing.

    To accomplish these objectives, a penalty function is added to the
    objective function.  Generally, the more the parameters change (i.e.
    the greater the norm of the parameter vector), the greater the
    penalty.  Note that this is added on after all of the other
    contributions have been computed.  This only matters if the penalty
    'multiplies' the objective function: Obj + Obj*Penalty, but we also
    have the option of an additive penalty: Obj + Penalty.

    Statistically, this is called regularization.  If the penalty function
    is the norm squared of the parameter vector, it is called ridge regression.
    There is also the option of using simply the norm, and this is called lasso,
    but I think it presents problems for the optimizer that I need to work out.

    Note that the penalty functions can be considered as part of a 'maximum
    likelihood' framework in which we assume a PRIOR PROBABILITY of the
    force field parameters around their initial values.  The penalty function
    is related to the prior by an exponential.  Ridge regression corresponds
    to a Gaussian prior and lasso corresponds to an exponential prior.  There
    is also 'elastic net regression' which interpolates between Gaussian
    and exponential using a tuning parameter.

    Our priors are adjustable too - there is one parameter, which is the width
    of the distribution.  We can even use a noninformative prior for the
    distribution widths (hyperprior!).  These are all important things to
    consider later.

    Importantly, note that here there is no code that treats the distribution
    width.  That is because the distribution width is wrapped up in the
    rescaling factors, which is essentially a coordinate transformation
    on the parameter space.  More documentation on this will follow, perhaps
    in the 'rsmake' method.

    """
    Pen_Names = {'HYP' : 1, 'HYPER' : 1, 'HYPERBOLIC' : 1, 'L1' : 1, 'HYPERBOLA' : 1,
                      'PARA' : 2, 'PARABOLA' : 2, 'PARABOLIC' : 2, 'L2': 2, 'QUADRATIC' : 2,
                      'FUSE' : 3, 'FUSION' : 3, 'FUSE_L0' : 4, 'FUSION_L0' : 4, 'FUSION-L0' : 4,
                      'FUSE-BARRIER' : 5, 'FUSE-BARRIER' : 5, 'FUSE_BARRIER' : 5, 'FUSION_BARRIER' : 5}

    def __init__(self, User_Option, ForceField, Factor_Add=0.0, Factor_Mult=0.0, Factor_B=0.1, Alpha=1.0):
        self.fadd = Factor_Add
        self.fmul = Factor_Mult
        self.a    = Alpha
        self.b    = Factor_B
        self.FF   = ForceField
        self.ptyp = self.Pen_Names[User_Option.upper()]
        self.Pen_Tab = {1 : self.HYP, 2: self.L2_norm, 3: self.FUSE, 4:self.FUSE_L0, 5: self.FUSE_BARRIER}
        if User_Option.upper() == 'L1':
            logger.info("L1 norm uses the hyperbolic penalty, make sure penalty_hyperbolic_b is set sufficiently small\n")
        elif self.ptyp == 1:
            logger.info("Using hyperbolic regularization (Laplacian prior) with strength %.1e (+), %.1e (x) and tightness %.1e\n" % (Factor_Add, Factor_Mult, Factor_B))
        elif self.ptyp == 2:
            logger.info("Using parabolic regularization (Gaussian prior) with strength %.1e (+), %.1e (x)\n" % (Factor_Add, Factor_Mult))
        elif self.ptyp == 3:
            logger.info("Using L1 Fusion Penalty (only relevant for basis set optimizations at the moment) with strength %.1e\n" % Factor_Add)
        elif self.ptyp == 4:
            logger.info("Using L0-L1 Fusion Penalty (only relevant for basis set optimizations at the moment) with strength %.1e and switching distance %.1e\n" % (Factor_Add, Alpha))
        elif self.ptyp == 5:
            logger.info("Using L1 Fusion Penalty with Log Barrier (only relevant for basis set optimizations at the moment) with strength %.1e and barrier distance %.1e\n" % (Factor_Add, Alpha))

        ## Find exponential spacings.
        if self.ptyp in [3,4,5]:
            self.spacings = self.FF.find_spacings()
            printcool_dictionary(self.spacings, title="Starting zeta spacings\n(Pay attention to these)")

    def compute(self, mvals, Objective):
        K0, K1, K2 = self.Pen_Tab[self.ptyp](mvals)
        if self.fadd > 0.0:
            XAdd = K0 * self.fadd
            GAdd = K1 * self.fadd
            HAdd = K2 * self.fadd
        else:
            NP = len(mvals)
            XAdd = 0.0
            GAdd = np.zeros(NP)
            HAdd = np.zeros((NP, NP))
        if self.fmul > 0.0:
            X = Objective['X']
            G = Objective['G']
            H = Objective['H']
            XAdd += ( X*K0 ) * self.fmul
            GAdd += np.array( G*K0 + X*K1 ) * self.fmul
            GK1 = np.reshape(G, (1, -1))*np.reshape(K1, (-1, 1))
            K1G = np.reshape(K1, (1, -1))*np.reshape(G, (-1, 1))
            HAdd += np.array( H*K0+GK1+K1G+X*K2 ) * self.fmul
        return XAdd, GAdd, HAdd

    def L2_norm(self, mvals):
        """
        Harmonic L2-norm constraints.  These are the ones that I use
        the most often to regularize my optimization.

        @param[in] mvals The parameter vector
        @return DC0 The norm squared of the vector
        @return DC1 The gradient of DC0
        @return DC2 The Hessian (just a constant)

        """
        mvals = np.array(mvals)
        DC0 = np.dot(mvals, mvals)
        DC1 = 2*np.array(mvals)
        DC2 = 2*np.eye(len(mvals))
        return DC0, DC1, DC2

    def HYP(self, mvals):
        """
        Hyperbolic constraints.  Depending on the 'b' parameter, the smaller it is,
        the closer we are to an L1-norm constraint.  If we use these, we expect
        a properly-behaving optimizer to make several of the parameters very nearly zero
        (which would be cool).

        @param[in] mvals The parameter vector
        @return DC0 The hyperbolic penalty
        @return DC1 The gradient
        @return DC2 The Hessian

        """
        mvals = np.array(mvals)
        NP = len(mvals)
        sqt   = (mvals**2 + self.b**2)**0.5
        DC0   = np.sum(sqt - self.b)
        DC1   = mvals*(1.0/sqt)
        DC2   = np.diag(self.b**2*(1.0/sqt**3))

        return DC0, DC1, DC2

    def FUSE(self, mvals):
        Groups = defaultdict(list)
        for p, pid in enumerate(self.FF.plist):
            if 'Exponent' not in pid or len(pid.split()) != 1:
                warn_press_key("Fusion penalty currently implemented only for basis set optimizations, where parameters are like this: Exponent:Elem=H,AMom=D,Bas=0,Con=0")
            Data = dict([(i.split('=')[0],i.split('=')[1]) for i in pid.split(':')[1].split(',')])
            if 'Con' not in Data or Data['Con'] != '0':
                warn_press_key("More than one contraction coefficient found!  You should expect the unexpected")
            key = Data['Elem']+'_'+Data['AMom']
            Groups[key].append(p)
        pvals = self.FF.create_pvals(mvals)
        DC0 = 0.0
        DC1 = np.zeros(self.FF.np)
        DC2 = np.zeros(self.FF.np)
        for gnm, pidx in Groups.items():
            # The group of parameters for a particular element / angular momentum.
            pvals_grp = pvals[pidx]
            # The order that the parameters come in.
            Order = np.argsort(pvals_grp)
            # The number of nearest neighbor pairs.
            #print Order
            for p in range(len(Order) - 1):
                # The pointers to the parameter indices.
                pi = pidx[Order[p]]
                pj = pidx[Order[p+1]]
                # pvals[pi] is the SMALLER parameter.
                # pvals[pj] is the LARGER parameter.
                dp = np.log(pvals[pj]) - np.log(pvals[pi])
                # dp = (np.log(pvals[pj]) - np.log(pvals[pi])) / self.spacings[gnm]
                DC0     += (dp**2 + self.b**2)**0.5 - self.b
                DC1[pi] -= dp*(dp**2 + self.b**2)**-0.5
                DC1[pj] += dp*(dp**2 + self.b**2)**-0.5
                # The second derivatives have off-diagonal terms,
                # but we're not using them right now anyway
                # I will implement them if necessary.
                # DC2[pi] -= self.b**2*(dp**2 + self.b**2)**-1.5
                # DC2[pj] += self.b**2*(dp**2 + self.b**2)**-1.5
                #print "pvals[%i] = %.4f, pvals[%i] = %.4f dp = %.4f" % (pi, pvals[pi], pj, pvals[pj], dp), 
                #print "First Derivative = % .4f, Second Derivative = % .4f" % (dp*(dp**2 + self.b**2)**-0.5, self.b**2*(dp**2 + self.b**2)**-1.5)
        return DC0, DC1, np.diag(DC2)

    def FUSE_BARRIER(self, mvals):
        Groups = defaultdict(list)
        for p, pid in enumerate(self.FF.plist):
            if 'Exponent' not in pid or len(pid.split()) != 1:
                warn_press_key("Fusion penalty currently implemented only for basis set optimizations, where parameters are like this: Exponent:Elem=H,AMom=D,Bas=0,Con=0")
            Data = dict([(i.split('=')[0],i.split('=')[1]) for i in pid.split(':')[1].split(',')])
            if 'Con' not in Data or Data['Con'] != '0':
                warn_press_key("More than one contraction coefficient found!  You should expect the unexpected")
            key = Data['Elem']+'_'+Data['AMom']
            Groups[key].append(p)
        pvals = self.FF.create_pvals(mvals)
        DC0 = 0.0
        DC1 = np.zeros(self.FF.np)
        DC2 = np.zeros(self.FF.np)
        for gnm, pidx in Groups.items():
            # The group of parameters for a particular element / angular momentum.
            pvals_grp = pvals[pidx]
            # The order that the parameters come in.
            Order = np.argsort(pvals_grp)
            # The number of nearest neighbor pairs.
            #print Order
            for p in range(len(Order) - 1):
                # The pointers to the parameter indices.
                pi = pidx[Order[p]]
                pj = pidx[Order[p+1]]
                # pvals[pi] is the SMALLER parameter.
                # pvals[pj] is the LARGER parameter.
                dp = np.log(pvals[pj]) - np.log(pvals[pi])
                # dp = (np.log(pvals[pj]) - np.log(pvals[pi])) / self.spacings[gnm]
                DC0     += (dp**2 + self.b**2)**0.5 - self.b - self.a*np.log(dp) + self.a*np.log(self.a)
                DC1[pi] -= dp*(dp**2 + self.b**2)**-0.5 - self.a/dp
                DC1[pj] += dp*(dp**2 + self.b**2)**-0.5 - self.a/dp
                # The second derivatives have off-diagonal terms,
                # but we're not using them right now anyway
                # I will implement them later if necessary.
                # DC2[pi] -= self.b**2*(dp**2 + self.b**2)**-1.5 - self.a/dp**2
                # DC2[pj] += self.b**2*(dp**2 + self.b**2)**-1.5 - self.a/dp**2
                #print "pvals[%i] = %.4f, pvals[%i] = %.4f dp = %.4f" % (pi, pvals[pi], pj, pvals[pj], dp), 
                #print "First Derivative = % .4f, Second Derivative = % .4f" % (dp*(dp**2 + self.b**2)**-0.5, self.b**2*(dp**2 + self.b**2)**-1.5)
        return DC0, DC1, np.diag(DC2)


    def FUSE_L0(self, mvals):
        Groups = defaultdict(list)
        for p, pid in enumerate(self.FF.plist):
            if 'Exponent' not in pid or len(pid.split()) != 1:
                warn_press_key("Fusion penalty currently implemented only for basis set optimizations, where parameters are like this: Exponent:Elem=H,AMom=D,Bas=0,Con=0")
            Data = dict([(i.split('=')[0],i.split('=')[1]) for i in pid.split(':')[1].split(',')])
            if 'Con' not in Data or Data['Con'] != '0':
                warn_press_key("More than one contraction coefficient found!  You should expect the unexpected")
            key = Data['Elem']+'_'+Data['AMom']
            Groups[key].append(p)
        pvals = self.FF.create_pvals(mvals)
        #print "pvals: ", pvals
        DC0 = 0.0
        DC1 = np.zeros(self.FF.np)
        DC2 = np.zeros((self.FF.np,self.FF.np))
        for gnm, pidx in Groups.items():
            # The group of parameters for a particular element / angular momentum.
            pvals_grp = pvals[pidx]
            # The order that the parameters come in.
            Order = np.argsort(pvals_grp)
            # The number of nearest neighbor pairs.
            #print Order
            Contribs = []
            dps = []
            for p in range(len(Order) - 1):
                # The pointers to the parameter indices.
                pi = pidx[Order[p]]
                pj = pidx[Order[p+1]]
                # pvals[pi] is the SMALLER parameter.
                # pvals[pj] is the LARGER parameter.
                dp = np.log(pvals[pj]) - np.log(pvals[pi])
                # dp = (np.log(pvals[pj]) - np.log(pvals[pi])) / self.spacings[gnm]
                dp2b2 = dp**2 + self.b**2
                h   = self.a*((dp2b2)**0.5 - self.b)
                hp  = self.a*(dp*(dp2b2)**-0.5)
                hpp = self.a*(self.b**2*(dp2b2)**-1.5)
                emh = np.exp(-1*h)
                dps.append(dp)
                Contribs.append((1.0 - emh))
                DC0     += (1.0 - emh)
                DC1[pi] -= hp*emh
                DC1[pj] += hp*emh
        # for i in self.FF.redirect:
        #     p = mvals[i]
        #     DC0 += 1e-6*p*p
        #     DC1[i] = 2e-6*p
            
                # The second derivatives have off-diagonal terms,
                # but we're not using them right now anyway
                #DC2[pi,pi] += (hpp - hp**2)*emh
                #DC2[pi,pj] -= (hpp - hp**2)*emh
                #DC2[pj,pi] -= (hpp - hp**2)*emh
                #DC2[pj,pj] += (hpp - hp**2)*emh
                #print "pvals[%i] = %.4f, pvals[%i] = %.4f dp = %.4f" % (pi, pvals[pi], pj, pvals[pj], dp), 
                #print "First Derivative = % .4f, Second Derivative = % .4f" % (dp*(dp**2 + self.b**2)**-0.5, self.b**2*(dp**2 + self.b**2)**-1.5)
            
            #print "grp:", gnm, "dp:", ' '.join(["% .1e" % i for i in dps]), "Contributions:", ' '.join(["% .1e" % i for i in Contribs])

        #print DC0, DC1, DC2
        #print pvals
        #raw_input()

        return DC0, DC1, DC2

    #return self.HYP(mvals)
