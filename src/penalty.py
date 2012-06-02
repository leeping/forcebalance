# Here I'm just noting a constraint type that I've used in the past.
# It is quartic at the origin and turns into a parabola.
# OMG = 1 - exp(-dp**2/2)
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
    def __init__(self, Penalty_Type, Factor_Add=0.0, Factor_Mult=0.0, Factor_B=0.1):
        self.fadd = Factor_Add
        self.fmul = Factor_Mult
        self.b    = Factor_B
        self.ptyp = Penalty_Type
        self.Pen_Tab = {'HYP' : self.HYP, 'HYPERBOLIC' : self.HYP, 'L2' : self.L2_norm, 'QUADRATIC' : self.L2_norm}

    def compute(self, mvals, Objective):
        X = Objective['X']
        G = Objective['G']
        H = Objective['H']
        np = len(mvals)
        K0, K1, K2 = self.Pen_Tab[self.ptyp.upper()](mvals)
        XAdd = 0.0
        GAdd = zeros(np, dtype=float)
        HAdd = zeros((np, np), dtype=float)
        if self.fadd > 0.0:
            XAdd += K0 * self.fadd
            GAdd += K1 * self.fadd
            HAdd += K2 * self.fadd
        if self.fmul > 0.0:
            XAdd += ( X*K0 ) * self.fmul
            GAdd += array( G*K0 + X*K1 ) * self.fmul
            GK1 = reshape(G, (1, -1))*reshape(K1, (-1, 1))
            K1G = reshape(K1, (1, -1))*reshape(G, (-1, 1))
            HAdd += array( H*K0+GK1+K1G+X*K2 ) * self.fmul
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
        mvals = array(mvals)
        DC0 = dot(mvals, mvals)
        DC1 = 2*array(mvals)
        DC2 = 2*eye(len(mvals))
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
        mvals = array(mvals)
        np = len(mvals)
        DC0   = sum((mvals**2 + self.b**2)**0.5 - self.b)
        DC1   = mvals*(mvals**2 + self.b**2)**-0.5
        DC2   = diag(self.b**2*(mvals**2 + self.b**2)**-1.5)
        return DC0, DC1, DC2
