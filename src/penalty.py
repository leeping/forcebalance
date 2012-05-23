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

from numpy import array, dot, eye, linalg, ones, reshape, zeros

# Here I'm just noting a constraint type that I've used in the past.
# It is quartic at the origin and turns into a parabola.
# OMG = 1 - exp(-dp**2/2)

def L2_norm(mvals):
    """
    Harmonic L2-norm constraints.  These are the ones that I use
    the most often to regularize my optimization.

    @param[in] mvals The parameter vector
    @return DC0 The norm squared of the vector
    @return DC1 The gradient of DC0
    @return DC2 The Hessian (just a constant)

    """
    DC0 = dot(mvals, mvals)
    DC1 = 2*array(mvals)
    DC2 = 2*eye(len(mvals))
    return DC0, DC1, DC2

def L1_norm(mvals):
    """
    Linear L1-norm constraints.  If we use these, we expect
    a properly-behaving optimizer to make several of the parameters zero
    (which would be cool).  However, I haven't tried it out yet.

    @param[in] mvals The parameter vector
    @return DC0 The norm of the vector
    @return DC1 The gradient of DC0, a constant
    @return DC2 The Hessian (zero)

    """
    np = len(mvals)
    DC0 = linalg.norm(mvals)
    DC1 = ones(len(mvals))
    DC2 = zeros((np, np), dtype=float)
    return DCO, DC1, DC2

Pen_Tab = {'L1' : L1_norm,
           'L2' : L2_norm
           }

def Penalty(mvals, Objective, Penalty_Type, Factor_Add=0.0, Factor_Mult=0.0):
    """
    This function is called by Project.Objective to actually impose the penalty
    on the objective function.  It allows us to switch between different penalty types
    (see Pen_Tab, I think we only have two options) and to choose either
    an additive penalty (adds to the objective function), a multiplicative penalty
    (multiplies the objective function by 1+P), or both.

    How to use: I can only imagine calling this from Project.Objective at the moment.

    Inputs:
    mvals        = The parameter vector
    Objective    = A dictionary of the objective function and its derivatives evaluated at the parameters
    Penalty_Type = A penalty functional form that comes out of Pen_Tab
    Factor_Add   = Prefactor for the strength of an additive penalty
    Factor_Mult  = Prefactor for the strength of a multiplicative penalty

    Outputs:
    XAdd         = The additional penalty
    GAdd         = The penalty's contribution to the gradient
    HAdd         = The penalty's contribution to the Hessian
    """
    X = Objective['X']
    G = Objective['G']
    H = Objective['H']
    np = len(mvals)
    K0, K1, K2 = Pen_Tab[Penalty_Type](mvals)
    XAdd = 0.0
    GAdd = zeros(np, dtype=float)
    HAdd = zeros((np, np), dtype=float)
    if Factor_Add > 0.0:
        XAdd += K0 * Factor_Add
        GAdd += K1 * Factor_Add
        HAdd += K2 * Factor_Add
    if Factor_Mult > 0.0:
        XAdd += ( X*K0 ) * Factor_Mult
        GAdd += array( G*K0 + X*K1 ) * Factor_Mult
        GK1 = reshape(G, (1, -1))*reshape(K1, (-1, 1))
        K1G = reshape(K1, (1, -1))*reshape(G, (-1, 1))
        HAdd += array( H*K0+GK1+K1G+X*K2 ) * Factor_Mult
    return XAdd, GAdd, HAdd
