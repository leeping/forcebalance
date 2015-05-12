""" Finite difference module. """

import traceback
from numpy import dot
from forcebalance.output import getLogger
logger = getLogger(__name__)

def f1d2p(f, h, f0 = None):
    """
    A two-point finite difference stencil.
    This function does either two computations or one,
    depending on whether the 'center' value is supplied.
    This is done in order to avoid recomputing the center
    value many times when we repeat this function for each
    index of the gradient.

    How to use: use fdwrap or something similar to generate
    a one-variable function from the (usually) much more complicated
    function that we wish to differentate.  Then pass it to this function.

    Inputs:
    f  = The one-variable function f(x) that we're differentiating
    h  = The finite difference step size, usually a small number

    Outputs:
    fp = The finite difference derivative of the function f(x) around x=0.
    """
    if f0 is None:
        f0, f1 = [f(i*h) for i in [0, 1]]
    else:
        f1 = f(h)
    fp = (f1-f0)/h
    return fp

def f1d5p(f, h):
    """
    A highly accurate five-point finite difference stencil
    for computing derivatives of a function.  It works on both
    scalar and vector functions (i.e. functions that return arrays).
    Since the function does four computations, it's costly but
    recommended if we really need an accurate reference value.

    The function is evaluated at points -2h, -h, +h and +2h
    and these values are combined to make the derivative according to:
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/

    How to use: use fdwrap or something similar to generate
    a one-variable function from the (usually) much more complicated
    function that we wish to differentate.  Then pass it to this function.

    Inputs:
    f  = The one-variable function f(x) that we're differentiating
    h  = The finite difference step size, usually a small number

    Outputs:
    fp = The finite difference derivative of the function f(x) around x=0.
    """
    fm2, fm1, f1, f2 = [f(i*h) for i in [-2, -1, 1, 2]]
    fp = (-1*f2+8*f1-8*fm1+1*fm2)/(12*h)
    return fp

def f1d7p(f, h):
    """
    A highly accurate seven-point finite difference stencil
    for computing derivatives of a function.  
    """
    fm3, fm2, fm1, f1, f2, f3 = [f(i*h) for i in [-3, -2, -1, 1, 2, 3]]
    fp = (f3-9*f2+45*f1-45*fm1+9*fm2-fm3)/(60*h)
    return fp

def f12d7p(f, h):
    fm3, fm2, fm1, f0, f1, f2, f3 = [f(i*h) for i in [-3, -2, -1, 0, 1, 2, 3]]
    fp = (f3-9*f2+45*f1-45*fm1+9*fm2-fm3)/(60*h)
    fpp = (2*f3-27*f2+270*f1-490*f0+270*fm1-27*fm2+2*fm3)/(180*h*h)
    return fp, fpp

def f12d3p(f, h, f0 = None):
    """
    A three-point finite difference stencil.
    This function does either two computations or three,
    depending on whether the 'center' value is supplied.
    This is done in order to avoid recomputing the center
    value many times.

    The first derivative is evaluated using central difference.
    One advantage of using central difference (as opposed to forward
    difference) is that we get zero at the bottom of a parabola.

    Using this formula we also get an approximate second derivative, which
    can then be inserted into the diagonal of the Hessian.  This is very
    useful for optimizations like BFGS where the diagonal determines
    how far we step in the parameter space.

    How to use: use fdwrap or something similar to generate
    a one-variable function from the (usually) much more complicated
    function that we wish to differentate.  Then pass it to this function.

    Inputs:
    f  = The one-variable function f(x) that we're differentiating
    h  = The finite difference step size, usually a small number

    Outputs:
    fp = The finite difference derivative of the function f(x) around x=0.
    """
    if f0 is None:
        fm1, f0, f1 = [f(i*h) for i in [-1, 0, 1]]
    else:
        fm1, f1 = [f(i*h) for i in [-1, 1]]
    fp = (f1-fm1)/(2*h)
    fpp = (fm1-2*f0+f1)/(h*h)
    return fp, fpp

def f2var(f, h):
    """ A finite difference stencil for a function of two variables. """
    fpp, fpm, fmp, fmm = [f(i*h, j*h) for i, j in [(1,1), (1,-1), (-1,1), (-1,-1)]]
    return (fpp-fpm-fmp+fmm)/(4*h**2)

def in_fd():
    """ Invoking this function from anywhere will tell us whether we're being called by a finite-difference function.
    This is mainly useful for deciding when to update the 'qualitative indicators' and when not to. """

    return any([i in [j[2] for j in traceback.extract_stack()] for i in ['f1d2p','f12d3p','f1d5p','f12d7p','f1d7p']])

def in_fd_srch():
    """ Invoking this function from anywhere will tell us whether we're being called by a finite-difference function.
    This is mainly useful for deciding when to update the 'qualitative indicators' and when not to. """

    return any([i in [j[2] for j in traceback.extract_stack()] for i in ['f1d2p','f12d3p','f1d5p','f12d7p','f1d7p','search_fun']])

def fdwrap(func,mvals0,pidx,key=None,**kwargs):
    """
    A function wrapper for finite difference designed for
    differentiating 'get'-type functions.

    Since our finite difference stencils take single-variable functions
    and differentiate them around zero, and our objective function is
    quite a complicated function, we need a wrapper to serve as a
    middleman.  The alternative would be to copy the finite difference
    formula to wherever we're taking the derivative, and that is prone
    to mistakes.

    Inputs:
    func   = Either get_X or get_G; these functions return dictionaries. ['X'] = 1.23, ['G'] = [0.12, 3,45, ...]
    mvals0 = The 'central' values of the mathematical parameters - i.e. the wrapped function's origin is here.
    pidx   = The index of the parameter that we're differentiating
    key    = either 'G' or 'X', the value we wish to take out of the dictionary
    kwargs = Anything else we want to pass to the objective function (for instance, Project.Objective takes Order as an argument)

    Outputs:
    func1  = Wrapped version of func, which takes a single float argument.
    """
    def func1(arg):
        mvals = list(mvals0)
        mvals[pidx] += arg
        logger.info("\rfdwrap: " + func.__name__ + " [%i] = % .1e " % (pidx, arg) + ' '*50 + '\r')
        if key is not None:
            return func(mvals,**kwargs)[key]
        else:
            return func(mvals,**kwargs)
    return func1
        
def fdwrap_G(tgt,mvals0,pidx):
    """
    A driver to fdwrap for gradients (see documentation for fdwrap)
    Inputs:
    tgt    = The Target containing the objective function that we want to differentiate
    mvals0 = The 'central' values of the mathematical parameters - i.e. the wrapped function's origin is here.
    pidx   = The index of the parameter that we're differentiating
    """
    return fdwrap(tgt.get_X,mvals0,pidx,'X')

def fdwrap_H(tgt,mvals0,pidx):
    """
    A driver to fdwrap for Hessians (see documentation for fdwrap)
    Inputs:
    tgt    = The Target containing the objective function that we want to differentiate
    mvals0 = The 'central' values of the mathematical parameters - i.e. the wrapped function's origin is here.
    pidx   = The index of the parameter that we're differentiating
    """
    return fdwrap(tgt.get_G,mvals0,pidx,'G')

#method resolution order
#type.mro(type(a))
