"""@package project

ForceBalance force field optimization project."""

from optimizer import Optimizer
from forcefield import FF
from simtab import SimTab
from parser import parse_inputs
from penalty import Penalty
from nifty import printcool_dictionary
from numpy import zeros

class Project(object):
    """Container for a ForceBalance force field optimization project.

    The triumvirate or trinity of components are:
    - The force field
    - The objective function
    - The optimizer

    The force field is a class defined in forcefield.py.
    The objective function is built here as a combination of fitting simulation classes.
    The optimizer is a class defined in this file.
    """
    
    def __init__(self,input_file):
        """Instantiation of a ForceBalance force field optimization project.

        Here's what we do:

        - Parse the input file
        - Create an instance of the force field
        - Create a list of fitting simulation instances
        - Create an optimizer instance
        - Print out the general options
        
        """
        #======================================#
        #    Call these functions at startup   #
        #======================================#
        ## The general options and simulation options that come from parsing the input file
        self.options, self.sim_opts = parse_inputs(input_file)
        ## The force field component of the project
        self.FF = FF(self.options)
        ## The list of fitting simulations
        self.Simulations = [SimTab[opts['simtype']](self.options,opts,self.FF) for opts in self.sim_opts]
        ## The optimizer component of the project
        self.Optimizer   = Optimizer(self.options,self.Objective,self.FF,self.Simulations)
        printcool_dictionary(self.options, title="Setup for optimizer")
        
    def Objective(self,mvals,Order=0,usepvals=False):
        """ Objective function defined within Project; can you think of a better place?

        The objective function is a combination of contributions from the different
        fitting simulations.  Basically, it loops through the fitting simulations,
        gets their contributions to the objective function and then sums all of them
        (although more elaborate schemes are conceivable).  The return value is the
        same data type as calling the fitting simulation itself: a dictionary containing
        the objective function, the gradient and the Hessian.

        The penalty function is also computed here; it keeps the parameters from straying
        too far from their initial values.

        @param[in] mvals The mathematical parameters that enter into computing the objective function
        @param[in] Order The requested order of differentiation
        @param[in] usepvals Switch that determines whether to use physical parameter values
        """
        ## This is the objective function; it's a dictionary containing the value, first and second derivatives
        Objective = {'X':0.0, 'G':zeros(self.FF.np), 'H':zeros((self.FF.np,self.FF.np))}
        ## This is the canonical lettering that corresponds to : objective function, gradient, Hessian.
        Letters = ['X','G','H']
        # Loop through the simulations.
        for Sim in self.Simulations:
            # List of functions that I can call.
            Funcs   = [Sim.get_X, Sim.get_G, Sim.get_H]
            # Call the appropriate function
            Ans = Funcs[Order](mvals)
            # Note that no matter which order of function we call, we still increment the objective / gradient / Hessian the same way.
            for i in range(3):
                Objective[Letters[i]] += Ans[Letters[i]]
        ## Compute the penalty function.
        Extra = Penalty(mvals,Objective,self.options['penalty_type'],self.options['penalty_additive'],self.options['penalty_multiplicative'])
        for i in range(3):
            Objective[Letters[i]] += Extra[i]
        return Objective

    def Run(self):
        """ Call the appropriate optimizer.  This is the method we might want to call from an executable. """
        self.Optimizer.OptTab[self.options['jobtype']]()

