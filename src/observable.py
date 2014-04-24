import os
import numpy as np

from forcebalance.finite_difference import fdwrap, f12d3p
from forcebalance.molecule import Molecule
from forcebalance.nifty import col, flat, getval
from forcebalance.nifty import printcool, statisticalInefficiency
from forcebalance.optimizer import Counter

from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

# method mean_stderr
def mean_stderr(ts):
    """Return mean and standard deviation of a time series ts."""
    return np.mean(ts), np.std(ts)*np.sqrt(statisticalInefficiency(ts, warn=False)/len(ts))

# method energy_derivatives
def energy_derivatives(engine, FF, mvals, h, pgrad, length, AGrad=True):
    """Compute the first derivatives of a set of snapshot energies with respect
    to the force field parameters. The function calls the finite
    difference subroutine on the energy_driver subroutine also in this
    script.

    Parameters
    ----------
    engine : Engine
        Use this Engine (`GMX`,`TINKER`,`OPENMM` etc.) object to get the energy
        snapshots.
    FF : FF
       Force field object.
    mvals : list
        Mathematical parameter values.
    h : float
        Finite difference step size.
    length : int
        Number of snapshots (length of energy trajectory).
    AGrad : Boolean
        Switch that turns derivatives on or off; if off, return all zeros.

    Returns
    -------
    G : np.array
        Derivative of the energy in a FF.np x length array.
    
    """
    G = np.zeros((FF.np, length))
    
    if not AGrad:
        return G
    def energy_driver(mvals_):
        FF.make(mvals_)
        return engine.energy()

    ED0 = energy_driver(mvals)
        
    for i in pgrad:
        logger.info("%i %s\r" % (i, (FF.plist[i] + " "*30)))
        EDG, _   = f12d3p(fdwrap(energy_driver, mvals, i), h, f0=ED0)

        G[i,:]   = EDG[:]
    return G

class Observable(object):
    """
    Base class for thermodynamical observable used for fitting. This can
    be any experimental data that can be calculated as an ensemble
    average from a simulation.

    Data attributes
    ---------------
    name : string
        Identifier for the observable that is specified in `observables` in Target
        options.
    """
    def __init__(self, source):
        # Reference data which can be useful in calculating the observable.
        if 'temp' in source: self.temp = getval(source, 'temp')
        if 'pres' in source: self.pres = getval(source, 'pres')
        self.Data = source[self.columns]
        
    def __str__(self):
        return "Observable = " + self.name.capitalize() + "; Columns = " + ', '.join(self.columns)

    def extract(self, engines, FF, mvals, h, AGrad=True):
        """Calculate and extract the observable from MD results. How this is done
        depends on the observable and the engine so this must be
        implemented in the subclass.

        Parameters
        ----------
        engines : list
            A list of Engine objects that are requred to calculate the observable.
        FF : FF
            Force field object.
        mvals : list
            Mathematical parameter values.
        h : float
            Finite difference step size.
        AGrad : Boolean
            Switch that turns derivatives on or off; if off, return all zeros.
        
        Returns
        -------
        result : (float, float, np.array)
            The returned tuple is (Q, Qerr, Qgrad), where Q is the calculated
            observable, Qerr is the calculated standard deviation of the observable,
            and Qgrad is a M-array with the calculated gradients for the
            observable, with M being the number of force field parameters that are
            being fitted. 
        
        """
        logger.error("Extract method not implemented in base class.\n")    
        raise NotImplementedError

    def aggregate(self, Sims, AGrad, cycle=None):
        print self.name
        if cycle == None: cycle = Counter()
        # Different from the Results objects in the Simulation, this
        # one is keyed by the simulation type then by the time series
        # data type.
        self.TimeSeries = OrderedDict([(i, OrderedDict()) for i, j in self.requires.items()])
        for stype in self.requires:
            for dtype in self.requires[stype]:
                self.TimeSeries[stype][dtype] = np.concatenate([Sim.Results[cycle][dtype] for Sim in Sims if Sim.type == stype])
        if AGrad:
            # Also aggregate the derivative information along the second axis (snapshot axis)
            self.Derivatives = OrderedDict()
            for stype in self.requires:
                # The derivatives that we have may be obtained from the 'derivatives' data structure of the first Simulation
                # that matches the required simulation type.
                self.Derivatives[stype] = OrderedDict()
                for dtype in [Sim.Results[cycle]['derivatives'].keys() for Sim in Sims if Sim.type == stype][0]:
                    self.Derivatives[stype][dtype] = np.concatenate([Sim.Results[cycle]['derivatives'][dtype] for Sim in Sims if Sim.type == stype], axis=1)

# class Observable_Density
class Observable_Density(Observable):

    """ 
    The Observable_Density class implements common methods for
    extracting the density from a simulation, but does not specify the
    simulation itself ('requires' attribute).  Don't create a
    Density object directly, use the Liquid_Density and Solid_Density
    derived classes.

    This is due to our overall framework that each observable must
    have a unique list of required simulations, yet the formula for
    calculating the density and its derivative is always the same.
    """

    def __init__(self, source):
        # Name of the observable.
        self.name = 'density'
        # Columns that are taken from the data table.
        self.columns = ['density']
        super(Observable_Density, self).__init__(source)

    def evaluate(self, AGrad):         
        #==========================================#
        #  Physical constants and local variables. #
        #==========================================#
        # Energies in kJ/mol and lengths in nanometers.
        kB    = 0.008314472471220214
        kT    = kB*self.temp
        Beta  = 1.0/kT
        mBeta = -Beta
        phase = self.requires.keys()[0]
        # Density time series.
        Density  = self.TimeSeries[phase]['density']
        # Average and error.
        Rho_avg, Rho_err = mean_stderr(Density)
        Answer = OrderedDict()
        Answer['mean'] = Rho_avg
        Answer['stderr'] = Rho_err
        if AGrad:
            G = self.Derivatives[phase]['potential']
            # Analytic first derivative.
            Rho_grad = mBeta * (flat(np.matrix(G) * col(Density)) / len(Density)
                                - np.mean(Density) * np.mean(G, axis=1))
            Answer['grad'] = Rho_grad
        return Answer

class Liquid_Density(Observable_Density):
    def __init__(self, source):
        # The density time series is required from the simulation.
        self.requires = OrderedDict([('liquid', ['density'])])
        super(Liquid_Density, self).__init__(source)

class Solid_Density(Observable_Density):
    def __init__(self, source):
        # The density time series is required from the simulation.
        self.requires = OrderedDict([('solid', ['density'])])
        super(Solid_Density, self).__init__(source)

# class Observable_H_vap
class Observable_H_vap(Observable):
    def __init__(self, source):
        """ Enthalpy of vaporization. """
        # Name of the observable.
        self.name = 'hvap'
        # Columns that are taken from the data table.
        self.columns = ['hvap']
        # Get energy/volume from liquid simulation, and energy from gas simulation.
        self.requires = OrderedDict([('liquid', ['energy', 'volume']), ('gas', ['energy'])])
        # Initialize the base class
        super(Observable_H_vap, self).__init__(source)

    def extract(self, engines, FF, mvals, h, pgrad, AGrad=True): 
        #==========================================#
        #  Physical constants and local variables. #
        #==========================================#
        # Energies in kJ/mol and lengths in nanometers.
        kB      = 0.008314472471220214
        kT      = kB*self.temperature
        Beta    = 1.0/kT
        mBeta   = -Beta
        # Conversion factor between 1 kJ/mol -> bar nm^3 
        pconv   = 16.6054

        # Number of molecules in the liquid phase.
        mol     = Molecule(os.path.basename(os.path.splitext(engines[0].mdtraj)[0]) +
                           ".gro")
        nmol = len(mol.molecules)

        #======================================================#
        #  Get simulation properties depending on the engines. #
        #======================================================#
        if self.engname == "gromacs":
            # Default names
            deffnm1 = os.path.basename(os.path.splitext(engines[0].mdene)[0])
            deffnm2 = os.path.basename(os.path.splitext(engines[1].mdene)[0])
            # Figure out which energy terms and present and their order. 
            energyterms1 = engines[0].energy_termnames(edrfile="%s.%s" % (deffnm1, "edr"))
            energyterms2 = engines[1].energy_termnames(edrfile="%s.%s" % (deffnm2, "edr"))
            # Grab energy terms to print and keep track of energy term order.
            ekeep1  = ['Total-Energy', 'Potential', 'Kinetic-En.', 'Temperature', 'Volume']
            ekeep2  = ['Total-Energy', 'Potential', 'Kinetic-En.', 'Temperature']

            ekeep_order1 = [key for (key, value)
                            in sorted(energyterms1.items(), key=lambda (k, v) : v)
                            if key in ekeep1]
            ekeep_order2 = [key for (key, value)
                            in sorted(energyterms2.items(), key=lambda (k, v) : v)
                            if key in ekeep2]

            # Perform energy component analysis and return properties.
            engines[0].callgmx(("g_energy " +
                                "-f %s.%s " % (deffnm1, "edr") +
                                "-o %s-energy.xvg " % deffnm1 +
                                "-xvg no"),
                                stdin="\n".join(ekeep1))
            engines[1].callgmx(("g_energy " +
                                "-f %s.%s " % (deffnm2, "edr") +
                                "-o %s-energy.xvg " % deffnm2 +
                                "-xvg no"),
                                stdin="\n".join(ekeep2))

        # Read data and store properties by grabbing columns in right order.
        data1       = np.loadtxt("%s-energy.xvg" % deffnm1)
        data2       = np.loadtxt("%s-energy.xvg" % deffnm2)
        Energy      = data1[:, ekeep_order1.index("Total-Energy") + 1]
        Potential   = data1[:, ekeep_order1.index("Potential") + 1]
        Kinetic     = data1[:, ekeep_order1.index("Kinetic-En.") + 1]
        Temperature = data1[:, ekeep_order1.index("Temperature") + 1]
        Volume      = data1[:, ekeep_order1.index("Volume") + 1]
        mEnergy     = data2[:, ekeep_order2.index("Total-Energy") + 1]
        mPotential  = data2[:, ekeep_order2.index("Potential") + 1]
        mKinetic    = data2[:, ekeep_order2.index("Kinetic-En.") + 1]
        
        #============================================#
        #  Compute the potential energy derivatives. #
        #============================================#
        logger.info(("Calculating potential energy derivatives " +
                     "with finite difference step size: %f\n" % h))
        printcool("Initializing arrays to lengths %d" % len(Energy),
                  color=4, bold=True)
        
        G  = energy_derivatives(engines[0], FF, mvals, h, pgrad, len(Energy), AGrad)
        Gm = energy_derivatives(engines[1], FF, mvals, h, pgrad, len(mEnergy), AGrad)
                
        #=========================================#
        #  Observable properties and derivatives. #
        #=========================================#
        # Average and error.
        E_avg, E_err     = mean_stderr(Energy)
        Em_avg, Em_err   = mean_stderr(mEnergy)
        Vol_avg, Vol_err = mean_stderr(Volume)
                
        Hvap_avg = Em_avg - E_avg/nmol - self.pressure*Vol_avg/nmol/pconv + kT 
        Hvap_err = np.sqrt((E_err/nmol)**2 + Em_err**2
                           + (self.pressure**2) * (Vol_err**2)/(float(nmol)**2)/(pconv**2))
        # Analytic first derivative.
        Hvap_grad  = np.mean(Gm, axis=1)
        Hvap_grad += mBeta * (flat(np.mat(Gm) * col(mEnergy)) / len(mEnergy) \
                               - np.mean(mEnergy) * np.mean(Gm, axis=1))
        Hvap_grad -= np.mean(G, axis=1)/nmol
        Hvap_grad += Beta * (flat(np.mat(G) * col(Energy)) / len(Energy) \
                               - np.mean(Energy) * np.mean(G, axis=1))/nmol
        Hvap_grad += (Beta*self.pressure/nmol/pconv) * \
          (flat(np.mat(G) * col(Volume)) / len(Volume) \
           - np.mean(Volume) * np.mean(G, axis=1))

        return Hvap_avg, Hvap_err, Hvap_grad

# class Observable_Al
class Observable_Al(Observable):
    def __init__(self, source):
        """ Area per lipid. """
        # Name of the observable.
        self.name = 'al'
        # Columns that are taken from the data table.
        self.columns = ['al']
        # Get area per lipid from the bilayer simulation.
        self.requires = OrderedDict([('bilayer', ['al'])])
        # Initialize the base class
        super(Observable_Al, self).__init__(source)

# class Observable_Scd
class Observable_Scd(Observable):
    def __init__(self, source):
        """ Deuterium order parameter. """
        # Name of the observable.
        self.name = 'scd'
        # Columns that are taken from the data table.
        self.columns = ['scd1_idx', 'scd1', 'scd2_idx', 'scd2']
        # Get deuterium order parameter from the bilayer simulation.
        self.requires = OrderedDict([('bilayer', ['scd1', 'scd2'])])
        # Initialize the base class
        super(Observable_Scd, self).__init__(source)

# class Lipid_Kappa
class Lipid_Kappa(Observable):
    def __init__(self, source):
        """ Compressibility as calculated for lipid bilayers. """
        # Name of the observable.
        self.name = 'kappa'
        # Columns that are taken from the data table.
        self.columns = ['kappa']
        # Get area per lipid from the bilayer simulation.
        self.requires = OrderedDict([('bilayer', ['al'])])
        # Initialize the base class
        super(Lipid_Kappa, self).__init__(source)

# class Liquid_Kappa
class Liquid_Kappa(Observable):
    def __init__(self, source):
        """ Compressibility as calculated for liquids. """
        # Name of the observable.
        self.name = 'kappa'
        # Columns that are taken from the data table.
        self.columns = ['kappa']
        # Get area per lipid from the bilayer simulation.
        self.requires = OrderedDict([('liquid', ['volume'])])
        # Initialize the base class
        super(Liquid_Kappa, self).__init__(source)

## A mapping that takes us from observable names to possible Observable objects.
OMap = {'density' : [Liquid_Density, Solid_Density],
        'rho' : [Liquid_Density, Solid_Density],
        'hvap' : [Observable_H_vap],
        'h_vap' : [Observable_H_vap],
        'al' : [Observable_Al],
        'kappa' : [Liquid_Kappa, Lipid_Kappa],
        'scd' : [Observable_Scd]}
