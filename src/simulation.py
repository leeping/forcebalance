import os
from forcebalance.molecule import Molecule
from collections import OrderedDict

class Simulation(object):

    """ 
    Data container for a MD simulation (specified by index, simulation
    type, initial condition).  These settings are written to a file
    then passed to md_one.py.

    The Simulation object is passed between the master ForceBalance
    process and the remote script (e.g. md_one.py).
    """

    type_settings = {'gas': {'pbc' : 0},
                     'liquid': {'pbc' : 1},
                     'solid': {'pbc' : 1, 'anisotropic_box' : 1},
                     'bilayer': {'pbc' : 1, 'anisotropic_box' : 1}}

    def __init__(self, target, name, index, stype, initial, iframe, tsnames):
        print target.root, target.tgtdir
        raw_input()
        # The simulation name will identify the simulation within a collection
        # belonging to the Index.
        self.name = name
        # The Index that the simulation belongs to.
        self.index = index
        # The type of simulation (liquid, gas, solid, bilayer...)
        if stype not in Simulation.type_settings.keys():
            logger.error('Simulation type %s is not supported at this time')
            raise RuntimeError
        self.type = stype
        # The file containing initial coordinates.
        self.initial = initial
        # The frame number in the initial coordinate file.
        self.iframe = iframe
        # The time series for the simulation.
        self.timeseries = OrderedDict([(i, []) for i in tsnames])
        # The file extension that the coordinate file will be written with.
        self.fext = os.path.splitext(initial)[1]
        # The file name of the coordinate file.
        self.coords = "%s%s" % (self.type, self.fext)
        # The number of threads for this simulation.
        self.threads = target.OptionDict.get('md_threads', 1)
        # Whether to use multiple timestep integrator.
        self.mts = target.OptionDict.get('mts_integrator', 0)
        # The number of beads in an RPMD simulation.
        self.rpmd_beads = target.OptionDict.get('rpmd_beads', 0)
        # Whether to use the CUDA platform (OpenMM only).
        self.force_cuda = target.OptionDict.get('force_cuda', 0)
        # Number of MD steps between successive calls to Monte Carlo barostat (OpenMM only).
        self.nbarostat = target.OptionDict.get('n_mcbarostat', 25)
        # Flag for anisotropic simulation cell.
        self.anisotropic = target.OptionDict.get('anisotropic_box', 0)
        # Flag for minimizing the energy.
        self.minimize = target.OptionDict.get('minimize_energy', 0)
        # Finite difference step size.
        self.h = target.h
        # Name of the simulation engine.
        self.engname = target.engname
        # Whether to use periodic boundary conditions.
        self.pbc = Simulation.type_settings[self.type]['pbc']
        # Gromacs-specific options.
        if self.engname == 'gromacs':
            self.gmxpath = target.gmxpath
            self.gmxsuffix = target.gmxsuffix
        elif self.engname == 'tinker':
            self.tinkerpath = target.tinkerpath

    def __str__(self):
        msg = []
        msg.append("Simulation: Name %s, Index %s, Type %s" % (self.name, self.index, self.type))
        msg.append("Initial Conditions: File %s Frame %i" % (self.initial, self.iframe))
        msg.append("Timeseries Names: %s" % (', '.join(self.timeseries.keys())))
        return "\n".join(msg)
