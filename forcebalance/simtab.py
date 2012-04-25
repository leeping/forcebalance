""" @package simtab Contains the dictionary of fitting simulation classes.

This is in a separate file to facilitate importing.  I would happily put it somewhere else.

"""

from gmxio import ForceEnergyMatch_GMX
from tinkerio import ForceEnergyMatch_TINKER
from openmmio import ForceEnergyMatch_OpenMM, PropertyMatch_OpenMM
from forceenergymatch_gmxx2 import ForceEnergyMatch_GMXX2
from forceenergymatch_internal import ForceEnergyMatch_Internal
from counterpoisematch import CounterpoiseMatch

## The table of fitting simulations
SimTab = {
    'FORCEENERGYMATCH_GMX':ForceEnergyMatch_GMX,
    'FORCEENERGYMATCH_TINKER':ForceEnergyMatch_TINKER,
    'FORCEENERGYMATCH_OPENMM':ForceEnergyMatch_OpenMM,
    'FORCEENERGYMATCH_GMXX2':ForceEnergyMatch_GMXX2,
    'FORCEENERGYMATCH_INTERNAL':ForceEnergyMatch_Internal,
    'PROPERTYMATCH_OPENMM':PropertyMatch_OpenMM,
    'COUNTERPOISEMATCH':CounterpoiseMatch
    }
