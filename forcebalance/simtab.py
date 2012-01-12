""" @package simtab Contains the dictionary of fitting simulation classes.

This is in a separate file to facilitate importing.  I would happily put it somewhere else.

"""

from gmxio import ForceEnergyMatch_GMX
from tinkerio import ForceEnergyMatch_TINKER
from forceenergymatch_gmxx2 import ForceEnergyMatch_GMXX2
from counterpoisematch import CounterpoiseMatch

## The table of fitting simulations
SimTab = {
    'FORCEENERGYMATCH_GMX':ForceEnergyMatch_GMX,
    'FORCEENERGYMATCH_TINKER':ForceEnergyMatch_TINKER,
    'FORCEENERGYMATCH_GMXX2':ForceEnergyMatch_GMXX2,
    'COUNTERPOISEMATCH':CounterpoiseMatch
    }
