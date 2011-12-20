""" @package simtab Contains the dictionary of fitting simulation classes.

This is in a separate file to facilitate importing.  I would happily put it somewhere else.

"""

from forceenergymatch_gmx import ForceEnergyMatch_GMX
from counterpoisematch import CounterpoiseMatch

## The table of fitting simulations
SimTab = {
    'FORCEENERGYMATCH_GMX':ForceEnergyMatch_GMX,
    'COUNTERPOISEMATCH':CounterpoiseMatch
    }
