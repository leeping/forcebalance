""" @package simtab Contains the dictionary of fitting simulation classes.

This is in a separate file to facilitate importing.  I would happily put it somewhere else.

"""

from gmxio import AbInitio_GMX, Interaction_GMX
from tinkerio import AbInitio_TINKER
from openmmio import AbInitio_OpenMM, Liquid_OpenMM
from abinitio_gmxx2 import AbInitio_GMXX2
from abinitio_internal import AbInitio_Internal
from counterpoise import Counterpoise
from amberio import AbInitio_AMBER
from psi4io import THCDF_Psi4

## The table of fitting simulations
SimTab = {
    'ABINITIO_GMX':AbInitio_GMX,
    'INTERACTION_GMX':Interaction_GMX,
    'ABINITIO_TINKER':AbInitio_TINKER,
    'ABINITIO_OPENMM':AbInitio_OpenMM,
    'ABINITIO_AMBER':AbInitio_AMBER,
    'ABINITIO_GMXX2':AbInitio_GMXX2,
    'ABINITIO_INTERNAL':AbInitio_Internal,
    'LIQUID_OPENMM':Liquid_OpenMM,
    'COUNTERPOISE':Counterpoise,
    'THCDF_PSI4':THCDF_Psi4,
    }
