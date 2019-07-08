## HACK: Improve the performance of the openff forcefield.create_openmm_system()

# time based on total 540s evaluation
# cache for find_smarts_matches (save 300+ s)
from openforcefield.utils.toolkits import OpenEyeToolkitWrapper, RDKitToolkitWrapper
original_find_smarts_matches = OpenEyeToolkitWrapper.find_smarts_matches
TOOLKIT_CACHE_find_smarts_matches = {}
def cached_find_smarts_matches(self, molecule, smarts, aromaticity_model='OEAroModel_MDL'):
    cache_key = hash((molecule, smarts, aromaticity_model))
    if cache_key not in TOOLKIT_CACHE_find_smarts_matches:
        TOOLKIT_CACHE_find_smarts_matches[cache_key] = original_find_smarts_matches(self, molecule, smarts, aromaticity_model=aromaticity_model)
    return TOOLKIT_CACHE_find_smarts_matches[cache_key]
# replace the original function with new one
OpenEyeToolkitWrapper.find_smarts_matches = cached_find_smarts_matches


# cache for the validate function (save 94s)
from openforcefield.typing.chemistry.environment import ChemicalEnvironment
original_validate = ChemicalEnvironment.validate
TOOLKIT_CACHE_ChemicalEnvironment_validate = {}
def cached_validate(smirks, ensure_valence_type=None, toolkit='openeye'):
    cache_key = hash((smirks, ensure_valence_type, toolkit))
    if cache_key not in TOOLKIT_CACHE_ChemicalEnvironment_validate:
        TOOLKIT_CACHE_ChemicalEnvironment_validate[cache_key] = original_validate(smirks, ensure_valence_type=None, toolkit=toolkit)
    return TOOLKIT_CACHE_ChemicalEnvironment_validate[cache_key]
ChemicalEnvironment.validate = cached_validate


# cache for compute_partial_charges_am1bcc (save 69s)
original_compute_partial_charges_am1bcc = OpenEyeToolkitWrapper.compute_partial_charges_am1bcc
TOOLKIT_CACHE_compute_partial_charges_am1bcc = {}
def cached_compute_partial_charges_am1bcc(self, molecule):
    cache_key = hash(molecule)
    if cache_key not in TOOLKIT_CACHE_compute_partial_charges_am1bcc:
        TOOLKIT_CACHE_compute_partial_charges_am1bcc[cache_key] = original_compute_partial_charges_am1bcc(self, molecule)
    return TOOLKIT_CACHE_compute_partial_charges_am1bcc[cache_key]
OpenEyeToolkitWrapper.compute_partial_charges_am1bcc = cached_compute_partial_charges_am1bcc

# cache the generate_conformers function (save 15s)
TOOLKIT_CACHE_molecule_conformers = {}
original_generate_conformers = OpenEyeToolkitWrapper.generate_conformers
def cached_generate_conformers(self, molecule, n_conformers=1, clear_existing=True):
    cache_key = hash((molecule, n_conformers, clear_existing))
    if cache_key not in TOOLKIT_CACHE_molecule_conformers:
        original_generate_conformers(self, molecule, n_conformers=n_conformers, clear_existing=clear_existing)
        TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
    molecule._conformers = TOOLKIT_CACHE_molecule_conformers[cache_key]
OpenEyeToolkitWrapper.generate_conformers = cached_generate_conformers

# final timing: 56s

# cache the ForceField creation (no longer needed since using OpenFF API for parameter modifications)

import hashlib
from openforcefield.typing.engines.smirnoff import ForceField
SMIRNOFF_FORCE_FIELD_CACHE = {}
def getForceField(*ffpaths):
    hasher = hashlib.md5()
    for path in ffpaths:
        with open(path, 'rb') as f:
            hasher.update(f.read())
    cache_key = hasher.hexdigest()
    if cache_key not in SMIRNOFF_FORCE_FIELD_CACHE:
        SMIRNOFF_FORCE_FIELD_CACHE[cache_key] = ForceField(*ffpaths, allow_cosmetic_attributes=True)
    return SMIRNOFF_FORCE_FIELD_CACHE[cache_key]
