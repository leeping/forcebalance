## HACK: Improve the performance of the openff forcefield.create_openmm_system()
import os
from openff.toolkit.utils.toolkits import OpenEyeToolkitWrapper, RDKitToolkitWrapper, AmberToolsToolkitWrapper
from openff.toolkit.topology.molecule import Molecule


# Add a mechanism for disabling SMIRNOFF hack entirely as it is prone to breaking
# when upstream dependencies (especially the toolkit) are updated.
_SHOULD_CACHE = os.environ.get("ENABLE_FB_SMIRNOFF_CACHING")

# Caching of SMIRNOFF functions is enabled by default, including when the
# "ENABLE_FB_SMIRNOFF_CACHING" environmental variable is not set. The user
# Can `export ENABLE_FB_SMIRNOFF_CACHING=False` to disable all caching.
if _SHOULD_CACHE is None:
    _SHOULD_CACHE = True
else:
    _SHOULD_CACHE = _SHOULD_CACHE.lower() in ["true", "1", "yes"]


def hash_molecule(molecule):

    atom_map = None

    if "atom_map" in molecule.properties:
        # Store a copy of any existing atom map
        atom_map = molecule.properties.pop("atom_map")

    cmiles = molecule.to_smiles(mapped=True)

    if atom_map is not None:
        molecule.properties["atom_map"] = atom_map

    return cmiles


def hash_molecule_args_and_kwargs(molecule, *args, **kwargs):
    arguments = [str(arg) for arg in args]
    keywords = [str(keyword) for keyword in kwargs.values()]
    arguments_plus_keywords = arguments + keywords
    # Deleted * in front of arguments_plus_keywords for Python 2.7 compatibility, convert list to tuple.
    return hash((hash_molecule(molecule), tuple(arguments_plus_keywords)))


if _SHOULD_CACHE:
    # Commented out because it is printed even for non-SMIRNOFF calculations.
    # print(
    #     "SMIRNOFF functions will be replaced with cached versions to improve their "
    #     "performance."
    # )

    # time based on total 540s evaluation
    # cache for OE find_smarts_matches (save 300+ s)
    oe_original_find_smarts_matches = OpenEyeToolkitWrapper.find_smarts_matches
    OE_TOOLKIT_CACHE_find_smarts_matches = {}
    def oe_cached_find_smarts_matches(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in OE_TOOLKIT_CACHE_find_smarts_matches:
            OE_TOOLKIT_CACHE_find_smarts_matches[cache_key] = oe_original_find_smarts_matches(self, molecule, *args, **kwargs)
        return OE_TOOLKIT_CACHE_find_smarts_matches[cache_key]
    # replace the original function with new one
    OpenEyeToolkitWrapper.find_smarts_matches = oe_cached_find_smarts_matches

    # cache for RDK find_smarts_matches
    rdk_original_find_smarts_matches = RDKitToolkitWrapper.find_smarts_matches
    RDK_TOOLKIT_CACHE_find_smarts_matches = {}
    def rdk_cached_find_smarts_matches(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in RDK_TOOLKIT_CACHE_find_smarts_matches:
            RDK_TOOLKIT_CACHE_find_smarts_matches[cache_key] = rdk_original_find_smarts_matches(self, molecule, *args, **kwargs)
        return RDK_TOOLKIT_CACHE_find_smarts_matches[cache_key]
    # replace the original function with new one
    RDKitToolkitWrapper.find_smarts_matches = rdk_cached_find_smarts_matches


    # cache for the validate function (save 94s)
    # The ChemicalEnvironment class has been deprecated in the OpenFF Toolkit 0.14 release.
    # from openff.toolkit.typing.chemistry.environment import ChemicalEnvironment
    # original_validate = ChemicalEnvironment.validate
    # TOOLKIT_CACHE_ChemicalEnvironment_validate = {}
    # def cached_validate(smirks, validate_valence_type=True, toolkit_registry=OpenEyeToolkitWrapper):
    #     cache_key = hash((smirks, validate_valence_type, toolkit_registry))
    #     if cache_key not in TOOLKIT_CACHE_ChemicalEnvironment_validate:
    #         TOOLKIT_CACHE_ChemicalEnvironment_validate[cache_key] = original_validate(smirks, validate_valence_type=validate_valence_type, toolkit_registry=toolkit_registry)
    #     return TOOLKIT_CACHE_ChemicalEnvironment_validate[cache_key]
    # ChemicalEnvironment.validate = cached_validate


    # cache for compute_partial_charges_am1bcc (save 69s)
    # No longer needed as of 0.7.0 since all partial charge assignment is routed through ToolkitWrapper.assign_partial_charges
    # original_compute_partial_charges_am1bcc = OpenEyeToolkitWrapper.compute_partial_charges_am1bcc
    # TOOLKIT_CACHE_compute_partial_charges_am1bcc = {}
    # def cached_compute_partial_charges_am1bcc(self, molecule, use_conformers=None, strict_n_conformers=False):
    #     cache_key = hash(molecule, use_conformers, strict_n_conformers)
    #     if cache_key not in TOOLKIT_CACHE_compute_partial_charges_am1bcc:
    #         TOOLKIT_CACHE_compute_partial_charges_am1bcc[cache_key] = original_compute_partial_charges_am1bcc(self, molecule, use_conformers=use_conformers, strict_n_conformers=strict_n_conformers)
    #     return TOOLKIT_CACHE_compute_partial_charges_am1bcc[cache_key]
    # OpenEyeToolkitWrapper.compute_partial_charges_am1bcc = cached_compute_partial_charges_am1bcc


    # Cache for OETK assign_partial_charges
    oe_original_assign_partial_charges = OpenEyeToolkitWrapper.assign_partial_charges
    OE_TOOLKIT_CACHE_assign_partial_charges = {}
    def oe_cached_assign_partial_charges(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in OE_TOOLKIT_CACHE_assign_partial_charges:
            oe_original_assign_partial_charges(self, molecule, *args, **kwargs)
            OE_TOOLKIT_CACHE_assign_partial_charges[cache_key] = molecule.partial_charges
        else:
            molecule.partial_charges = OE_TOOLKIT_CACHE_assign_partial_charges[cache_key]
        return
    OpenEyeToolkitWrapper.assign_partial_charges = oe_cached_assign_partial_charges


    # Cache for AmberTools assign_partial_charges
    at_original_assign_partial_charges = AmberToolsToolkitWrapper.assign_partial_charges
    AT_TOOLKIT_CACHE_assign_partial_charges = {}
    def at_cached_assign_partial_charges(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in AT_TOOLKIT_CACHE_assign_partial_charges:
            at_original_assign_partial_charges(self, molecule, *args, **kwargs)
            AT_TOOLKIT_CACHE_assign_partial_charges[cache_key] = molecule.partial_charges
        else:
            molecule.partial_charges = AT_TOOLKIT_CACHE_assign_partial_charges[cache_key]
        return
    AmberToolsToolkitWrapper.assign_partial_charges = at_cached_assign_partial_charges


    # cache the OE generate_conformers function (save 15s)
    OE_TOOLKIT_CACHE_molecule_conformers = {}
    oe_original_generate_conformers = OpenEyeToolkitWrapper.generate_conformers
    def oe_cached_generate_conformers(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in OE_TOOLKIT_CACHE_molecule_conformers:
            oe_original_generate_conformers(self, molecule, *args, **kwargs)
            OE_TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
        molecule._conformers = OE_TOOLKIT_CACHE_molecule_conformers[cache_key]
    OpenEyeToolkitWrapper.generate_conformers = oe_cached_generate_conformers


    # cache the RDKit generate_conformers function
    RDK_TOOLKIT_CACHE_molecule_conformers = {}
    rdk_original_generate_conformers = RDKitToolkitWrapper.generate_conformers
    def rdk_cached_generate_conformers(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in RDK_TOOLKIT_CACHE_molecule_conformers:
            rdk_original_generate_conformers(self, molecule, *args, **kwargs)
            RDK_TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
        molecule._conformers = RDK_TOOLKIT_CACHE_molecule_conformers[cache_key]
    RDKitToolkitWrapper.generate_conformers = rdk_cached_generate_conformers


    # final timing: 56s

    # cache the ForceField creation (no longer needed since using OpenFF API for parameter modifications)

    # import hashlib
    # from openff.toolkit.typing.engines.smirnoff import ForceField
    # SMIRNOFF_FORCE_FIELD_CACHE = {}
    # def getForceField(*ffpaths):
    #     hasher = hashlib.md5()
    #     for path in ffpaths:
    #         with open(path, 'rb') as f:
    #             hasher.update(f.read())
    #     cache_key = hasher.hexdigest()
    #     if cache_key not in SMIRNOFF_FORCE_FIELD_CACHE:
    #         SMIRNOFF_FORCE_FIELD_CACHE[cache_key] = ForceField(*ffpaths, allow_cosmetic_attributes=True)
    #     return SMIRNOFF_FORCE_FIELD_CACHE[cache_key]
