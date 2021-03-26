import pytest

has_openff_toolkit = True
try:
    import openff.toolkit
except ModuleNotFoundError:
    has_openff_toolkit = False


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_smirnoff_hack():
    """Test basic behavior of smirnoff_hack.py, in particular
    the compatibility with Molecule API"""
    from openff.toolkit.topology import Molecule, Topology
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.toolkit.utils.toolkits import (AmberToolsToolkitWrapper,
                                               RDKitToolkitWrapper,
                                               ToolkitRegistry)

    from forcebalance import smirnoff_hack

    top = Topology.from_molecules(Molecule.from_smiles("CCO"))
    parsley = ForceField("openff-1.0.0.offxml")

    registry = ToolkitRegistry()
    registry.register_toolkit(RDKitToolkitWrapper)
    registry.register_toolkit(AmberToolsToolkitWrapper)
    parsley.create_openmm_system(top, toolkit_registry=registry)
