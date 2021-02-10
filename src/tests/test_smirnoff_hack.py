import pytest

has_openforcefield = True
try:
    import openforcefield
except ModuleNotFoundError:
    has_openforcefield = False


@pytest.mark.skipif(
    not has_openforcefield, reason="openforcefield/openff.toolkit module not found"
)
def test_smirnoff_hack():
    """Test basic behavior of smirnoff_hack.py, in particular
    the compatibility with Molecule API"""
    from openforcefield.topology import Molecule, Topology
    from openforcefield.typing.engines.smirnoff import ForceField
    from openforcefield.utils.toolkits import (AmberToolsToolkitWrapper,
                                               RDKitToolkitWrapper,
                                               ToolkitRegistry)

    from forcebalance import smirnoff_hack

    top = Topology.from_molecules(Molecule.from_smiles("CCO"))
    parsley = ForceField("openff-1.0.0.offxml")

    registry = ToolkitRegistry()
    registry.register_toolkit(RDKitToolkitWrapper)
    registry.register_toolkit(AmberToolsToolkitWrapper)
    out = parsley.create_openmm_system(top, toolkit_registry=registry)
