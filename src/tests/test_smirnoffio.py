import os

import pytest

from forcebalance.nifty import logger
from forcebalance.smirnoffio import SMIRNOFF

from .__init__ import ForceBalanceTestCase


class TestSMIRNOFF(ForceBalanceTestCase):
    """Test behavior of SMIRNOFF class"""

    @classmethod
    def setup_class(cls):
        """
        setup any state specific to the execution of the given class (which usually contains tests).
        """
        super(TestSMIRNOFF, cls).setup_class()

        cls.cwd = os.path.dirname(os.path.realpath(__file__))

        os.chdir(os.path.join(cls.cwd, "files", "opc"))
        cls.tmpfolder = os.path.join(cls.cwd, "files", "opc", "temp")

        if not os.path.exists(cls.tmpfolder):
            os.makedirs(cls.tmpfolder)

        os.chdir(cls.tmpfolder)

        pytest.importorskip("openeye.oechem")
        from openeye import oechem

        if not oechem.OEChemIsLicensed():
            pytest.skip("Need OEChem license to run this test")
        for file in ["dimer.pdb", "dimer.mol2", "opc.offxml"]:
            os.system(f"ln -fs ../{file}")

        cls.engines = dict()

        try:
            import openmm  # noqa

            cls.engines["SMIRNOFF"] = SMIRNOFF(
                mol2=["dimer.mol2"],
                coords="dimer.pdb",
                ffxml="opc.offxml",
                platname="Reference",
                precision="double",
            )

        except ModuleNotFoundError:
            logger.warning("OpenMM cannot be imported, skipping OpenMM tests.")

    def test_energy_with_virtual_sites(self):

        data = {name: eng.energy_force() for name, eng in self.engines.items()}

        assert data['SMIRNOFF'] is not None