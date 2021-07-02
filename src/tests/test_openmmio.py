from __future__ import absolute_import
import forcebalance
from forcebalance.openmmio import PrepareVirtualSites, ResetVirtualSites_fast
import simtk.openmm as mm
from simtk.openmm import app
import numpy as np
from simtk import unit
import os
import shutil
from .test_target import TargetTests # general targets tests defined in test_target.py
import pytest
"""
The testing functions for this class are located in test_target.py.
"""
class TestLiquid_OpenMM(TargetTests):
    def setup_method(self, method):
        pytest.skip("Needs optimizing to reduce runtime")
        super(TestLiquid_OpenMM, self).setup_method(method)
        # settings specific to this target
        self.options.update({
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})

        self.tgt_opt.update({'type':'LIQUID_OPENMM',
            'name':'dms-liquid'})

        self.ff = forcebalance.forcefield.FF(self.options)

        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.mvals = [.5]*self.ff.np

        self.target = forcebalance.openmmio.Liquid_OpenMM(self.options, self.tgt_opt, self.ff)
        self.target.stage(self.mvals)

    def teardown_method(self):
        shutil.rmtree('temp')
        super(TestLiquid_OpenMM, self).teardown_method()

class TestInteraction_OpenMM(TargetTests):

    def setup_method(self, method):
        super(TestInteraction_OpenMM, self).setup_method(method)
        # TargetTests.setup_class(cls)
        # settings specific to this target
        self.options.update({
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})

        self.tgt_opt.update({"type" : "Interaction_OpenMM",
                            "name" : "S2EPose",
                            "fragment1" : "1-9",
                            "fragment2" : "10-18"})

        self.ff = forcebalance.forcefield.FF(self.options)

        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.mvals = [.5]*self.ff.np

        self.target = forcebalance.openmmio.Interaction_OpenMM(self.options, self.tgt_opt, self.ff)

    def teardown_method(self):
        shutil.rmtree('temp')
        super(TestInteraction_OpenMM, self).teardown_method()


def test_local_coord_sites():
    """Make sure that the internal prep of vs positions matches that given by OpenMM."""
    # make a system
    mol = app.PDBFile(os.path.join("files", "vs_mol.pdb"))
    modeller = app.Modeller(topology=mol.topology, positions=mol.positions)
    forcefield = app.ForceField(os.path.join("files", "forcefield", "vs_mol.xml"))
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology, constraints=None)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    # update the site positions
    simulation.context.computeVirtualSites()
    # one vs and it should be last
    vs_pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)[-1]
    # now compute and compare
    vsinfo = PrepareVirtualSites(system=system)
    new_pos = ResetVirtualSites_fast(positions=modeller.positions, vsinfo=vsinfo)[-1]
    assert np.allclose(vs_pos._value, np.array([new_pos.x, new_pos.y, new_pos.z]))
