from __init__ import ForceBalanceTestCase
import unittest
import forcebalance
import os, re, shutil, sys
import numpy as np
from subprocess import call
from collections import OrderedDict
from forcebalance.openmmio import OpenMM

class TestRPMD(ForceBalanceTestCase):

    """ RPMD unit test that runs a short RPMD simulation
    and compares calculated potential energies from
    MD with postprocessed potential energies as an
    internal consistency check.
    """

    def setUp(self):
        self.logger.debug('\nBuilding OpenMM engine...\n')
        os.chdir(os.path.join('test','files','rpmd_files'))
        openmm = False
        try:
            import simtk.openmm
            openmm = True
        except: logger.warn("OpenMM cannot be imported. Make sure it's installed!")
        print os.getcwd()        

        if openmm:
            self.ommEngine = OpenMM(ffxml='qtip4pf.xml', coords='liquid.pdb', pbc=True, platname='CUDA', precision='double')
             
    def test_rpmd_simulation(self):
        self.logger.debug('\nRunning MD...\n')
        if not os.path.exists('temp'): os.mkdir('temp')
        os.chdir('temp')
        # We're in the temp directory so need to copy the force field file here.
        shutil.copy2('../qtip4pf.xml','./qtip4pf.xml')
        self.addCleanup(os.system, 'cd .. ; rm -rf temp')

        MD_data = self.ommEngine.molecular_dynamics(nsteps=1000, nsave=100, timestep=0.5, temperature=300, pressure=1.0, verbose=False, save_traj=True, rpmd_opts=['32','9'])
        # Line below performs same MD run, but using verbose option
        #MD_data = self.ommEngine.molecular_dynamics(nsteps=1000, nsave=100, timestep=0.5, temperature=300, pressure=1.0, verbose=True, save_traj=True, rpmd_opts=['32','9'])
        postprocess_potentials = self.ommEngine.evaluate_(traj=self.ommEngine.xyz_rpmd, dipole=True)
        self.assertEqual(MD_data['Dips'].all(), postprocess_potentials['Dipole'].all())
        self.assertEqual(MD_data['Potentials'].all(), postprocess_potentials['Energy'].all())

suite = unittest.TestLoader().loadTestsFromTestCase(TestRPMD)
unittest.TextTestRunner().run(suite)
#if __name__ == '__main__':
#    unittest.main()
