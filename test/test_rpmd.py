import unittest
import sys, os, re, shutil
from subprocess import call
import forcebalance
import numpy as np
from __init__ import ForceBalanceTestCase
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
        if openmm:
            self.ommEngine = OpenMM(ffxml='qtip4pf.xml', pdb='liquid.pdb', pbc=True, platname='CUDA', precision='double')
             
    def test_rpmd_simulation(self):
        self.logger.debug('\nRunning MD...\n')
        if not os.path.exists('temp'): os.mkdir('temp')
        os.chdir('temp')
        # We're in the temp directory so need to copy the force field file here.
        shutil.copy2('../qtip4pf.xml','./qtip4pf.xml')
        self.addCleanup(os.system, 'cd .. ; rm -r temp')
        #MD_data = self.ommEngine.molecular_dynamics(nsteps=1000, nsave=100, timestep=0.5, temperature=300, pressure=1.0, verbose=False, save_traj=True, rpmd_opts=['32','6'])
        # Line below performs same MD run, but using verbose option
        MD_data = self.ommEngine.molecular_dynamics(nsteps=400, nsave=100, timestep=0.5, temperature=300, verbose=True, save_traj=True, rpmd_opts=['32','6'])
        postprocess_potentials = self.ommEngine.evaluate_(traj=self.ommEngine.xyz_rpmd, dipole=True)
        #print MD_data['Dips']
        #print postprocess_potentials['Dipole']
        print MD_data['CV_T']
        print MD_data['CV_CV']

        os.chdir('../../..')
        os.chdir(os.path.join('test','files','rpmd_files','temp'))
        self.assertEqual(MD_data['Dips'].all(), postprocess_potentials['Dipole'].all())
        self.assertEqual(MD_data['Potentials'].all(), postprocess_potentials['Energy'].all())

suite = unittest.TestLoader().loadTestsFromTestCase(TestRPMD)
unittest.TextTestRunner().run(suite)
#if __name__ == '__main__':
#    unittest.main()
