from builtins import object
import os, re
import subprocess
from shutil import which
from packaging.version import Version
import forcebalance.output

forcebalance.output.getLogger("forcebalance.test").propagate=False

os.chdir(os.path.dirname(__file__))
__all__ = [module[:-3] for module in sorted(os.listdir('.'))
           if re.match("^test_.*\.py$",module)]


class ForceBalanceTestCase(object):
    @classmethod
    def setup_class(cls):
        """Override default test case constructor to set longMessage=True, reset cwd after test
        @override unittest.TestCase.__init(methodName='runTest')"""

        cls.logger = forcebalance.output.getLogger('forcebalance.test.' + __name__[5:])
        cls.start_directory = os.getcwd()

        # unset this env to prevent error in mdrun
        if 'OMP_NUM_THREADS' in os.environ:
            os.environ.pop('OMP_NUM_THREADS')


    def setup_method(self, method):
        # Change to the test files directory before each test
        self._test_files_dir = os.path.join(os.path.dirname(__file__), "files")
        if os.path.isdir(self._test_files_dir):
            os.chdir(self._test_files_dir)
        else:
            # If the files directory does not exist, stay in the current directory
            self._test_files_dir = None

    def teardown_method(self):
        # Restore the original directory after each test
        os.chdir(self.start_directory)

def check_for_openmm():
    try:
        try:
            # Try importing openmm using >=7.6 namespace
            from openmm import app
            import openmm as mm
            from openmm import unit
        except ImportError:
            # Try importing openmm using <7.6 namespace
            import simtk.openmm as mm
            from simtk.openmm import app
            from simtk import unit
        return True
    except ImportError:
        # If OpenMM classes cannot be imported, then set this flag 
        # so the testing classes/functions can use to skip.
        return False


def get_gromacs_version_from_path(gmxpath):
    """Return the parsed GROMACS version for gmxpath, or None if unavailable.

    gmxpath may be a directory containing the gmx executable, or the full path
    to the executable itself.
    """
    candidates = [gmxpath]
    if os.path.isdir(gmxpath):
        candidates = [
            os.path.join(gmxpath, 'gmx_d'),
            os.path.join(gmxpath, 'gmx'),
        ]
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            version_output = result.stdout + result.stderr
            match = re.search(r'GROMACS\s+version:?\s*(\d+(?:\.\d+)*)', version_output, re.IGNORECASE)
            if match:
                return Version(match.group(1))
        except Exception:
            continue
    return None


def get_gromacs_version():
    """Return the detected system GROMACS version as a Version object, or None."""
    for candidate in [which('gmx_d'), which('gmx'), which('mdrun_d'), which('mdrun')]:
        if candidate:
            gmx_version = get_gromacs_version_from_path(candidate)
            if gmx_version is not None:
                return gmx_version
    return None


def is_buggy_gmx_dump_version(gmx_version):
    """Return True for versions affected by https://gitlab.com/gromacs/gromacs/-/issues/5124."""
    return gmx_version is not None and Version("2020") < gmx_version < Version("2024.3")
