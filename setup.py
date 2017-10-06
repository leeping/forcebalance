#!/usr/bin/env python
"""
setup.py: Install ForceBalance. 
"""
__author__ = "Lee-Ping Wang"

from setuptools import setup,Extension
import os,sys,re
import shutil
import glob
import argparse
import subprocess

try:
    import numpy
    import scipy
except ImportError:
    print "Error importing numpy and scipy but these are required to install ForceBalance"
    print "Please make sure the numpy and scipy modules are installed and try again"
    exit()
    
#===================================#
#|   ForceBalance version number   |#
#| Make sure to update the version |#
#| manually in :                   |#
#|                                 |#
#| doc/header.tex                  |#
#| doc/api_header.tex              |#
#| src/__init__.py                 |#
#===================================#
__version__ = "v1.3.2"
try:
    # use git to find current version
    git_describe = subprocess.check_output(["git", "describe"]).strip()
    __version__ = re.sub('-g[0-9a-f]*$','',git_describe)
except: pass

# The versioning file logic does not work.  
# Commenting out until further notice.
# versioning_file = os.path.join(os.path.dirname(__file__), '.__version__')
# try:
#     git_describe = subprocess.check_output(["git", "describe"]).strip()
#     __version__ = re.sub('-g[0-9a-f]*$','',git_describe)
#     with open(versioning_file, 'w') as fh:
#         fh.write(__version__)
#     #subprocess.call(["git", "add", ".__version__"])
# except:
#     with open(versioning_file, 'r') as fh:
#         __version__ = fh.read().strip()


# DCD file reading module
DCD = Extension('forcebalance/_dcdlib',
                sources = [ "ext/molfile_plugin/dcdplugin_s.c" ],
                libraries=['m'],
                include_dirs = ["ext/molfile_plugin/include/","ext/molfile_plugin"] 
                )

# Multistate Bennett acceptance ratios
CMBAR = Extension('forcebalance/pymbar/_pymbar',
                  sources = ["ext/pymbar/_pymbar.c"],
                  extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3"],
                  include_dirs = [numpy.get_include(),numpy.get_include()+"/numpy/"]
                  )

# Hungarian algorithm for permutations
# Used for identifying normal modes
PERMUTE = Extension('forcebalance/_assign',
                    sources = ['ext/permute/apc.c', 'ext/permute/assign.c'],
                    include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]
                    )

# 'contact' library from MSMBuilder for rapidly computing interatomic distances.
# If we're on Mac OS, it can't find the OpenMP libraries
import platform
if platform.system() == 'Darwin':
    CONTACT = Extension('forcebalance/_contact_wrap',
                        sources = ["ext/contact/contact.c",
                                   "ext/contact/contact_wrap.c"],
                        extra_compile_args=["-std=c99","-O3","-shared",
                                            "-Wall"],
                        include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
else:
    CONTACT = Extension('forcebalance/_contact_wrap',
                        sources = ["ext/contact/contact.c",
                                   "ext/contact/contact_wrap.c"],
                        extra_compile_args=["-std=c99","-O3","-shared",
                                            "-fopenmp", "-Wall"],
                        extra_link_args=['-lgomp'],
                        include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
    

def buildKeywordDictionary(args):
    setupKeywords = {}
    setupKeywords["name"]              = "forcebalance"
    # Don't create a separate installed version number for every commit
    setupKeywords["version"]           = re.sub('-[0-9]*$','',__version__)
    setupKeywords["author"]            = "Lee-Ping Wang, Arthur Vigil"
    setupKeywords["author_email"]      = "leeping@stanford.edu"
    setupKeywords["license"]           = "GPL 3.0"
    setupKeywords["url"]               = "https://simtk.org/home/forcebalance"
    setupKeywords["download_url"]      = "https://simtk.org/home/forcebalance"
    setupKeywords["scripts"]           = glob.glob("bin/*.py") + glob.glob("bin/*.sh") + glob.glob("bin/*.bash") + glob.glob("bin/ForceBalance") + glob.glob("bin/TidyOutput")
    setupKeywords["packages"]          = ["forcebalance","forcebalance/pymbar"]
    setupKeywords["package_dir"]       = {"forcebalance"         : "src",
                                          "forcebalance/pymbar"  : "ext/pymbar"
                                          }
    setupKeywords["package_data"]      = {
        "forcebalance"                   : ["AUTHORS","LICENSE.txt","data/*.py","data/*.sh","data/*.bash","data/uffparms.in","data/oplsaa.ff/*"]
                                         }
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = [CMBAR, DCD, PERMUTE, CONTACT]
    setupKeywords["platforms"]         = ["Linux"]
    setupKeywords["description"]       = "Automated force field optimization."
    setupKeywords["install_requires"]  = ['networkx>=1.9,<2.0', 'decorator>=3.4.0']
    setupKeywords["long_description"]  = """

    ForceBalance (https://simtk.org/home/forcebalance) is a library
    that provides tools for automated optimization of force fields and
    empirical potentials.

    The philosophy of this program is to present force field
    optimization in a unified and easily extensible framework.  Since
    there are many different ways in theoretical chemistry to compute
    the potential energy of a collection of atoms, and similarly many
    types of reference data to fit these potentials to, we do our best
    to provide an infrastructure which allows a user or a contributor
    to fit any type of potential to any type of reference data.

    """

    if not args.dirty: doClean()
    setupKeywords["packages"].append("forcebalance.unit")
    if args.test:
        setupKeywords["packages"].append("forcebalance.test")
        setupKeywords["package_dir"].update({"forcebalance.test" : "test"})
        
        os.chdir("test") # change directories to glob test files
        test_data = glob.glob("files/*.*") + glob.glob("files/forcefield/*.*") + glob.glob("files/targets/*/*.*") + glob.glob("files/*.*") + ["files/work_queue_worker"]
        os.chdir("..")
        setupKeywords["package_data"].update({'forcebalance.test': test_data})
    if args.gui:
        setupKeywords["packages"].append("forcebalance.gui")

    return setupKeywords

def doClean():
    """Remove existing forcebalance module folder before installing"""
    try:
        forcebalance_dir=os.path.dirname(__import__('forcebalance').__file__)
    except ImportError:
        print "Couldn't find existing forcebalance installation. Nothing to clean...\n"
        return
    except:
        print "Couldn't read forcebalance location... Continuing with regular install"
        return

    #raw_input("All files in %s will be deleted for clean\nPress <Enter> to continue, <Ctrl+C> to abort\n" % forcebalance_dir)
    print "Removing the directory tree prior to install: %s" % forcebalance_dir
    subprocess.call("rm -f %s/../forcebalance-*.egg-info" % forcebalance_dir, shell=True)
    if os.path.exists(forcebalance_dir):
        shutil.rmtree(forcebalance_dir, ignore_errors=True)
    
def main():
    # if len(os.path.split(__file__)[0]) > 0:
    #     os.chdir(os.path.split(__file__)[0])

    ## Install options
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirty', action='store_true', help="don't remove previously installed forcebalance installation first")
    parser.add_argument('-t', '--test', action='store_true', help='install forcebalance test suite')
    parser.add_argument('-g', '--gui', action='store_true', help='install forcebalance gui module')
    args, sys.argv= parser.parse_known_args(sys.argv)
    setupKeywords=buildKeywordDictionary(args)
    ## Run setuptools command.
    ## Refer to setup.cfg for customizing installation behavior
    setup(**setupKeywords)

    if os.path.exists('build'):
        shutil.rmtree('build')


    try:
        import bz2
    except ImportError:
        print "Error importing bz2, which is important for distributed calculations and remote targets"
        print "Please either (1) make sure Python is built/installed with bz2 support"
        print "or (2) proceed with inefficient reading/writing of files; remote targets won't work."

if __name__ == '__main__':
    main()

