#!/usr/bin/env python
"""
setup.py: Install ForceBalance. 
"""
VERSION="1.1" # Make sure to change the version here, and also in bin/ForceBalance.py, doc/header.tex and doc/doxygen.cfg!
__author__ = "Lee-Ping Wang"
__version__ = VERSION

from distutils.sysconfig import get_config_var
from distutils.core import setup,Extension
import os,sys
import shutil
import glob

try:
    import numpy
except ImportError:
    print "Couldn't import numpy but this is required to install ForceBalance"
    print "Please install the numpy package and try again"
    exit()

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

# Copied from MSMBuilder 'contact' library for rapidly computing interatomic distances.
CONTACT = Extension('forcebalance/_contact_wrap',
                    sources = ["ext/contact/contact.c",
                               "ext/contact/contact_wrap.c"],
                    extra_compile_args=["-std=c99","-O3","-shared",
                                        "-fopenmp", "-Wall"],
                    extra_link_args=['-lgomp'],
                    include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "forcebalance"
    setupKeywords["version"]           = VERSION
    setupKeywords["author"]            = "Lee-Ping Wang"
    setupKeywords["author_email"]      = "leeping@stanford.edu"
    setupKeywords["license"]           = "GPL 3.0"
    setupKeywords["url"]               = "https://simtk.org/home/forcebalance"
    setupKeywords["download_url"]      = "https://simtk.org/home/forcebalance"
    setupKeywords["scripts"]           = glob.glob("bin/*.py") + glob.glob("bin/*.sh") + glob.glob("bin/ForceBalance") + glob.glob("bin/TidyOutput")
    setupKeywords["packages"]          = ["forcebalance","forcebalance/pymbar"]
    setupKeywords["package_dir"]       = {"forcebalance"         : "src",
                                          "forcebalance/pymbar"  : "ext/pymbar"
                                          }
    setupKeywords["package_data"]      = {
        "forcebalance"                   : ["AUTHORS","LICENSE.txt","data/*.py","data/*.sh","data/uffparms.in","data/oplsaa.ff/*"]
                                         }
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = [CMBAR, DCD, PERMUTE, CONTACT]
    setupKeywords["platforms"]         = ["Linux"]
    setupKeywords["description"]       = "Automated force field optimization."
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

    outputString=""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setupKeywords.iterkeys() ):
         value         = setupKeywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    
    print "%s" % outputString

    return setupKeywords

def doClean():
    """Remove existing forcebalance module folder before installing"""
    try:
        dir=os.path.dirname(__import__('forcebalance').__file__)
    except ImportError:
        print "Couldn't find existing forcebalance installation. Nothing to clean..\n"
        return

    raw_input("All files in %s will be deleted for clean install\nPress <Enter> to continue, <Ctrl+C> to abort\n" % dir)
    shutil.rmtree(dir)
    
def main():
    # if len(os.path.split(__file__)[0]) > 0:
    #     os.chdir(os.path.split(__file__)[0])

    for i, option in enumerate(sys.argv):
        if option == "-c" or option== "--clean":
            doClean()
            del sys.argv[i]
            break
    
    setupKeywords=buildKeywordDictionary()
    setup(**setupKeywords)

    shutil.rmtree('build')

if __name__ == '__main__':
    main()

