"""
setup.py: Install ForceBalance. 
"""
VERSION="0.13.0"
__author__ = "Lee-Ping Wang"
__version__ = VERSION

from distutils.sysconfig import get_config_var
from distutils.core import setup,Extension
import os
import numpy
import glob

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


def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "ForceBalance"
    setupKeywords["version"]           = VERSION
    setupKeywords["author"]            = "Lee-Ping Wang"
    setupKeywords["author_email"]      = "leeping@stanford.edu"
    setupKeywords["license"]           = "GPL 3.0"
    setupKeywords["url"]               = "https://simtk.org/home/forcebalance"
    setupKeywords["download_url"]      = "https://simtk.org/home/forcebalance"
    setupKeywords["scripts"]           = glob.glob("bin/*.py") + glob.glob("bin/*.sh")
    setupKeywords["packages"]          = ["forcebalance","forcebalance/pymbar"]
    setupKeywords["package_dir"]       = {"forcebalance"        : "src",
                                          "forcebalance/pymbar" : "ext/pymbar"
                                          }
    setupKeywords["package_data"]      = {
        "ForceBalance"                   : ["AUTHORS","LICENSE.txt"]
                                         }
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = [CMBAR, DCD, PERMUTE]
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
    

def main():
    setupKeywords=buildKeywordDictionary()
    setup(**setupKeywords)

if __name__ == '__main__':
    main()

