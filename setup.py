"""
setup.py: Install ForceBalance. 
"""
VERSION="0.12.0"
__author__ = "Lee-Ping Wang"
__version__ = VERSION

from distutils.core import setup,Extension
import numpy
import glob

# Stuff from the msmbuilder setup script (external extensions)
# Retained here for reference
# IRMSD = Extension('msmbuilder/rmsdcalc',
#                   sources = ["msmbuilder/IRMSD/theobald_rmsd.c","msmbuilder/IRMSD/rmsd_numpy_array.c"],
#                   extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3","-fopenmp"],
#                   extra_link_args=['-lgomp'],
#                   include_dirs = [numpy.get_include(),numpy.get_include()+"/numpy/"]
#                   )
# XTC = Extension('msmbuilder/libxdrfile',
#                   sources = [
#                     "xdrfile-1.1b/src/trr2xtc.c",
#                     "xdrfile-1.1b/src/xdrfile.c",
#                     "xdrfile-1.1b/src/xdrfile_trr.c",
#                     "xdrfile-1.1b/src/xdrfile_xtc.c",
#                             ],
#                   extra_compile_args=[],
#                   extra_link_args=["--enable-shared"],
#                   include_dirs = ["xdrfile-1.1b/include/"]
#                   )
# DCD = Extension('msmbuilder/dcdplugin_s',
#                 sources = [ "molfile_plugin/dcdplugin_s.c" ],
#                 libraries=['m'],
#                 include_dirs = ["molfile_plugin/include/","molfile_plugin"] 
#                 )

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
    setupKeywords["packages"]          = ["forcebalance"]
    setupKeywords["scripts"]           = glob.glob("bin/*.py") + glob.glob("bin/*.sh")
    setupKeywords["package_data"]      = {
        "ForceBalance"                   : ["AUTHORS","LICENSE.txt"]
                                         }
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = []
    setupKeywords["platforms"]         = ["Linux"]
    setupKeywords["description"]       = "Automated force field optimization."
    setupKeywords["long_description"]  = """

    ForceBalance (https://simtk.org/home/forcebalance ) is a library
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
