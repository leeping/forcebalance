"""
setup.py: Install ForceBalance. 
"""
VERSION="0.12.0"
__author__ = "Lee-Ping Wang"
__version__ = VERSION

from distutils.sysconfig import get_config_var
from distutils.core import setup,Extension
import numpy
import glob
# These imports were required to install the mslib extension.
import sys, os, re
from os import path
from distutils.file_util import copy_file
from distutils.util import get_platform
from distutils.cmd import Command
from distutils.command.install_data import install_data
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist

#=============#
# mslib stuff #
#=============#

libVersion = "1.4.4"

# platform we are building Python extension for:
platform =  sys.platform

# amd64
if (platform == "linux2"):
    lUname = os.uname()
    if lUname[-1] == 'x86_64':
        platform = lUname[-1] + lUname[0] + lUname[2][0]

#location of prebuilt C-library libmsms:

clib_dir = path.join(".", "ext", "mslib-1.5.4", "lib", platform)
if platform == "darwin":
    arch = os.uname()[-1]
    if arch == 'Power Macintosh':
        clib_dir = path.join(clib_dir, "ppcDarwin")
    elif re.match("i*\d86", arch):
        clib_dir = path.join(clib_dir, "i86Darwin")
    else:
        print "Unsupported architecture: ", os.uname()

########################################################################
# Overwrite the run method of the install_data class to install data
# files in the package.
########################################################################

class modified_install_data(install_data):

    def run(self):
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

########################################################################
# Overwrite the sub_commands list of the build command so that
# the build_py is called after build_clib and build_ext. This way
# a python module generated  by SWIG in 'build_ext'command is copied to
#the build directory by 'build_py' command.
########################################################################

class modified_build(build):
    sub_commands = [('build_clib',    build.has_c_libraries),
                    ('build_ext',     build.has_ext_modules),
                    ('build_py',      build.has_pure_modules),
                    ('build_scripts', build.has_scripts),
                   ]
    
########################################################################

class modified_build_ext(build_ext):
    def run(self):
        if sys.platform == "darwin":
            # on MacOS ranlib command should be used for updating
            # the table of contents of archive library libmsms.a.
            #staticlib = path.join("lib", "darwin", "libmsms"+libVersion+".a")
            staticlib = path.join(clib_dir,  "libmsms"+libVersion+".a")
            if path.isfile(staticlib):
                self.spawn(["ranlib", "-s", staticlib]) 
        build_ext.run(self)

        
pack_name = "ext/mslib-1.5.4/mslib"
ext_name = "_msms"

#####################################################################
# Overwrite the prune_file_list method of sdist to not
# remove automatically the RCS/CVS directory from the distribution.
####################################################################

class modified_sdist(sdist):
    def prune_file_list(self):

        build = self.get_finalized_command('build')
        base_dir = self.distribution.get_fullname()
        self.filelist.exclude_pattern(None, prefix=build.build_base)
        self.filelist.exclude_pattern(None, prefix=base_dir)

if platform == "win32":
    clib_name = "mslib"
else:
    clib_name = "msms"+libVersion

try:   
    from version import VERSION
except:
    VERSION = "1.0"

dist = setup(name="mslib",
             version=VERSION,
             description="MSMS library python extension module",
             author="Michel F.Sanner",
             author_email="sanner@scripps.edu",
             url="http://www.scripps.edu/~sanner/software",
             license="to be specified",
             packages=['forcebalance/mslib'],
             package_dir={'forcebalance/mslib':'ext/mslib-1.5.4/mslib'},
             py_modules=[],
             data_files=[],
             cmdclass = {'build':modified_build,
                         'build_ext':modified_build_ext,
                         'sdist': modified_sdist,
                         'install_data':modified_install_data},
             ext_package="forcebalance/mslib",
             ext_modules=[Extension (ext_name, [path.join(pack_name, "msms.i")],
                                     include_dirs=[path.join(".","ext","mslib-1.5.4","include"), numpy.get_include()],
                                     define_macros=[('LONGJMP_EXIT', None),],
                                     library_dirs=[clib_dir],
                                     libraries=[clib_name],
                                     )])

# Remove the .py and wrap.c files generated by swig from the
# MyPack directory
import os
pyName = './ext/mslib-1.5.4/mslib/msms.py'
if os.path.exists(pyName):
    os.system("rm -f %s"%pyName)
    print "Removing %s generated by swig"%pyName
wrapName = './ext/mslib-1.5.4/mslib/msms_wrap.c'
if os.path.exists(wrapName):
    os.system("rm -f %s"%wrapName)
    print "Removing %s generated by swig"%wrapName

#======================#
# ForceBalance package #
#======================#

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

DCD = Extension('forcebalance/_dcdlib',
                sources = [ "ext/molfile_plugin/dcdplugin_s.c" ],
                libraries=['m'],
                include_dirs = ["ext/molfile_plugin/include/","ext/molfile_plugin"] 
                )

CMBAR = Extension('forcebalance/pymbar/_pymbar',
                  sources = ["ext/pymbar/_pymbar.c"],
                  extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3"],
                  include_dirs = [numpy.get_include(),numpy.get_include()+"/numpy/"]
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
                                          "forcebalance/pymbar" : "ext/pymbar"}
    setupKeywords["package_data"]      = {
        "ForceBalance"                   : ["AUTHORS","LICENSE.txt"]
                                         }
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = [CMBAR, DCD]
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

    This code includes a number of subpackages that are installed underneath
    the Python module's directory.  Here's what's included:

    The pymbar package by Michael Shirts and John Chodera (https://simtk.org/home/pymbar).  
    It is used in the PropertyMatch module.

    The msms package by Michel Sanner (http://mgltools.scripps.edu/packages/MSMS/).  
    It is used to generate grid points for electrostatic potential fitting.
    
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
