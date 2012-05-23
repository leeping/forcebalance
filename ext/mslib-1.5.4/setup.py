from distutils.core import setup, Extension
from distutils.cmd import Command
from distutils.command.install_data import install_data
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist
from glob import glob

from distutils.file_util import copy_file
from distutils.util import get_platform

import sys, os, re
from os import path

libVersion = "1.4.4"

# platform we are building Python extension for:
platform =  sys.platform

# amd64
if (platform == "linux2"):
    lUname = os.uname()
    if lUname[-1] == 'x86_64':
        platform = lUname[-1] + lUname[0] + lUname[2][0]

#location of prebuilt C-library libmsms:

clib_dir = path.join(".", "lib", platform)
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
        
pack_name = "mslib"
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
        
# list of the python packages to be included in this distribution.
# sdist doesn't go recursively into subpackages so they need to be
# explicitaly listed.
# From these packages only the python modules will be taken
packages = ['mslib', 'mslib.Tests']

# list of the python modules not part of a package. Give the path and the
# filename without the extension. i.e you want to add the
# test.py module which is located in MyPack/Tests/ you give
# 'MyPack/Tests/test'
py_modules = []


# list of the files that are not python packages but are included in the
# distribution and need to be installed at the proper place  by distutils.
# The list in MANIFEST.in lists is needed for including those files in
# the distribution, data_files in setup.py is needed to install them
# at the right place.
data_files = []
for dir in ['mslib/Tests/Data']:
    files = []
    for f in glob(os.path.join(dir, '*')):
        if f[-3:] != '.py' and f[-4:-1] != '.py' and os.path.isfile(f):
            files.append(f)
    data_files.append((dir, files))


source = path.join(pack_name, "msms.i")

if platform == "win32":
    clib_name = "mslib"
else:
#    if platform in ("linux2", "sunos5", "irix6", "darwin"):
#       libVersion = "1.4.1" 
    clib_name = "msms"+libVersion
    # use dinamic mslib instead of static
    #dinamiclib = path.join(clib_dir, "lib"+clib_name+"_d.so")
    #if (path.exists(dinamiclib)):
    #    os.rename(dinamiclib, path.join(clib_dir, "lib"+clib_name+".so"))
    
import numpy
numpy_include =  numpy.get_include()
incl_dir = [path.join(".","include"), numpy_include]

comp_args = []
#if platform == "win32":
#    comp_args = ["/Za"]

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
             packages=packages,
             py_modules=py_modules,
             data_files=data_files,
             cmdclass = {'build':modified_build,
                         'build_ext':modified_build_ext,
                         'sdist': modified_sdist,
                         'install_data':modified_install_data},
             ext_package=pack_name,
             ext_modules=[Extension (ext_name, [source],
                                     include_dirs=incl_dir,
                                     define_macros=[('LONGJMP_EXIT', None),],
                                     library_dirs=[clib_dir],
                                     libraries=[clib_name],
                                     extra_compile_args = comp_args
                                     )
                          ] ,
             )


# Remove the .py and wrap.c files generated by swig from the
# MyPack directory
import os
pyName = './%s/%slib.py'%(pack_name, pack_name)
if os.path.exists(pyName):
    os.system("rm -f %s"%pyName)
    print "Removing %s generated by swig"%pyName
wrapName = './%s/%slib_wrap.c'%(pack_name, pack_name)
if os.path.exists(wrapName):
    os.system("rm -f %s"%wrapName)
    print "Removing %s generated by swig"%wrapName


