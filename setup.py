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
import argparse

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
                    
def installNetworkX():          
    setup(
        packages=["networkx",
          "networkx.algorithms",
          "networkx.algorithms.assortativity",
          "networkx.algorithms.bipartite",
          "networkx.algorithms.centrality",
          "networkx.algorithms.chordal",
          "networkx.algorithms.community",
          "networkx.algorithms.components",
          "networkx.algorithms.flow",
          "networkx.algorithms.traversal",
          "networkx.algorithms.isomorphism",
          "networkx.algorithms.shortest_paths",
          "networkx.algorithms.link_analysis",
          "networkx.algorithms.operators",
          "networkx.algorithms.approximation",
          "networkx.classes",
          "networkx.external",
          "networkx.external.decorator",
          "networkx.generators",
          "networkx.drawing",
          "networkx.linalg",
          "networkx.readwrite",
          "networkx.readwrite.json_graph",
          "networkx.tests",
          "networkx.testing",
          "networkx.utils"],
        package_dir      = {"networkx" : "ext/networkx"},
        description = "Python package for creating and manipulating graphs and networks",
        long_description = \
        """
        NetworkX is a Python package for the creation, manipulation, and
        study of the structure, dynamics, and functions of complex networks.
        """,
        license = 'BSD',
        authors = {'Hagberg' : ('Aric Hagberg','hagberg@lanl.gov'),
                   'Schult' : ('Dan Schult','dschult@colgate.edu'),
                   'Swart' : ('Pieter Swart','swart@lanl.gov')
                   },
        maintainer = "NetworkX Developers",
        maintainer_email = "networkx-discuss@googlegroups.com",
        url = 'http://networkx.lanl.gov/',
        download_url="http://networkx.lanl.gov/download/networkx",
        platforms = ['Linux','Mac OSX','Windows','Unix'],
        keywords = ['Networks', 'Graph Theory', 'Mathematics', 'network', 'graph', 'discrete mathematics', 'math'],
        classifiers = [
                'Development Status :: 4 - Beta',
                'Intended Audience :: Developers',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Operating System :: OS Independent',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.6',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.1',
                'Programming Language :: Python :: 3.2',
                'Topic :: Software Development :: Libraries :: Python Modules',
                'Topic :: Scientific/Engineering :: Bio-Informatics',
                'Topic :: Scientific/Engineering :: Information Analysis',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics'],
      )
      
    try:
        m = __import__("networkx")
    except ImportError:
        raw_input("Error installing networkx, press <Enter> to continue ForceBalance installation")

def buildKeywordDictionary(args):
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

    if args.clean: doClean()
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
        dir=os.path.dirname(__import__('forcebalance').__file__)
    except ImportError:
        print "Couldn't find existing forcebalance installation. Nothing to clean...\n"
        return
    except:
        print "Couldn't read forcebalance location... Continuing with regular install"
        return

    raw_input("All files in %s will be deleted for clean install\nPress <Enter> to continue, <Ctrl+C> to abort\n" % dir)
    shutil.rmtree(dir)
    
def main():
    # if len(os.path.split(__file__)[0]) > 0:
    #     os.chdir(os.path.split(__file__)[0])

    ## Install options
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true', help='remove previously installed forcebalance installation first')
    parser.add_argument('-t', '--test', action='store_true', help='install forcebalance test suite')
    parser.add_argument('-g', '--gui', action='store_true', help='install forcebalance gui module')
    args, sys.argv= parser.parse_known_args(sys.argv)
    
    try:
        __import__("networkx")
    except ImportError:
        print "Could not import networkx module! Topology tools will not work without this..."
        if raw_input("Would you like to install it now (y/n)? ").lower() == 'y':
            installNetworkX()
            print "NetworkX module successfully installed!"
    
    setupKeywords=buildKeywordDictionary(args)
    setup(**setupKeywords)

    shutil.rmtree('build')

if __name__ == '__main__':
    main()

