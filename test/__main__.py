import unittest, os, sys, re
import forcebalance
from forcebalance import logging
from __init__ import ForceBalanceTestRunner
import getopt

def getOptions():
    """Parse options passed to forcebalance testing framework"""

    exclude = []
    options = {
        'pretend':False
        }
    # handle options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "me:pvq", ["metatest","exclude=","pretend", "verbose", "quiet"])
        options['loglevel']=logging.INFO
        if args:
            options['test_modules']=['test_' + module for module in args]
        for o, a in opts:
            if o in ("-m", "--metatest"):
                options['test_modules']=['__test__']
            elif o in ("-e", "--exclude"):
                exclude=a.split(',')    # used below when generating module list
            if o in ("-p", "--pretend"):
                options['pretend']=True
            if o in ("-v", "--verbose"):
                options['loglevel']=logging.DEBUG
            elif o in ("-q", "--quiet"):
                options['loglevel']=logging.WARNING
                
    except getopt.GetoptError as err:
        usage()
        sys.exit()
    
    return options

def usage():
    """Print information on running tests using this script"""

    print """ForceBalance Test Suite
Usage: python test [OPTIONS] [MODULES]
If no modules are specified, all test modules in test/ are run

Valid options are:
-e, \t\t\tDo not run tests for MODULE
--exclude=MODULE[,MODULE2[,...]]
-m, --metatest\t\tRun tests on testing framework
-p, --pretend\t\tLoad tests but don't actually run them
-v, --verbose\t\tSet log level to DEBUG, printing additional test information
-q, --quiet\t\tSet log level to WARNING, printing only on failure or error
"""


#### main block ####

options=getOptions()
logging.getLogger("test").addHandler(forcebalance.nifty.RawStreamHandler(sys.stderr))
runner=ForceBalanceTestRunner()
results=runner.run(**options)
