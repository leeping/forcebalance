import unittest, os, sys, re
import forcebalance
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
        opts, args = getopt.getopt(sys.argv[1:], "e:mpi:", ["metatest","exclude=","pretend"])
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
                options['verbose']=True
    except getopt.GetoptError as err:
        usage()
        sys.exit()
    
    # if modules to be run were not already provided in args, generate them automatically
    if not options.has_key('test_modules'):
        options['test_modules'] = [module[:-3] for module in sorted(os.listdir('test'))
                                  if re.match("^test_.*\.py$",module)
                                  and module[5:-3] not in exclude]
    
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
"""

def runTests(options):
    """Run tests with options given from command line"""

    #print "\x1b[2J\x1b[80A"
    runner=ForceBalanceTestRunner()
    results=runner.run(**options)
    return results

#### main block ####

options=getOptions()
runTests(options)
    