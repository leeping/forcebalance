import unittest, os, sys, re
import forcebalance
import __init__
import getopt

def getOptions():
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
                exclude=a.split(',')
            if o in ("-i", "--include"):
                options['test_modules']=['test_' + module for module in a.split(',')]
            if o in ("-p", "--pretend"):
                options['pretend']=True
    except getopt.GetoptError as err:
        usage()
        sys.exit()
    
    if not options.has_key('test_modules'):
        options['test_modules'] = [module[:-3] for module in os.listdir('test')
                                  if re.match("^test_.*\.py$",module)
                                  and module[5:-3] not in exclude]
    
    return options

def usage():
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

    #print "\x1b[2J\x1b[80A"
    runner=__init__.ForceBalanceTestRunner()
    results=runner.run(**options)
    print "\n<run=%d errors=%d fail=%d>" % (results.testsRun,len(results.errors),len(results.failures))
    if results.wasSuccessful(): print "All tests passed successfully"
    else: print "Some tests failed or had errors!"
    return results

#### main block ####

options=getOptions()
runTests(options)
    