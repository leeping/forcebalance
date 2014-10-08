from __init__ import ForceBalanceTestCase
import unittest
import numpy
import os, re
import subprocess
import forcebalance
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from collections import defaultdict

try:
    import work_queue
except ImportError:
    work_queue = None

class TestNifty(ForceBalanceTestCase):
    def setUp(self):
        # skip work_queue tests if work_queue could not be imported
        if re.match(".*work_queue.*", self.id().split('.')[-1]) and not work_queue:
            self.skipTest("work_queue module not installed")
        
    def test_nifty_functions(self):
        """Check utility functions in forcebalance.nifty"""

        ##variable manipulation functions
        self.logger.debug("Checking nifty.isint()\n")
        self.assertTrue(isint("1"))
        self.assertFalse(isint("1."))
        self.assertTrue(isint("-4"))
        self.assertFalse(isint("-3.14"))

        self.logger.debug("Checking nifty.isfloat()\n")
        self.assertTrue(isfloat("1.5"))
        self.assertTrue(isfloat("1"))
        self.assertFalse(isfloat("a"))

        self.logger.debug("Checking nifty.isdecimal()\n")
        self.assertTrue(isdecimal("1.0"))
        self.assertFalse(isdecimal("1"))

        for result in get_least_squares(([0]),[0]):
            self.assertFalse(result.any())

        self.logger.debug("Verifying nifty.get_least_squares() results for some trivial cases\n")
        ##least squares function tests
        #   trivial fully determined
        X=((1,3,-2),(3,5,6),(2,4,3))
        Y=(5,7,8)
        result = get_least_squares(X,Y)[0]
        self.assertAlmostEqual(result[0], -15)
        self.assertAlmostEqual(result[1], 8)
        self.assertAlmostEqual(result[2], 2)

        #   inconsistent system
        X=((1,),(1,))
        Y=(0,1)
        result = get_least_squares(X,Y)[0]
        self.assertAlmostEqual(result[0], .5)

        #   overdetermined system
        X=((2,0),(-1,1),(0,2))
        Y=(1,0,-1)
        result = get_least_squares(X,Y)[0]
        self.assertAlmostEqual(result[0], 1./3.)
        self.assertAlmostEqual(result[1], -1./3.)

        self.logger.debug("Verify nifty matrix manipulations perform as expected\n")
        ##matrix manipulations
        X=flat(X)
        self.assertEqual(X.shape, (6,))
        X=row(X)
        self.assertEqual(X.shape, (1,6))
        X=col(X)
        self.assertEqual(X.shape, (6,1))

        self.logger.debug("Running some test processes using nifty._exec()\n")
        ##_exec
        self.assertEqual(type(_exec("")),list)
        self.assertEqual(_exec("echo test")[0],"test")
        _exec("touch .test")
        self.assertTrue(os.path.isfile(".test"))
        _exec("rm .test")
        self.assertFalse(os.path.isfile(".test"))
        self.assertRaises(Exception, _exec, "exit 255")
    
    def test_work_queue_functions(self):
        """Check work_queue functions behave as expected"""
        
        # Work Queue will no longer be initialized to None
        self.logger.debug("\nChecking Work Queue is initialized to None...\n")
        self.assertEqual(forcebalance.nifty.WORK_QUEUE, None,
            msg="\nUnexpected initialization of forcebalance.nifty.WORK_QUEUE to %s" % str(forcebalance.nifty.WORK_QUEUE))
        self.logger.info("\n")

        createWorkQueue(9191, debug=False)
        self.logger.debug("Created work queue, verifying...\n")
        self.assertEqual(type(forcebalance.nifty.WORK_QUEUE), work_queue.WorkQueue,
            msg="\nExpected forcebalance.nifty.WORK_QUEUE to be a WorkQueue object, but got a %s instead" % str(type(forcebalance.nifty.WORK_QUEUE)))
            
        self.logger.debug("Checking that getWorkQueue() returns valid WorkQueue object...\n")
        wq = getWorkQueue()
        self.assertEqual(type(wq), work_queue.WorkQueue,
            msg="\nExpected getWorkQueue() to return a WorkQueue object, but got %s instead" % str(type(wq)))
        
        worker_program = which('work_queue_worker')
        if worker_program != '':
            self.logger.debug("Submitting test job 'echo work queue test > test.job'\n")
            queue_up(wq, "echo work queue test > test.job", [], ["test.job"], tgt=None, verbose=False)
            self.logger.debug("Verifying that work queue has a task waiting\n")
            self.assertEqual(wq.stats.tasks_waiting, 1, msg = "\nExpected queue to have a task waiting")
            
            self.logger.debug("Creating work_queue_worker process... ")
            worker = subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)], stdout=subprocess.PIPE)
            self.addCleanup(worker.terminate)
            self.logger.debug("Done\nTrying to get task from work queue\n")
            
            self.logger.debug("Calling wq_wait1 to fetch task\n")
            wq_wait1(wq, wait_time=5)
            self.logger.debug("wq_wait1(wq, wait_time=5) finished\n")
            self.logger.debug("Checking that wq.stats.total_tasks_complete == 1\n")
            self.assertEqual(wq.stats.total_tasks_complete, 1, msg = "\nExpected queue to have a task completed")
        else:
            self.logger.debug("work_queue_worker is not in the PATH.")
        
        # Destroy the Work Queue object so it doesn't interfere with the rest of the tests.
        destroyWorkQueue()
        
if __name__ == '__main__':           
    unittest.main()
