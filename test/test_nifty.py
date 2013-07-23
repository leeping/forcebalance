from __init__ import ForceBalanceTestCase
import unittest
import numpy
import os
from forcebalance.nifty import *
from forcebalance.nifty import _exec

class TestNifty(ForceBalanceTestCase):
    def runTest(self):
        """Check utility functions in forcebalance.nifty"""

        ##variable manipulation functions
        self.assertTrue(isint("1"))
        self.assertFalse(isint("1."))
        self.assertTrue(isint("-4"))
        self.assertFalse(isint("-3.14"))

        self.assertTrue(isfloat("1.5"))
        self.assertTrue(isfloat("1"))
        self.assertFalse(isfloat("a"))

        self.assertTrue(isdecimal("1.0"))
        self.assertFalse(isdecimal("1"))

        for result in get_least_squares(([0]),[0]):
            self.assertFalse(result.any())

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

        ##matrix manipulations
        X=flat(X)
        self.assertEqual(X.shape, (6,))
        X=row(X)
        self.assertEqual(X.shape, (1,6))
        X=col(X)
        self.assertEqual(X.shape, (6,1))

        ##_exec
        self.assertEqual(type(_exec("")),list)
        self.assertEqual(_exec("echo test")[0],"test")
        _exec("touch .test")
        self.assertTrue(os.path.isfile(".test"))
        _exec("rm .test")
        self.assertFalse(os.path.isfile(".test"))
        self.assertRaises(Exception, _exec, "exit 255")

if __name__ == '__main__':           
    unittest.main()