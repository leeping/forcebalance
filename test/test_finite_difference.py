from __init__ import ForceBalanceTestCase
import unittest
import numpy
import forcebalance
from math import cos, pi

class TestFiniteDifference(ForceBalanceTestCase):
    def test_fdwrap(self):
        """Check fdwrap properly wraps function"""

        msg = "\nfdwrap alters function behavior"
        # wrap simple linear function f(x) == 2x
        func = lambda x: 2*x[0]
        f=forcebalance.finite_difference.fdwrap(func, [0], 0)
        self.assertEqual(type(f), type(lambda : ''), "\nfdwrap did not return a function")
        # some test values
        self.assertEqual(2,f(1),msg)
        self.assertEqual(7,f(3.5),msg)
        self.assertEqual(-3.4,f(-1.7), msg)

        self.assertEqual(2, f(.5)-f(-.5),msg)

        # wrap simple trig function f(x) == cos(x)
        func = lambda x: cos(x[0])
        f=forcebalance.finite_difference.fdwrap(func, [pi/2], 0)
        self.assertEqual(type(f), type(lambda : ''), "\nfdwrap did not return a function")
        # some test values (need to use assertAlmostEqual for floats)
        self.assertAlmostEqual(0, f(0), msg=msg)
        self.assertAlmostEqual(-1, f(pi/2), msg=msg)

        self.assertAlmostEqual(-1, (f(.0005)-f(-.0005))/.001, msg=msg)

if __name__ == '__main__':           
    unittest.main()