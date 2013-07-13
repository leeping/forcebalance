from __init__ import ForceBalanceTestCase
import unittest
import numpy
import forcebalance
import re
from math import cos, sin, pi

class TestFiniteDifference(ForceBalanceTestCase):
    def setUp(self):
        # functions is a list of tuples containing (function, first derivative, second derivative)
        self.functions = []

        # f(x) = 2x
        self.functions.append((lambda x: 2*x[0],
                               lambda x: 2,
                               lambda x: 0))
        # f(x) = cos(x)
        self.functions.append((lambda x: cos(x[0]),
                               lambda x: -sin(x[0]),
                               lambda x: cos(x[0])))
        # f(x) = x^2
        self.functions.append((lambda x: x[0]**2,
                               lambda x: 2*x[0],
                               lambda x: 2))

    def test_fdwrap(self):
        """Check fdwrap properly wraps function"""

        for func in self.functions:
            msg = "\nfdwrap alters function behavior"
            # wrap simple linear function f(x) == 2x
            f=forcebalance.finite_difference.fdwrap(func[0], [0]*5, 0)
            self.assertEqual(type(f), type(lambda : ''), "\nfdwrap did not return a function")
            # some test values
            for x in range(-10, 11):
                self.assertAlmostEqual(f(x), func[0]([x,0,0,0,0]))

    def test_fd_stencils(self):
        """Check finite difference stencils return approximately correct results"""
        func = lambda x: x[0]**2
        fd_stencils = [function for function in dir(forcebalance.finite_difference) if re.match('^f..?d.p$',function)]

        for x in range(10):
            f=forcebalance.finite_difference.fdwrap(func, [x], 0)
            for stencil in fd_stencils:
                stencil = eval("forcebalance.finite_difference.%s" % stencil)
                result = stencil(f,.0001)
                if type(result)==tuple:
                    self.assertAlmostEqual(result[0], 2*x, places=3)
                    self.assertAlmostEqual(result[1], 2, places=3)
                else:
                    print stencil
                    print str(result) + "\n\n\n"
                    self.assertAlmostEqual(result, 2*x, places=3)
            
            

if __name__ == '__main__':           
    unittest.main()