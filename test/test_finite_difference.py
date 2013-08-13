from __init__ import ForceBalanceTestCase
import unittest
import numpy
import forcebalance
import re, sys
from math import cos, sin, pi

class TestFiniteDifference(ForceBalanceTestCase):
    def setUp(self):
        # functions is a list of 3-tuples containing (function p, d/dp, d^2/dp^2)
        self.functions = []

        # f(x) = 2x
        self.functions.append((lambda x: 2,
                               lambda x,p: 0,
                               lambda x,p: 0))
        # f(x) = cos(x)
        self.functions.append((lambda x: cos(x[0]),
                               lambda x,p: -sin(x[0])*(p==0),
                               lambda x,p: -cos(x[0])*(p==0)))
        # f(x) = x^2
        self.functions.append((lambda x: x[0]**2,
                               lambda x,p: 2*x[0]*(p==0),
                               lambda x,p: 2*(p==0)))

    def test_fdwrap(self):
        """Check fdwrap properly wraps function"""

        for func in self.functions:
            msg = "\nfdwrap alters function behavior"
            f=forcebalance.finite_difference.fdwrap(func[0], [0]*3, 0)
            self.logger.debug("Checking to make sure fdwrap returns a function\n")
            self.assertEqual(type(f), type(lambda : ''), "\nfdwrap did not return a function")
            # some test values
            for x in range(-10, 11):
                self.assertAlmostEqual(f(x), func[0]([x,0,0]))

    def test_fd_stencils(self):
        """Check finite difference stencils return approximately correct results"""
        func = lambda x: x[0]**2
        fd_stencils = [function for function in dir(forcebalance.finite_difference) if re.match('^f..?d.p$',function)]

        self.logger.debug("Comparing fd stencils against some simple functions\n")
        for func in self.functions:
            for p in range(1):
                for x in range(10):
                    input = [0,0,0]
                    input[p]=x
                    f=forcebalance.finite_difference.fdwrap(func[0], input, p)
                    for stencil in fd_stencils:
                        fd = eval("forcebalance.finite_difference.%s" % stencil)
                        result = fd(f,.0001)
                        if re.match('^f..d.p$', stencil):
                            self.assertAlmostEqual(result[0], func[1](input,p), places=3)
                            self.assertAlmostEqual(result[1], func[2](input,p), places=3)
                        else:
                            self.assertAlmostEqual(result, func[1](input,p), places=3)
            
            

if __name__ == '__main__':           
    unittest.main()
