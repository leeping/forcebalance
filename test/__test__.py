from __init__ import ForceBalanceTestCase
import unittest
import numpy

class TestTest(ForceBalanceTestCase):
    def testFail(self):
        """This test will always fail"""
        self.fail(msg="This message describes the failure")

    def testSuccess(self):
        """This test will always pass"""
        self.assertTrue(True)

    def testError(self):
        """This test will always exit with an error"""
        raise Exception("This is some random unexpected exception")

    def testNumpyArrayEqual(self):
        """These are two equal numpy arrays"""
        self.assertEqual(numpy.array((1,2,3,4.)),numpy.array((1,2,3,4.0001)))

    def testNumpyArrayUnequal(self):
        """These are two unequal numpy arrays"""
        self.assertEqual(numpy.array([(1,2,3,4),(5,6,7,8)]),numpy.array([(1,2,3,6),(5,6,7,8)]), msg="These arrays should not be equal")


if __name__ == '__main__':           
    unittest.main()