from __init__ import ForceBalanceTestCase
import unittest

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

if __name__ == '__main__':           
    unittest.main()