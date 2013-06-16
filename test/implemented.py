import unittest
import sys, os, re
import forcebalance.implemented
import forcebalance.target
import forcebalance
import abc

class TestImplemented(unittest.TestCase):
    def setUp(self): pass
    
    def tearDown(self): pass

    def test_implemented_targets_derived_from_target(self):
        for key in forcebalance.implemented.Implemented_Targets.iterkeys():
            self.assertTrue(issubclass(forcebalance.implemented.Implemented_Targets[key],forcebalance.target.Target))
    
    def test_no_unlisted_classes_derived_from_Target(self):
        unlisted = []
        forcebalance_modules=[module[:-3] for module in os.listdir(forcebalance.__path__[0])
                     if re.compile(".*\.py$").match(module)
                     and module not in ["__init__.py"]]
        for module in forcebalance_modules:
            m = __import__('forcebalance.' + module)
            objects = dir(eval('m.' + module))
            for object in objects:
                object = eval('m.'+module+'.'+object)
                if type(object) == abc.ABCMeta:
                    #self.assertFalse(issubclass(eval('m.' + module + '.' + object), forcebalance.target.Target))
                    implemented = [i for i in forcebalance.implemented.Implemented_Targets.itervalues()]
                    if object not in implemented and object is not forcebalance.target.Target and object not in unlisted:
                        unlisted.append(object)
        self.assertFalse(len(unlisted), msg="Unlisted Object found")

if __name__ == '__main__':        
    unittest.main()