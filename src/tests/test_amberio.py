from __future__ import absolute_import
import json
from forcebalance.nifty import *
from forcebalance.amberio import splitComment, parse_amber_namelist

class TestSplitComment(object):
    def test_split_comment(self):
        """ Test splitting of comment field in AMBER namelist """
        printcool("Test splitting of comment field in AMBER namelist")
        def test_input_output(a, b):
            assert splitComment(a, debug=False) == b

        # Exclamation mark outside of quote
        test_input_output(" restraint_wt=0.1, ! Restraint force constant ", " restraint_wt=0.1, ")

        # Exclamation marks enclosed in quote
        test_input_output(" restraintmask='!:WAT,NA&!@H=', ! This is a comment", " restraintmask='!:WAT,NA&!@H=', ")

        # Exclamation marks enclosed in quote that contains some double quotes
        test_input_output(" restraintmask='!:WAT,''NA''&!@H=', ! This is a comment", " restraintmask='!:WAT,''NA''&!@H=', ")

        # One exclamation mark is outside of the quote (i.e. '... NA'); this string is actually incorrect
        test_input_output(" restraintmask='!:WAT,''NA'&!@H=', ! This is a comment", " restraintmask='!:WAT,''NA'&")

        test_input_output("buffermask='', ! buffer region mask", "buffermask='', ")

    def test_parse_amber_mdin(self):
        """ Test if parsing AMBER namelist gives the expected result """
        cwd = os.path.dirname(os.path.realpath(__file__))
        datadir = os.path.join(cwd, 'files', 'test_amber_mdin')
        for i in range(1,8):
            fobj = open(os.path.join(datadir, '%i.mdin.txt' % i))
            fstr = fobj.read()
            fobj.close()
            a1, a2, a3, a4 = json.loads(fstr, object_pairs_hook=OrderedDict)
            b1, b2, b3, b4 = parse_amber_namelist(os.path.join(datadir, '%i.mdin' % i))
            assert a1 == b1
            assert a2 == b2
            assert a3 == b3
            assert a4 == b4

            # missing_pkgs = []
            # for eng in ['TINKER', 'GMX', 'OpenMM']:
            #     if eng not in self.engines:
            #         missing_pkgs.append(eng)
            # if len(missing_pkgs) > 0:
            #     self.skipTest("Missing packages: %s" % ', '.join(missing_pkgs))
            # Data = OrderedDict()
            # for name, eng in self.engines.items():
            #     Data[name] = eng.interaction_energy(fraga=list(range(22)), fragb=list(range(22, 49)))
            # datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
            # if SAVEDATA:
            #     fout = os.path.join(datadir, 'test_interaction_energies.dat')
            #     if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            #     np.savetxt(fout, Data[list(self.engines.keys())[0]])
            # fin = os.path.join(datadir, 'test_interaction_energies.dat')
            # RefData = np.loadtxt(fin)
            # for n1 in self.engines.keys():
            #     self.assertNdArrayEqual(Data[n1], RefData, delta=0.0001,
            #                             msg="%s interaction energies do not match the reference" % n1)

# class TestAbInitio_GMX(ForceBalanceTestCase, TargetTests):
#     def setUp(self):
#         TargetTests.setUp(self)
#         self.options.update({
#                 'penalty_additive': 0.01,
#                 'jobtype': 'NEWTON',
#                 'forcefield': ['water.itp']})

#         self.tgt_opt.update({'type':'ABINITIO_GMX',
#             'name':'cluster-02'})

#         self.ff = forcebalance.forcefield.FF(self.options)

#         self.ffname = self.options['forcefield'][0][:-3]
#         self.filetype = self.options['forcefield'][0][-3:]
#         self.mvals = [.5]*self.ff.np

#         self.logger.debug("Setting up AbInitio_GMX target\n")
#         self.target = forcebalance.gmxio.AbInitio_GMX(self.options, self.tgt_opt, self.ff)
#         self.addCleanup(os.system, 'rm -rf temp')

#     def shortDescription(self):
#         """@override ForceBalanceTestCase.shortDescription()"""
#         return super(TestAbInitio_GMX,self).shortDescription() + " (AbInitio_GMX)"

# if __name__ == '__main__':
#     unittest.main()
