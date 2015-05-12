""" @package forcebalance.psi4io PSI4 force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os, sys, glob, shutil
from re import match, sub, split, findall
from forcebalance.nifty import isint, isfloat, _exec, warn_press_key, printcool_dictionary, wopen
import numpy as np
from forcebalance.leastsq import LeastSquares, CheckBasis
from forcebalance import BaseReader
from string import capitalize
from forcebalance.finite_difference import in_fd, f1d2p, f12d3p, fdwrap
from collections import defaultdict, OrderedDict
import itertools
from forcebalance.target import Target
from forcebalance.nifty import queue_up_src_dest, getWorkQueue
from forcebalance.output import getLogger
logger = getLogger(__name__)

##Interaction type -> Parameter Dictionary.
#pdict = {'Exponent':{0:'A', 1:'C'},
#         'BASSP' :{0:'A', 1:'B', 2:'C'}
#         }

class GBS_Reader(BaseReader):
    """Finite state machine for parsing basis set files.
    
    """
    
    def __init__(self,fnm=None):
        super(GBS_Reader,self).__init__(fnm)
        self.element = ''
        self.amom  = ''
        self.last_amom = ''
        self.basis_number  = defaultdict(int)
        self.contraction_number = -1
        self.adict={None:None}
        self.isdata=False
        self.destroy=False
    
    def build_pid(self, pfld):
        if pfld == 0:
            ptype = 'Exponent'
        elif pfld == 1:
            ptype = 'Contraction'
        return ptype+":"+"Elem=%s,AMom=%s,Bas=%i,Con=%i" % (self.element, self.amom, self.basis_number[self.element], self.contraction_number)
        
    def feed(self, line, linindep=False):
        """ Feed in a line.

        @param[in] line     The line of data

        """
        if linindep:
            if match('^ *!',line): 
                self.destroy = True
            else:
                self.destroy = False
            line = sub('^ *!','',line)

        line       = line.split('!')[0].strip()
        s          = line.split()
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^ *!',line): return None, None
        # Now go through all the cases.
        if match('^[A-Za-z][A-Za-z]? +[0-9]$',line):
            # This is supposed to match the element line. For example 'Li 0'
            self.element = capitalize(s[0])
            self.isdata = False
            self.destroy = False
        elif len(s) == 3 and match('[SPDFGH]+',s[0]) and isint(s[1]) and isfloat(s[2]):
            self.amom = s[0]
            if self.amom == self.last_amom:
                self.basis_number[self.element] += 1
            else:
                self.basis_number[self.element] = 0
                self.last_amom = self.amom
            self.contraction_number = -1
            self.isdata = True
            # This is supposed to match a line like 'P   1   1.00'
        elif len(s) == 2 and isfloat(s[0]) and isfloat(s[1]):
            self.contraction_number += 1
            self.isdata = True
        else:
            self.isdata = False
            self.destroy = False

class THCDF_Psi4(LeastSquares):

    def __init__(self,options,tgt_opts,forcefield):
        super(THCDF_Psi4,self).__init__(options,tgt_opts,forcefield)

        # Parse the input.dat file to figure out the elements and molecules
        MolSection = False
        ElemList = []
        self.Molecules = []
        self.throw_outs = []
        for line in open(os.path.join(self.root,self.tgtdir,"input.dat")).readlines():
            line = line.strip()
            s = line.split()
            if len(s) >= 3 and s[0].lower() == 'molecule' and s[2] == '{':
                MolSection = True
                self.Molecules.append(s[1])
            elif len(s) >= 1 and s[0] == '}':
                MolSection = False
            elif MolSection and len(s) >= 4 and match("^[A-Za-z]+$",s[0]) and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
                ElemList.append(capitalize(s[0]))
        self.Elements = set(ElemList)
        xgrad = []
        for p in self.pgrad:
            Pelem = []
            for pid in self.FF.plist[p].split():
                # Extract the chemical element.
                Pelem.append(pid.split(':')[1].split(',')[0].split('=')[1])
            Pelem = set(Pelem)
            if len(self.Elements.intersection(Pelem)) == 0:
                xgrad.append(p)
        for p in xgrad:
            self.pgrad.remove(p)
        ## Psi4 basis set file
        gbslist = [i for i in self.FF.fnms if os.path.splitext(i)[1] == '.gbs']
        if len(gbslist) != 1:
            warn_press_key("In %s, you should only have exactly one .gbs file in the list of force field files!" % __file__)
        self.GBSfnm = gbslist[0]
        ## Psi4 input file for calculation of linear dependencies
        ## This is actually a file in 'forcefield' until we can figure out a better system
        if CheckBasis():
            datlist = [i for i in self.FF.fnms if os.path.splitext(i)[1] == '.dat']
            if len(datlist) != 1:
                warn_press_key("In %s, you should only have exactly one .dat file in the list of force field files!" % __file__)
            self.DATfnm = datlist[0]
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,tgt_opts)

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        o = wopen(os.path.join(abstempdir,"input.dat"))
        for line in open(os.path.join(self.root,self.tgtdir,"input.dat")).readlines():
            s = line.split("#")[0].split()
            if len(s) == 3 and s[0].lower() == 'basis' and s[1].lower() == 'file':
                print >> o, "basis file %s" % self.GBSfnm
            else:
                print >> o, line,
        o.close()

    def indicate(self):
        MAD = np.mean(np.abs(self.D))
        logger.info("\rTarget: %-15s Molecules = %-30s %s" % (self.name, str(self.Molecules), "Mean (Max) Error: %8.4f%% (%8.4f%%) Energies: DF %+.3e MP2 %+.3e Delta % .3e Objective = %.5e" % (100*MAD / self.MAQ, 100*np.max(np.abs(self.D)) / self.MAQ, self.DF_Energy, self.MP2_Energy, self.DF_Energy - self.MP2_Energy, self.objective)))
        return

    def write_nested_destroy(self, fnm, linedestroy):
        ln0 = range(len(open(fnm).readlines()))
        for layer in linedestroy:
            f = open(fnm).readlines()
            o = wopen('.tmp.gbs')
            newln = []
            for ln, line in enumerate(f):
                if ln not in layer:
                    print >> o, line,
                    newln.append(ln0[ln])
            ln0 = newln[:]
            _exec("mv .tmp.gbs %s" % fnm, print_command=False)
            o.close()
        return ln0

    def driver(self):
        ## Actually run PSI4.
        if not in_fd() and CheckBasis():
            logger.info("Now checking for linear dependencies.\n")
            _exec("cp %s %s.bak" % (self.GBSfnm, self.GBSfnm), print_command=False)
            ln0 = self.write_nested_destroy(self.GBSfnm, self.FF.linedestroy_save)
            o = wopen(".lindep.dat")
            for line in open(self.DATfnm).readlines():
                s = line.split("#")[0].split()
                if len(s) == 3 and s[0].lower() == 'basis' and s[1].lower() == 'file':
                    print >> o, "basis file %s" % self.GBSfnm
                else:
                    print >> o, line,
            o.close()
            _exec("mv .lindep.dat %s" % self.DATfnm, print_command=False)
            _exec("psi4 %s" % self.DATfnm, print_command=False)
            LI = GBS_Reader()
            LI_lines = {}
            ## Read in the commented linindep.gbs file and ensure that these same lines are commented in the new .gbs file
            for line in open('linindep.gbs'):
                LI.feed(line,linindep=True)
                key = '.'.join([str(i) for i in LI.element,LI.amom,LI.basis_number[LI.element],LI.contraction_number])
                if LI.isdata:
                    if key in LI_lines:
                        logger.info("Duplicate key found:\n")
                        logger.info("%s\n" % key)
                        logger.info(str(LI_lines[key]))
                        logger.info(line)
                        warn_press_key("In %s, the LI_lines dictionary should not contain repeated keys!" % __file__)
                    LI_lines[key] = (line, LI.destroy)
            ## Now build a "Frankenstein" .gbs file composed of the original .gbs file but with data from the linindep.gbs file!
            FK = GBS_Reader()
            FK_lines = []
            self.FF.linedestroy_this = []
            self.FF.prmdestroy_this = []
            for ln, line in enumerate(open(self.GBSfnm).readlines()):
                FK.feed(line)
                key = '.'.join([str(i) for i in FK.element,FK.amom,FK.basis_number[FK.element],FK.contraction_number])
                if FK.isdata and key in LI_lines:
                    if LI_lines[key][1]:
                        logger.info("Destroying line %i (originally %i): " % (ln, ln0[ln]))
                        logger.info(line)
                        self.FF.linedestroy_this.append(ln)
                        for p_destroy in [i for i, fld in enumerate(self.FF.pfields) if any([subfld[0] == self.GBSfnm and subfld[1] == ln0[ln] for subfld in fld])]:
                            logger.info("Destroying parameter %i located at line %i (originally %i) with fields given by: %s" % (p_destroy, ln, ln0[ln], str(self.FF.pfields[p_destroy])))
                            self.FF.prmdestroy_this.append(p_destroy)
                    FK_lines.append(LI_lines[key][0])
                else:
                    FK_lines.append(line)
            o = wopen('franken.gbs')
            for line in FK_lines:
                print >> o, line,
            o.close()
            _exec("cp %s.bak %s" % (self.GBSfnm, self.GBSfnm), print_command=False)
            
            if len(list(itertools.chain(*(self.FF.linedestroy_save + [self.FF.linedestroy_this])))) > 0:
                logger.info("All lines removed: " + self.FF.linedestroy_save + [self.FF.linedestroy_this] + '\n')
                logger.info("All prms removed: " + self.FF.prmdestroy_save + [self.FF.prmdestroy_this] + '\n')

        self.write_nested_destroy(self.GBSfnm, self.FF.linedestroy_save + [self.FF.linedestroy_this])
        _exec("psi4", print_command=False, outfnm="psi4.stdout")
        if not in_fd():
            for line in open('psi4.stdout').readlines():
                if "MP2 Energy:" in line:
                    self.MP2_Energy = float(line.split()[-1])
                elif "DF Energy:" in line:
                    self.DF_Energy = float(line.split()[-1])
        Ans = np.array([[float(i) for i in line.split()] for line in open("objective.dat").readlines()])
        os.unlink("objective.dat")
        return Ans

class Grid_Reader(BaseReader):
    """Finite state machine for parsing DVR grid files.
    
    """
    
    def __init__(self,fnm=None):
        super(Grid_Reader,self).__init__(fnm)
        self.element = ''
        self.point = 0
        self.radii = OrderedDict()
    
    def build_pid(self, pfld):
        if pfld == 1:
            ptype = 'Position'
        elif pfld == 2:
            ptype = 'Weight'
        else:
            ptype = 'None'
        return ptype+":"+"Elem=%s,Point=%i" % (self.element, self.point)
        
    def feed(self, line, linindep=False):
        """ Feed in a line.

        @param[in] line     The line of data

        """
        line       = line.split('!')[0].strip()
        s          = line.split()
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^ *!',line): return None, None
        # Now go through all the cases.
        if match('^[A-Za-z][A-Za-z]? +[0-9]$',line):
            # This is supposed to match the element line. For example 'Li 0'
            self.element = capitalize(s[0])
            self.radii[self.element] = float(s[1])
            self.isdata = False
            self.point = 0
        elif len(s) >= 2 and isint(s[0]) and isfloat(s[1]):
            self.point += 1
            self.isdata = True
        else:
            self.isdata = False

class RDVR3_Psi4(Target):

    """ Subclass of Target for R-DVR3 grid fitting.
    Main features:
    - Multiple molecules are treated as a single target.
    - R-DVR3 can only print out the objective function, it cannot print out the residual vector.
    - We should be smart enough to mask derivatives.
    """
    
    def __init__(self,options,tgt_opts,forcefield):
        super(RDVR3_Psi4,self).__init__(options,tgt_opts,forcefield)
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Which parameters are differentiated?
        self.objfiles = OrderedDict()
        self.objvals = OrderedDict()
        self.elements = OrderedDict()
        self.molecules = OrderedDict()
        self.callderivs = OrderedDict()
        self.factor = 1e6
        self.bidirect = False
        for d in sorted(os.listdir(self.tgtdir)):
            if os.path.isdir(os.path.join(self.tgtdir,d)) and os.path.exists(os.path.join(self.tgtdir,d,'objective.dat')):
                self.callderivs[d] = [True for i in range(forcefield.np)]
                self.objfiles[d] = open(os.path.join(self.tgtdir,d,'objective.dat')).readlines()
                ElemList = []
                Molecules = []
                for line in self.objfiles[d]:
                    line = line.strip()
                    s = line.split()
                    if len(s) >= 3 and s[0].lower() == 'molecule' and s[2] == '{':
                        MolSection = True
                        Molecules.append(s[1])
                    elif len(s) >= 1 and s[0] == '}':
                        MolSection = False
                    elif MolSection and len(s) >= 4 and match("^[A-Za-z]+$",s[0]) and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
                        ElemList.append(capitalize(s[0]))
                self.elements[d] = set(ElemList)
                self.molecules[d] = Molecules
                for p in range(self.FF.np):
                    Pelem = []
                    for pid in self.FF.plist[p].split():
                        # Extract the chemical element.
                        Pelem.append(pid.split(':')[1].split(',')[0].split('=')[1])
                    Pelem = set(Pelem)
                    if len(self.elements[d].intersection(Pelem)) == 0:
                        self.callderivs[d][p] = False
        
    def indicate(self):
        PrintDict = OrderedDict()
        for d in self.objvals:
            PrintDict[d] = "%15.9f" % self.objvals[d]
        printcool_dictionary(PrintDict,title="Target: %s\nR-DVR Objective Function, Total = %15.9f\n %-10s %15s" % 
                             (self.name, self.objective, "Molecule", "Objective"),keywidth=15)

        return

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        # This routine is called by Objective.stage() will run before "get".
        # It submits the jobs to the Work Queue and the stage() function will wait for jobs to complete.
        #
        self.tdir = os.getcwd()
        wq = getWorkQueue()
        if wq is None:
            return

        def submit_psi(this_apath, dname, these_mvals):
            """ Create a grid file and a psi4 input file in the absolute path and submit it to the work queue. """
            cwd = os.getcwd()
            if not os.path.exists(this_apath) : os.makedirs(this_apath)
            os.chdir(this_apath)
            self.FF.make(these_mvals)
            o = wopen('objective.dat')
            for line in self.objfiles[d]:
                s = line.split()
                if len(s) > 2 and s[0] == 'path' and s[1] == '=':
                    print >> o, "path = '%s'" % os.getcwd()
                elif len(s) > 2 and s[0] == 'set' and s[1] == 'objective_path':
                    print >> o, "opath = '%s'" % os.getcwd()
                    print >> o, "set objective_path $opath"
                else:
                    print >> o, line,
            o.close()
            os.system("rm -f objective.out")
            if wq is None:
                logger.info("There is no Work Queue!!!\n")
                sys.exit()
            else:
                input_files = [(os.path.join(this_apath, i), i) for i in glob.glob("*")]
                input_files += [(os.path.join(self.root, self.tgtdir, dname, "build.dat"), "build.dat")]
                input_files += [(os.path.join(os.path.split(__file__)[0],"data","run_psi_rdvr3_objective.sh"), "run_psi_rdvr3_objective.sh")]
                logger.info("\r")
                queue_up_src_dest(wq,"sh run_psi_rdvr3_objective.sh -c %s &> run_psi_rdvr3_objective.log" % os.path.join(self.root, self.tgtdir, dname),
                                  input_files=input_files,
                                  output_files=[(os.path.join(this_apath, i),i) for i in ["run_psi_rdvr3_objective.log", "output.dat"]], verbose=False)
            os.chdir(cwd)

        for d in self.objfiles:
            logger.info("\rNow working on" + str(d) + 50*' ' + '\r')
            odir = os.path.join(os.getcwd(),d)
            #if os.path.exists(odir):
            #    shutil.rmtree(odir)
            if not os.path.exists(odir): os.makedirs(odir)
            apath = os.path.join(odir, "current")
            submit_psi(apath, d, mvals)
            for p in range(self.FF.np):
                def subjob(mvals_,h):
                    apath = os.path.join(odir, str(p), str(h))
                    submit_psi(apath, d, mvals_)
                    #logger.info("Will set up a job for %s, parameter %i\n" % (d, p))
                    return 0.0
                if self.callderivs[d][p]:
                    if AHess:
                        f12d3p(fdwrap(subjob, mvals, p, h=self.h), h = self.h, f0 = 0.0)
                    elif AGrad:
                        if self.bidirect:
                            f12d3p(fdwrap(subjob, mvals, p, h=self.h), h = self.h, f0 = 0.0)
                        else:
                            f1d2p(fdwrap(subjob, mvals, p, h=self.h), h = self.h, f0 = 0.0)

    def driver(self, mvals, d):
        ## Create the force field file.
        pvals = self.FF.make(mvals)
        ## Actually run PSI4.
        odir = os.path.join(os.getcwd(),d)
        #if os.path.exists(odir):
        #    shutil.rmtree(odir)
        if not os.path.exists(odir): os.makedirs(odir)
        os.chdir(odir)
        o = wopen('objective.dat')
        for line in self.objfiles[d]:
            s = line.split()
            if len(s) > 2 and s[0] == 'path' and s[1] == '=':
                print >> o, "path = '%s'" % self.tdir
            elif len(s) > 2 and s[0] == 'set' and s[1] == 'objective_path':
                print >> o, "opath = '%s'" % os.getcwd()
                print >> o, "set objective_path $opath"
            else:
                print >> o, line,
        o.close()
        os.system("rm -f objective.out")
        _exec("psi4 objective.dat", print_command=False)
        answer = float(open('objective.out').readlines()[0].split()[1])*self.factor
        os.chdir('..')
        return answer

    def get(self, mvals, AGrad=False, AHess=False):
	"""
        LPW 04-17-2013
        
        This subroutine builds the objective function from Psi4.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        """
        Answer = {}
        Fac = 1000000
        n = len(mvals)
        X = 0.0
        G = np.zeros(n)
        H = np.zeros((n,n))
        pvals = self.FF.make(mvals)
        self.tdir = os.getcwd()
        self.objd = OrderedDict()
        self.gradd = OrderedDict()
        self.hdiagd = OrderedDict()
        wq = getWorkQueue()

        def fdwrap2(func,mvals0,pidx,qidx,key=None,**kwargs):
            def func2(arg1,arg2):
                mvals = list(mvals0)
                mvals[pidx] += arg1
                mvals[qidx] += arg2
                logger.info("\rfdwrap2:" + func.__name__ + "[%i] = % .1e , [%i] = % .1e" % (pidx, arg1, qidx, arg2) + ' '*50)
                if key is not None:
                    return func(mvals,**kwargs)[key]
                else:
                    return func(mvals,**kwargs)
            return func2

        def f2d5p(f, h):
            fpp, fpm, fmp, fmm = [f(i*h,j*h) for i,j in [(1,1),(1,-1),(-1,1),(-1,-1)]]
            fpp = (fpp-fpm-fmp+fmm)/(4*h*h)
            return fpp

        def f2d4p(f, h, f0 = None):
            if f0 is None:
                fpp, fp0, f0p, f0 = [f(i*h,j*h) for i,j in [(1,1),(1,0),(0,1),(0,0)]]
            else:
                fpp, fp0, f0p = [f(i*h,j*h) for i,j in [(1,1),(1,0),(0,1)]]
            fpp = (fpp-fp0-f0p+f0)/(h*h)
            return fpp

        for d in self.objfiles:
            logger.info("\rNow working on" + str(d) + 50*' ' + '\r')
            if wq is None:
                x = self.driver(mvals, d)
            grad  = np.zeros(n)
            hdiag = np.zeros(n)
            hess  = np.zeros((n,n))
            apath = os.path.join(self.tdir, d, "current")
            x = float(open(os.path.join(apath,'objective.out')).readlines()[0].split()[1])*self.factor
            for p in range(self.FF.np):
                if self.callderivs[d][p]:
                    def reader(mvals_,h):
                        apath = os.path.join(self.tdir, d, str(p), str(h))
                        answer = float(open(os.path.join(apath,'objective.out')).readlines()[0].split()[1])*self.factor
                        return answer
                    if AHess:
                        if wq is not None:
                            apath = os.path.join(self.tdir, d, "current")
                            x = float(open(os.path.join(apath,'objective.out')).readlines()[0].split()[1])*self.factor
                            grad[p], hdiag[p] = f12d3p(fdwrap(reader, mvals, p, h=self.h), h = self.h, f0 = x)
                        else:
                            grad[p], hdiag[p] = f12d3p(fdwrap(self.driver, mvals, p, d=d), h = self.h, f0 = x)
                        hess[p,p] = hdiag[p]
                    elif AGrad:
                        if self.bidirect:
                            if wq is not None:
                                apath = os.path.join(self.tdir, d, "current")
                                x = float(open(os.path.join(apath,'objective.out')).readlines()[0].split()[1])*self.factor
                                grad[p], _ = f12d3p(fdwrap(reader, mvals, p, h=self.h), h = self.h, f0 = x)
                            else:
                                grad[p], _ = f12d3p(fdwrap(self.driver, mvals, p, d=d), h = self.h, f0 = x)
                        else:
                            if wq is not None:
                                # Since the calculations are submitted as 3-point finite difference, this part of the code
                                # actually only reads from half of the completed calculations.
                                grad[p] = f1d2p(fdwrap(reader, mvals, p, h=self.h), h = self.h, f0 = x)
                            else:
                                grad[p] = f1d2p(fdwrap(self.driver, mvals, p, d=d), h = self.h, f0 = x)
                            
            self.objd[d] = x
            self.gradd[d] = grad
            self.hdiagd[d] = hdiag
            X += x
            G += grad
            #H += np.diag(hdiag)
            H += hess
        if not in_fd():
            self.objective = X
            self.objvals = self.objd
        # print self.objd
        # print self.gradd
        # print self.hdiagd
                    
        if float('Inf') in pvals:
            return {'X' : 1e10, 'G' : G, 'H' : H}
        return {'X' : X, 'G' : G, 'H' : H}
        
