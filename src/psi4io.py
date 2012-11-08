""" @package psi4io PSI4 force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os
from re import match, sub, split, findall
from nifty import isint, isfloat, _exec, warn_press_key
import numpy as np
from leastsq import LeastSquares, CheckBasis
from basereader import BaseReader
from string import capitalize
from finite_difference import in_fd
from collections import defaultdict
import itertools

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

    def __init__(self,options,sim_opts,forcefield):
        super(THCDF_Psi4,self).__init__(options,sim_opts,forcefield)
        # Parse the input.dat file to figure out the elements and molecules
        MolSection = False
        ElemList = []
        self.Molecules = []
        self.throw_outs = []
        for line in open(os.path.join(self.root,self.simdir,"input.dat")).readlines():
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
        for p in range(self.FF.np):
            Pelem = []
            for pid in self.FF.plist[p].split():
                # Extract the chemical element.
                Pelem.append(pid.split(':')[1].split(',')[0].split('=')[1])
            Pelem = set(Pelem)
            if len(self.Elements.intersection(Pelem)) == 0:
                self.call_derivatives[p] = False
        gbslist = [i for i in self.FF.fnms if os.path.splitext(i)[1] == '.gbs']
        if len(gbslist) != 1:
            warn_press_key("In %s, you should only have exactly one .gbs file in the list of force field files!" % __file__)
        self.GBSfnm = gbslist[0]

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        os.symlink(os.path.join(self.root,self.simdir,"input.dat"),os.path.join(abstempdir,"input.dat"))

    def indicate(self):
        print "\rSim: %-15s" % self.name, 
        print "Molecules = %-20s" % str(self.Molecules),
        print "Objective = %.5e" % self.objective
        return

    def write_nested_destroy(self, fnm, linedestroy):
        ln0 = range(len(open(fnm).readlines()))
        for layer in linedestroy:
            f = open(fnm).readlines()
            o = open('.tmp.gbs','w')
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
            _exec("cp %s %s.bak" % (self.GBSfnm, self.GBSfnm), print_command=False)
            ln0 = self.write_nested_destroy(self.GBSfnm, self.FF.linedestroy_save)
            _exec("psi4 throwout.dat", print_command=False)
            LI = GBS_Reader()
            LI_lines = {}
            ## Read in the commented linindep.gbs file and ensure that these same lines are commented in the new .gbs file
            for line in open('linindep.gbs'):
                LI.feed(line,linindep=True)
                key = '.'.join([str(i) for i in LI.element,LI.amom,LI.basis_number[LI.element],LI.contraction_number])
                if LI.isdata:
                    if key in LI_lines:
                        print "Duplicate key found:"
                        print key
                        print LI_lines[key],
                        print line,
                        warn_press_key("In %s, the LI_lines dictionary should not contain repeated keys!" % __file__)
                    LI_lines[key] = (line, LI.destroy)
            ## Now build a "Frankenstein" .gbs file composed of the original .gbs file but with data from the linindep.gbs file!
            FK = GBS_Reader()
            FK_lines = []
            self.FF.linedestroy_this = []
            self.FF.parmdestroy_this = []
            for ln, line in enumerate(open(self.GBSfnm).readlines()):
                FK.feed(line)
                key = '.'.join([str(i) for i in FK.element,FK.amom,FK.basis_number[FK.element],FK.contraction_number])
                if FK.isdata and key in LI_lines:
                    if LI_lines[key][1]:
                        print "Destroying line %i (originally %i):" % (ln, ln0[ln]), 
                        print line,
                        self.FF.linedestroy_this.append(ln)
                        for p_destroy in [i for i, fld in enumerate(self.FF.pfields) if any([subfld[0] == self.GBSfnm and subfld[1] == ln0[ln] for subfld in fld])]:
                            print "Destroying parameter %i located at line %i (originally %i) with fields given by:" % (p_destroy, ln, ln0[ln]), self.FF.pfields[p_destroy]
                            self.FF.parmdestroy_this.append(p_destroy)
                    FK_lines.append(LI_lines[key][0])
                else:
                    FK_lines.append(line)
            o = open('franken.gbs','w')
            for line in FK_lines:
                print >> o, line,
            o.close()
            _exec("cp %s.bak %s" % (self.GBSfnm, self.GBSfnm), print_command=False)
            
            if len(list(itertools.chain(*(self.FF.linedestroy_save + [self.FF.linedestroy_this])))) > 0:
                print "All lines removed:", self.FF.linedestroy_save + [self.FF.linedestroy_this]
                print "All parms removed:", self.FF.parmdestroy_save + [self.FF.parmdestroy_this]

        self.write_nested_destroy(self.GBSfnm, self.FF.linedestroy_save + [self.FF.linedestroy_this])
        _exec("psi4", print_command=False)
        Ans = np.array([[float(i) for i in line.split()] for line in open("objective.dat").readlines()])
        os.unlink("objective.dat")
        return Ans
