""" @package interaction Interaction energy fitting module.

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, printcool_dictionary, bohrang, warn_press_key
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt, std
from fitsim import FittingSimulation
from molecule import Molecule, format_xyz_coord
import re
import subprocess
from subprocess import PIPE
from finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict
from multiprocessing import Pool
try:
    from simtk.unit import *
except: pass

def parse_interactions(input_file):
    """ Parse through the interactions input file.

    @param[in]  input_file The name of the input file.
    
    """
    # Three dictionaries of return variables.
    Systems = OrderedDict()
    Interactions = OrderedDict()
    Globals = {}
    InterNum = 0
    InterName = "I0"
    InterDict = {}
    SystemName = None
    SystemDict = {}
    
    print "Reading interactions from file: %s" % input_file
    section = "NONE"
    fobj = open(input_file).readlines()
    for ln, line in enumerate(fobj):
        # Anything after "#" is a comment
        line = line.split("#")[0].strip()
        s = line.split()
        # Skip over blank lines
        if len(s) == 0:
            continue
        key = s[0].lower()
        # If line starts with a $, this signifies that we're in a new section.
        if re.match('^\$',line):
            word = re.sub('^\$','',line).upper()
            if word == "END": # End of a section, time to reinitialize variables.
                if section == "GLOBAL": pass
                elif section == "SYSTEM":
                    if SystemName == None:
                        warn_press_key("You need to specify a name for the system on line %i" % ln)
                    elif SystemName in Systems:
                        warn_press_key("A system named %s already exists in Systems" % SystemName)
                    Systems[SystemName] = SystemDict
                    SystemName = None
                    SystemDict = {}
                elif section == "INTERACTION":
                    if InterName in InterDict:
                        warn_press_key("A system named %s already exists in InterDict" % InterName)
                    Interactions[InterName] = InterDict
                    InterNum += 1
                    InterName = "I%i" % InterNum
                    InterDict = {}
                else:
                    warn_press_key("Encountered $end for unsupported section %s on line %i" % (word, ln))
                section = "NONE"
            elif section == "NONE":
                section = word
            else:
                warn_press_key("Encountered section keyword %s when already in section %s" % (word, section))
        elif section == "GLOBAL":
            if key in ['keyfile', 'energy_unit']:
                Globals[key] = s[1]
            elif key == 'optimize':
                if len(s) == 1 or s[1].lower() in ['y','yes','true']:
                    Globals[key] = True
                else:
                    Globals[key] = False
            else:
                warn_press_key("Encountered unsupported key %s in section %s on line %i" % (key, section, ln))
        elif section == "SYSTEM":
            if key == 'name':
                SystemName = s[1]
            elif key == 'geometry':
                SystemDict[key] = s[1]
            elif key == 'rmsd_weight':
                SystemDict[key] = float(s[1])
            elif key == 'optimize':
                if len(s) == 1 or s[1].lower() in ['y','yes','true']:
                    SystemDict[key] = True
                    print "Optimizing system %s" % SystemName
                else:
                    SystemDict[key] = False
            else:
                warn_press_key("Encountered unsupported key %s in section %s on line %i" % (key, section, ln))
        elif section == "INTERACTION":
            if key == 'name':
                InterName = s[1]
            elif key == 'equation':
                InterDict[key] = ' '.join(s[1:])
            elif key == 'energy':
                InterDict[key] = float(s[1])
            elif key == 'weight':
                InterDict[key] = float(s[1])
            else:
                warn_press_key("Encountered unsupported key %s in section %s on line %i" % (key, section, ln))
    return Globals, Systems, Interactions

class Interactions(FittingSimulation):

    """ Improved subclass of FittingSimulation for fitting force fields to interaction energies. """

    def __init__(self,options,sim_opts,forcefield):
        super(Interactions,self).__init__(options,sim_opts,forcefield)
        self.masterfile   = os.path.join(self.simdir,sim_opts['masterfile'])
        self.global_opts, self.sys_opts, self.inter_opts = parse_interactions(self.masterfile)
        # If the global option doesn't exist in the system / interaction, then it is copied over.
        for opt in self.global_opts:
            for sys in self.sys_opts:
                if opt not in self.sys_opts[sys]:
                    self.sys_opts[sys][opt] = self.global_opts[opt]
            for inter in self.inter_opts:
                if opt not in self.inter_opts[inter]:
                    self.inter_opts[inter][opt] = self.global_opts[opt]
        for inter in self.inter_opts:
            self.inter_opts[inter]['reference_physical'] = self.inter_opts[inter]['energy'] * eval(self.inter_opts[inter]['energy_unit'])

        if sim_opts['energy_denom'] == 0.0:
            self.energy_denom = std(array([val['reference_physical'].value_in_unit(kilocalories_per_mole) for val in self.inter_opts.values()])) * kilocalories_per_mole
        else:
            self.energy_denom = sim_opts['energy_denom'] * kilocalories_per_mole

        self.rmsd_denom = sim_opts['rmsd_denom'] * angstrom

        print "The energy denominator is:", self.energy_denom 
        print "The RMSD denominator is:", self.rmsd_denom

    def indicate(self):
        printcool_dictionary(self.PrintDict,title="Interaction Energies (kcal/mol), Objective = % .5e\n %-20s %9s %9s %9s %11s" % 
                             (self.energy_part, "Interaction", "Calc.", "Ref.", "Delta", "Term"))
        if len(self.RMSDDict) > 0:
            printcool_dictionary(self.RMSDDict,title="Geometry Optimized Systems (Angstrom), Objective = %.5e\n %-38s %11s %11s" % (self.rmsd_part, "System", "RMSD", "Term"), keywidth=45)

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':zeros(self.FF.np, dtype=float), 'H':zeros((self.FF.np, self.FF.np), dtype=float)}
        self.PrintDict = OrderedDict()
        self.RMSDDict = OrderedDict()
        #pool = Pool(processes=4)
        def compute(mvals_):
            # This function has automatically assigned variable names from the interaction master file
            # Thus, all variable names in here are protected using an underscore.
            self.FF.make(mvals_)
            VectorD_ = []
            for sys_ in self.sys_opts:
                Energy_, RMSD_ = self.system_driver(sys_)
                #print "Setting %s to" % sys_, Energy_
                exec("%s = Energy_" % sys_) in locals()
                RMSDNrm_ = RMSD_ / self.rmsd_denom
                w_ = self.sys_opts[sys_]['rmsd_weight'] if 'rmsd_weight' in self.sys_opts[sys_] else 1.0
                VectorD_.append(sqrt(w_)*RMSDNrm_)
                if not in_fd() and RMSD_ != 0.0 * angstrom:
                    self.RMSDDict[sys_] = "% 9.3f % 12.5f" % (RMSD_ / angstrom, w_*RMSDNrm_**2)
            VectorE_ = []
            for inter_ in self.inter_opts:
                Calculated_ = eval(self.inter_opts[inter_]['equation'])
                Reference_ = self.inter_opts[inter_]['reference_physical']
                Delta_ = Calculated_ - Reference_
                DeltaNrm_ = Delta_ / self.energy_denom
                w_ = self.inter_opts[inter_]['weight'] if 'weight' in self.inter_opts[inter_] else 1.0
                VectorE_.append(sqrt(w_)*DeltaNrm_)
                if not in_fd():
                    self.PrintDict[inter_] = "% 9.3f % 9.3f % 9.3f % 12.5f" % (Calculated_ / kilocalories_per_mole, 
                                                                                  Reference_ / kilocalories_per_mole, 
                                                                                  Delta_ / kilocalories_per_mole, w_*DeltaNrm_**2)
                # print "%-20s" % inter_, "Calculated:", Calculated_, "Reference:", Reference_, "Delta:", Delta_, "DeltaNrm:", DeltaNrm_
            # The return value is an array of normalized interaction energy differences.
            if not in_fd():
                self.rmsd_part = dot(array(VectorD_),array(VectorD_))
                if len(VectorE_) > 0:
                    self.energy_part = dot(array(VectorE_),array(VectorE_))
                else:
                    self.energy_part = 0.0
            if len(VectorE_) > 0 and len(VectorD_) > 0:
                return array(VectorD_ + VectorE_)
            elif len(VectorD_) > 0:
                return array(VectorD_)
            elif len(VectorE_) > 0:
                return array(VectorE_)
                    
        V = compute(mvals)

        dV = zeros((self.FF.np,len(V)),dtype=float)
        if AGrad or AHess:
            for p in range(self.FF.np):
                dV[p,:], _ = f12d3p(fdwrap(compute, mvals, p), h = self.h, f0 = V)

        Answer['X'] = dot(V,V)
        for p in range(self.FF.np):
            Answer['G'][p] = 2*dot(V, dV[p,:])
            for q in range(self.FF.np):
                Answer['H'][p,q] = 2*dot(dV[p,:], dV[q,:])

        if not in_fd():
            self.objective = Answer['X']
            self.FF.make(mvals)

        return Answer
