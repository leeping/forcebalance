""" @package forcebalance.binding Binding energy fitting module.

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
import numpy as np
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, printcool_dictionary, bohrang, warn_press_key
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
import re
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict
from multiprocessing import Pool

from forcebalance.output import getLogger
logger = getLogger(__name__)

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
    
    logger.info("Reading interactions from file: %s\n" % input_file)
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
                    if SystemName is None:
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
                    logger.info("Optimizing ALL systems by default\n")
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
            elif key == 'select':
                SystemDict[key] = s[1]
            elif key == 'optimize':
                if len(s) == 1 or s[1].lower() in ['y','yes','true']:
                    SystemDict[key] = True
                    logger.info("Optimizing system %s\n" % SystemName)
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

class BindingEnergy(Target):

    """ Improved subclass of Target for fitting force fields to binding energies. """

    def __init__(self,options,tgt_opts,forcefield):
        super(BindingEnergy,self).__init__(options,tgt_opts,forcefield)
        self.set_option(None, None, 'inter_txt', os.path.join(self.tgtdir,tgt_opts['inter_txt']))
        self.global_opts, self.sys_opts, self.inter_opts = parse_interactions(self.inter_txt)
        # If the global option doesn't exist in the system / interaction, then it is copied over.
        for opt in self.global_opts:
            for sys in self.sys_opts:
                if opt not in self.sys_opts[sys]:
                    self.sys_opts[sys][opt] = self.global_opts[opt]
            for inter in self.inter_opts:
                if opt not in self.inter_opts[inter]:
                    self.inter_opts[inter][opt] = self.global_opts[opt]
        for inter in self.inter_opts:
            if 'energy_unit' in self.inter_opts[inter] and self.inter_opts[inter]['energy_unit'].lower() not in ['kilocalorie_per_mole', 'kilocalories_per_mole']:
                logger.error('Usage of physical units is has been removed, please provide all binding energies in kcal/mole\n')
                raise RuntimeError
            self.inter_opts[inter]['reference_physical'] = self.inter_opts[inter]['energy']

        if tgt_opts['energy_denom'] == 0.0:
            self.set_option(None, None, 'energy_denom', val=np.std(np.array([val['reference_physical'] for val in self.inter_opts.values()])))
        else:
            self.set_option(None, None, 'energy_denom', val=tgt_opts['energy_denom'])

        self.set_option(None, None, 'rmsd_denom', val=tgt_opts['rmsd_denom'])

        self.set_option(tgt_opts,'cauchy')
        self.set_option(tgt_opts,'attenuate')

        logger.info("The energy denominator is: %s\n" % str(self.energy_denom)) 
        logger.info("The RMSD denominator is: %s\n" % str(self.rmsd_denom))

        if self.cauchy:
            logger.info("Each contribution to the interaction energy objective function will be scaled by 1.0 / ( denom**2 + reference**2 )\n")
            if self.attenuate:
                logger.error('attenuate and cauchy are mutually exclusive\n')
                raise RuntimeError
        elif self.attenuate:
            logger.info("Repulsive interactions beyond energy_denom will be scaled by 1.0 / ( denom**2 + (reference-denom)**2 )\n")
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(self.OptionDict.items() + options.items())
        del engine_args['name']
        ## Create engine objects.
        self.engines = OrderedDict()
        for sysname,sysopt in self.sys_opts.items():
            M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']))
            if 'select' in sysopt:
                atomselect = np.array(uncommadash(sysopt['select']))
                M = M.atom_select(atomselect)
            if self.FF.rigid_water: M.rigid_water()
            self.engines[sysname] = self.engine_(target=self, mol=M, name=sysname, tinker_key=os.path.join(sysopt['keyfile']), **engine_args)

    def system_driver(self, sysname):
        opts = self.sys_opts[sysname]
        return self.engines[sysname].energy_rmsd(optimize = (opts['optimize'] if 'optimize' in opts else False))

    def indicate(self):
        printcool_dictionary(self.PrintDict,title="Interaction Energies (kcal/mol), Objective = % .5e\n %-20s %9s %9s %9s %11s" % 
                             (self.energy_part, "Interaction", "Calc.", "Ref.", "Delta", "Term"))
        if len(self.RMSDDict) > 0:
            printcool_dictionary(self.RMSDDict,title="Geometry Optimized Systems (Angstrom), Objective = %.5e\n %-38s %11s %11s" % (self.rmsd_part, "System", "RMSD", "Term"), keywidth=45)

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
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
                VectorD_.append(np.sqrt(w_)*RMSDNrm_)
                if not in_fd() and RMSD_ != 0.0:
                    self.RMSDDict[sys_] = "% 9.3f % 12.5f" % (RMSD_, w_*RMSDNrm_**2)
            VectorE_ = []
            for inter_ in self.inter_opts:
                Calculated_ = eval(self.inter_opts[inter_]['equation'])
                Reference_ = self.inter_opts[inter_]['reference_physical']
                Delta_ = Calculated_ - Reference_
                Denom_ = self.energy_denom
                if self.cauchy:
                    Divisor_ = np.sqrt(Denom_**2 + Reference_**2)
                elif self.attenuate:
                    if Reference_ < Denom_:
                        Divisor_ = Denom_
                    else:
                        Divisor_ = np.sqrt(Denom_**2 + (Reference_-Denom_)**2)
                else:
                    Divisor_ = Denom_
                DeltaNrm_ = Delta_ / Divisor_
                w_ = self.inter_opts[inter_]['weight'] if 'weight' in self.inter_opts[inter_] else 1.0
                VectorE_.append(np.sqrt(w_)*DeltaNrm_)
                if not in_fd():
                    self.PrintDict[inter_] = "% 9.3f % 9.3f % 9.3f % 12.5f" % (Calculated_, Reference_, Delta_, w_*DeltaNrm_**2)
                # print "%-20s" % inter_, "Calculated:", Calculated_, "Reference:", Reference_, "Delta:", Delta_, "DeltaNrm:", DeltaNrm_
            # The return value is an array of normalized interaction energy differences.
            if not in_fd():
                self.rmsd_part = np.dot(np.array(VectorD_),np.array(VectorD_))
                if len(VectorE_) > 0:
                    self.energy_part = np.dot(np.array(VectorE_),np.array(VectorE_))
                else:
                    self.energy_part = 0.0
            if len(VectorE_) > 0 and len(VectorD_) > 0:
                return np.array(VectorD_ + VectorE_)
            elif len(VectorD_) > 0:
                return np.array(VectorD_)
            elif len(VectorE_) > 0:
                return np.array(VectorE_)
                    
        V = compute(mvals)

        dV = np.zeros((self.FF.np,len(V)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(compute, mvals, p), h = self.h, f0 = V)

        Answer['X'] = np.dot(V,V)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(V, dV[p,:])
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:])

        if not in_fd():
            self.objective = Answer['X']
            self.FF.make(mvals)

        return Answer
