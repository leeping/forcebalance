""" @package forcebalance.hydration Hydration free energy fitting module

@author Lee-Ping Wang
@date 09/2014
"""

import os
import shutil
import numpy as np
from copy import deepcopy
from forcebalance.target import Target
from forcebalance.molecule import Molecule
from re import match, sub
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import defaultdict, OrderedDict
from forcebalance.nifty import getWorkQueue, queue_up, LinkFile, printcool, link_dir_contents, lp_dump, lp_load, _exec, kb, col, flat, uncommadash, statisticalInefficiency, isfloat

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Hydration(Target):

    """ Subclass of Target for fitting force fields to hydration free energies."""
    
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Hydration,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts,'hfedata_txt','datafile')
        self.set_option(tgt_opts,'hfemode')
        # Normalize the weights for molecules in this target
        self.set_option(tgt_opts,'normalize')
        # Energy denominator for evaluating this target
        self.set_option(tgt_opts,'energy_denom','denom')
        # Number of time steps in the liquid "equilibration" run
        self.set_option(tgt_opts,'liquid_eq_steps',forceprint=True)
        # Number of time steps in the liquid "production" run
        self.set_option(tgt_opts,'liquid_md_steps',forceprint=True)
        # Time step length (in fs) for the liquid production run
        self.set_option(tgt_opts,'liquid_timestep',forceprint=True)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'liquid_interval',forceprint=True)
        # Number of time steps in the gas "equilibration" run
        self.set_option(tgt_opts,'gas_eq_steps',forceprint=True)
        # Number of time steps in the gas "production" run
        self.set_option(tgt_opts,'gas_md_steps',forceprint=True)
        # Time step length (in fs) for the gas production run
        self.set_option(tgt_opts,'gas_timestep',forceprint=True)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'gas_interval',forceprint=True)
        # Single temperature for calculating hydration free energies
        self.set_option(tgt_opts,'hfe_temperature',forceprint=True)
        # Single pressure for calculating hydration free energies
        self.set_option(tgt_opts,'hfe_pressure',forceprint=True)
        # Whether to save trajectories (0 = never, 1 = delete after good step, 2 = keep all)
        self.set_option(tgt_opts,'save_traj')
        # Optimize only a subset of the 
        self.set_option(tgt_opts,'subset')
        # List of trajectory files that may be deleted if self.save_traj == 1.
        self.last_traj = []
        # Extra files to be copied back at the end of a run.
        self.extra_output = []
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The vdata.txt file that contains the hydrations.
        self.datafile = os.path.join(self.tgtdir,self.datafile)
        ## Scripts to be copied from the ForceBalance installation directory.
        self.scripts += ['md_ism_hfe.py']
        ## Read in the reference data
        self.read_reference_data()
        ## Set engname in OptionDict, which gets printed to disk.
        ## This is far from an ideal solution...
        self.OptionDict['engname'] = self.engname
        ## Copy target options into engine options.
        self.engine_opts = OrderedDict(self.OptionDict.items() + options.items())
        del self.engine_opts['name']
        ## Carry out necessary operations for specific modes.
        if self.hfemode.lower() in ['sp', 'single']:
            logger.info("Hydration free energies from geometry optimization and single point energy evaluation\n")
            self.build_engines()
        elif self.hfemode.lower() == 'ti2':
            logger.info("Hydration free energies from thermodynamic integration (linear response approximation)\n")
        elif self.hfemode.lower() == 'exp_gas':
            logger.info("Hydration free energies from exponential reweighting from the gas to the aqueous phase\n")
        elif self.hfemode.lower() == 'exp_liq':
            logger.info("Hydration free energies from exponential reweighting from the aqueous to the gas phase\n")
        elif self.hfemode.lower() == 'exp_both':
            logger.info("Hydration free energies from exponential reweighting from the aqueous to the gas phase and vice versa, taking the average\n")
        else:
            logger.error("Please choose hfemode from single, sp, ti2, exp_gas, exp_liq, or exp_both\n")
            raise RuntimeError

        if self.FF.rigid_water:
            logger.error('This class cannot be used with rigid water molecules.\n')
            raise RuntimeError

    def read_reference_data(self):
        """ Read the reference hydrational data from a file. """
        # Read the HFE data file.  This is a very simple file format:
        self.IDs = []
        self.expval = OrderedDict()
        self.experr = OrderedDict()
        self.nicknames = OrderedDict()
        # We don't need every single line to be the same.  This
        # indicates whether *any* molecule has a nickname for printing
        # out the nickname column.
        self.have_nicks = False
        # Again for experimental errors.  Note that we're NOT using them in the optimization at this time.
        self.have_experr = False
        for line in open(self.datafile).readlines():
            s = line.expandtabs().strip().split('#')[0].split()
            if len(s) == 0: continue
            ID = s[0]
            self.IDs.append(ID)
            # Dynamic field number for the experimental data.
            nxt = 1
            # If the next field is a string, then it's the "nickname"
            if not isfloat(s[1]):
                self.have_nicks = True
                self.nicknames[ID] = s[1]
                nxt += 1
            else:
                # We don't need nicknames on every single line.
                self.nicknames[ID] = ID
            # Read the experimental value.
            self.expval[ID] = float(s[nxt])
            # Read the experimental error bar, or use a default value of 0.6 (from Mobley).
            if len(s) > (nxt+1):
                self.have_experr = True
                self.experr[ID] = float(s[nxt+1])
            else:
                self.experr[ID] = 0.6
        
        self.molecules = OrderedDict([(i, os.path.abspath(os.path.join(self.root, self.tgtdir, 'molecules', i+self.crdsfx))) for i in self.IDs])
        for fnm, path in self.molecules.items():
            if not os.path.isfile(path):
                logger.error('Coordinate file %s does not exist!\nMake sure coordinate files are in the right place\n' % path)
                raise RuntimeError
        if self.subset is not None:
            subset = uncommadash(self.subset)
            self.whfe = np.array([1 if i in subset else 0 for i in range(len(self.IDs))])
        else:
            self.whfe = np.ones(len(self.IDs))

    def run_simulation(self, label, liq, AGrad=True):
        """ 
        Submit a simulation to the Work Queue or run it locally.

        Inputs:
        label = The name of the molecule (and hopefully the folder name that you're running in)
        liq = True/false flag indicating whether to run in liquid or gas phase
        """
        wq = getWorkQueue()

        # Create a dictionary of MD options that the script will read.
        md_opts = OrderedDict()
        md_opts['temperature'] = self.hfe_temperature
        md_opts['pressure'] = self.hfe_pressure
        md_opts['minimize'] = True
        if liq: 
            sdnm = 'liq'
            md_opts['nequil'] = self.liquid_eq_steps
            md_opts['nsteps'] = self.liquid_md_steps
            md_opts['timestep'] = self.liquid_timestep
            md_opts['sample'] = self.liquid_interval
        else: 
            sdnm = 'gas'
            md_opts['nequil'] = self.gas_eq_steps
            md_opts['nsteps'] = self.gas_md_steps
            md_opts['timestep'] = self.gas_timestep
            md_opts['sample'] = self.gas_interval

        eng_opts = deepcopy(self.engine_opts)
        # Enforce implicit solvent in the liquid simulation.
        # We need to be more careful with this when running explicit solvent. 
        eng_opts['implicit_solvent'] = liq
        eng_opts['coords'] = os.path.basename(self.molecules[label])

        if not os.path.exists(sdnm):
            os.makedirs(sdnm)
        os.chdir(sdnm)
        if not os.path.exists('md_result.p'):
            # Link in a bunch of files... what were these again?
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            # Link in the scripts required to run the simulation
            for f in self.scripts:
                LinkFile(os.path.join(os.path.split(__file__)[0],"data",f),os.path.join(os.getcwd(),f))
            # Link in the coordinate file.
            LinkFile(self.molecules[label], './%s' % os.path.basename(self.molecules[label]))
            # Store names of previous trajectory files.
            self.last_traj += [os.path.join(os.getcwd(), i) for i in self.extra_output]
            # Write target, engine and simulation options to disk.
            lp_dump((self.OptionDict, eng_opts, md_opts), 'simulation.p')
            # Execute the script for running molecular dynamics.
            cmdstr = '%s python md_ism_hfe.py %s' % (self.prefix, "-g" if AGrad else "")
            if wq is None:
                logger.info("Running condensed phase simulation locally.\n")
                logger.info("You may tail -f %s/npt.out in another terminal window\n" % os.getcwd())
                _exec(cmdstr, copy_stderr=True, outfnm='md.out')
            else:
                queue_up(wq, command = cmdstr+' &> md.out', tag='%s:%s/%s' % (self.name, label, "liq" if liq else "gas"),
                         input_files = self.scripts + ['simulation.p', 'forcefield.p', os.path.basename(self.molecules[label])],
                         output_files = ['md_result.p', 'md.out'] + self.extra_output, tgt=self, verbose=False, print_time=3600)
        os.chdir('..')

    def submit_liq_gas(self, mvals, AGrad=True):
        """
        Set up and submit/run sampling simulations in the liquid and gas phases.
        """
        # This routine called by Objective.stage() will run before "get".
        # It submits the jobs to the Work Queue and the stage() function will wait for jobs to complete.
        printcool("Target: %s - launching %i MD simulations\nTime steps (liq):" 
                  "%i (eq) + %i (md)\nTime steps (g): %i (eq) + %i (md)" % 
                  (self.name, 2*len(self.IDs), self.liquid_eq_steps, self.liquid_md_steps,
                   self.gas_eq_steps, self.gas_md_steps), color=0)
        # If self.save_traj == 1, delete the trajectory files from a previous good optimization step.
        if self.evaluated and self.goodstep and self.save_traj < 2:
            for fn in self.last_traj:
                if os.path.exists(fn):
                    os.remove(fn)
        self.last_traj = []
        # Set up and run the NPT simulations.
        # Less fully featured than liquid simulation; NOT INCLUDED are
        # 1) Temperature and pressure
        # 2) Multiple initial conditions
        for label in self.IDs:
            if not os.path.exists(label):
                os.makedirs(label)
            os.chdir(label)
            # Run liquid and gas phase simulations.
            self.run_simulation(label, 0, AGrad)
            self.run_simulation(label, 1, AGrad)
            os.chdir('..')

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        # If not calculating HFE using simulations, exit this function.
        if self.hfemode.lower() not in ['ti2', 'exp_gas', 'exp_liq', 'exp_both']:
            return
        else:
            # Prior to running simulations, write the force field pickle
            # file which will be shared by all simulations.
            self.serialize_ff(mvals)
        if self.hfemode.lower() in ['ti2', 'exp_gas', 'exp_liq', 'exp_both']:
            self.submit_liq_gas(mvals, AGrad)

    def build_engines(self):
        """ Create a list of engines which are used to calculate HFEs using single point evaluation. """
        self.engines = OrderedDict()
        self.liq_engines = OrderedDict()
        self.gas_engines = OrderedDict()
        for mnm in self.IDs:
            pdbfnm = os.path.abspath(os.path.join(self.root,self.tgtdir, 'molecules', mnm+'.pdb'))
            self.liq_engines[mnm] = self.engine_(target=self, coords=pdbfnm, implicit_solvent=True, **self.engine_opts)
            self.gas_engines[mnm] = self.engine_(target=self, coords=pdbfnm, implicit_solvent=False, **self.engine_opts)

    def indicate(self):
        """ Print qualitative indicator. """
        banner = "Hydration free energies (kcal/mol)"
        headings = ["ID"]
        data = OrderedDict([(ID, []) for ID in self.IDs])
        if self.have_nicks:
            headings.append("Nickname")
            for ID in self.IDs:
                data[ID].append(self.nicknames[ID])
        if self.have_experr:
            headings.append("Reference +- StdErr")
            for ID in self.IDs:
                data[ID].append("% 9.3f +- %6.3f" % (self.expval[ID], self.experr[ID]))
        else:
            headings.append("Reference")
            for ID in self.IDs:
                data[ID].append("% 9.3f" % self.expval[ID])
        if hasattr(self, 'calc_err'):
            headings.append("Calculated +- StdErr")
            for ID in self.IDs:
                data[ID].append("% 10.3f +- %6.3f" % (self.calc[ID], self.calc_err[ID]))
        else:
            headings.append("Calculated")
            for ID in self.IDs:
                data[ID].append("% 10.3f" % self.calc[ID])
        headings += ["Calc-Ref", "Weight", "Residual"]
        for iid, ID in enumerate(self.IDs):
            data[ID].append("% 8.3f" % (self.calc[ID] - self.expval[ID]))
            data[ID].append("%8.3f" % (self.whfe[iid]))
            data[ID].append("%8.3f" % (self.whfe[iid]*(self.calc[ID] - self.expval[ID])**2))
        self.printcool_table(data, headings, banner)

    def hydration_driver_sp(self):
        """ Calculate HFEs using single point evaluation. """
        hfe = OrderedDict()
        for mnm in self.IDs:
            eliq, rmsdliq = self.liq_engines[mnm].optimize()
            egas, rmsdgas = self.gas_engines[mnm].optimize()
            hfe[mnm] = eliq - egas
        return hfe

    def get_sp(self, mvals, AGrad=False, AHess=False):
        """ Get the hydration free energy and first parameteric derivatives using single point energy evaluations. """
        def get_hfe(mvals_):
            self.FF.make(mvals_)
            self.hfe_dict = self.hydration_driver_sp()
            return np.array(self.hfe_dict.values())
        calc_hfe = get_hfe(mvals)
        D = calc_hfe - np.array(self.expval.values())
        dD = np.zeros((self.FF.np,len(self.IDs)))
        if AGrad or AHess:
            for p in self.pgrad:
                dD[p,:], _ = f12d3p(fdwrap(get_hfe, mvals, p), h = self.h, f0 = calc_hfe)
        return D, dD

    def get_exp(self, mvals, AGrad=False, AHess=False):
        """ Get the hydration free energy using the Zwanzig formula.  We will obtain two different estimates along with their uncertainties. """
        self.hfe_dict = OrderedDict()
        self.hfe_err = OrderedDict()
        dD = np.zeros((self.FF.np,len(self.IDs)))
        kT = (kb * self.hfe_temperature)
        beta = 1. / (kb * self.hfe_temperature)
        for ilabel, label in enumerate(self.IDs):
            os.chdir(label)
            # This dictionary contains observables keyed by each phase.
            data = defaultdict(dict)
            for p in ['gas', 'liq']:
                os.chdir(p)
                # Load the results from molecular dynamics.
                results = lp_load('md_result.p')
                L = len(results['Potentials'])
                if p == "gas":
                    Eg = results['Potentials']
                    Eaq = results['Potentials'] + results['Hydration']
                    # Mean and standard error of the exponentiated hydration energy.
                    expmbH = np.exp(-1.0*beta*results['Hydration'])
                    data[p]['Hyd'] = -kT*np.log(np.mean(expmbH))
                    # Estimate standard error by bootstrap method.  We also multiply by the 
                    # square root of the statistical inefficiency of the hydration energy time series.
                    data[p]['HydErr'] = np.std([-kT*np.log(np.mean(expmbH[np.random.randint(L,size=L)])) for i in range(100)]) * np.sqrt(statisticalInefficiency(results['Hydration']))
                    if AGrad: 
                        dEg = results['Potential_Derivatives']
                        dEaq = results['Potential_Derivatives'] + results['Hydration_Derivatives']
                        data[p]['dHyd'] = (flat(np.matrix(dEaq)*col(expmbH)/L)-np.mean(dEg,axis=1)*np.mean(expmbH)) / np.mean(expmbH)
                elif p == "liq":
                    Eg = results['Potentials'] - results['Hydration']
                    Eaq = results['Potentials']
                    # Mean and standard error of the exponentiated hydration energy.
                    exppbH = np.exp(+1.0*beta*results['Hydration'])
                    data[p]['Hyd'] = +kT*np.log(np.mean(exppbH))
                    # Estimate standard error by bootstrap method.  We also multiply by the 
                    # square root of the statistical inefficiency of the hydration energy time series.
                    data[p]['HydErr'] = np.std([+kT*np.log(np.mean(exppbH[np.random.randint(L,size=L)])) for i in range(100)]) * np.sqrt(statisticalInefficiency(results['Hydration']))
                    if AGrad: 
                        dEg = results['Potential_Derivatives'] - results['Hydration_Derivatives']
                        dEaq = results['Potential_Derivatives']
                        data[p]['dHyd'] = -(flat(np.matrix(dEg)*col(exppbH)/L)-np.mean(dEaq,axis=1)*np.mean(exppbH)) / np.mean(exppbH)
                os.chdir('..')
            # Calculate the hydration free energy using gas phase, liquid phase or the average of both.
            # Note that the molecular dynamics methods return energies in kJ/mol.
            if self.hfemode == 'exp_gas':
                self.hfe_dict[label] = data['gas']['Hyd'] / 4.184
                self.hfe_err[label] = data['gas']['HydErr'] / 4.184
            elif self.hfemode == 'exp_liq':
                self.hfe_dict[label] = data['liq']['Hyd'] / 4.184
                self.hfe_err[label] = data['liq']['HydErr'] / 4.184
            elif self.hfemode == 'exp_both':
                self.hfe_dict[label] = 0.5*(data['liq']['Hyd']+data['gas']['Hyd']) / 4.184
                self.hfe_err[label] = 0.5*(data['liq']['HydErr']+data['gas']['HydErr']) / 4.184
            if AGrad:
                # Calculate the derivative of the hydration free energy.
                if self.hfemode == 'exp_gas':
                    dD[:, ilabel] = self.whfe[ilabel]*data['gas']['dHyd'] / 4.184
                elif self.hfemode == 'exp_liq':
                    dD[:, ilabel] = self.whfe[ilabel]*data['liq']['dHyd'] / 4.184
                elif self.hfemode == 'exp_both':
                    dD[:, ilabel] = 0.5*self.whfe[ilabel]*(data['liq']['dHyd']+data['gas']['dHyd']) / 4.184
            os.chdir('..')
        calc_hfe = np.array(self.hfe_dict.values())
        D = self.whfe*(calc_hfe - np.array(self.expval.values()))
        return D, dD

    def get_ti2(self, mvals, AGrad=False, AHess=False):
        """ Get the hydration free energy using two-point thermodynamic integration. """
        self.hfe_dict = OrderedDict()
        dD = np.zeros((self.FF.np,len(self.IDs)))
        beta = 1. / (kb * self.hfe_temperature)
        for ilabel, label in enumerate(self.IDs):
            os.chdir(label)
            # This dictionary contains observables keyed by each phase.
            data = defaultdict(dict)
            for p in ['gas', 'liq']:
                os.chdir(p)
                # Load the results from molecular dynamics.
                results = lp_load('md_result.p')
                # Time series of hydration energies.
                H = results['Hydration']
                # Store the average hydration energy.
                data[p]['Hyd'] = np.mean(H)
                if AGrad:
                    dE = results['Potential_Derivatives']
                    dH = results['Hydration_Derivatives']
                    # Calculate the parametric derivative of the average hydration energy.
                    data[p]['dHyd'] = np.mean(dH,axis=1)-beta*(flat(np.matrix(dE)*col(H)/len(H))-np.mean(dE,axis=1)*np.mean(H))
                os.chdir('..')
            # Calculate the hydration free energy as the average of liquid and gas hydration energies.
            # Note that the molecular dynamics methods return energies in kJ/mol.
            self.hfe_dict[label] = 0.5*(data['liq']['Hyd']+data['gas']['Hyd']) / 4.184
            if AGrad:
                # Calculate the derivative of the hydration free energy.
                dD[:, ilabel] = 0.5*self.whfe[ilabel]*(data['liq']['dHyd']+data['gas']['dHyd']) / 4.184
            os.chdir('..')
        calc_hfe = np.array(self.hfe_dict.values())
        D = self.whfe*(calc_hfe - np.array(self.expval.values()))
        return D, dD

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        if self.hfemode.lower() == 'single' or self.hfemode.lower() == 'sp':
            D, dD = self.get_sp(mvals, AGrad, AHess)
        elif self.hfemode.lower() == 'ti2':
            D, dD = self.get_ti2(mvals, AGrad, AHess)
        elif self.hfemode.lower() in ['exp_gas', 'exp_liq', 'exp_both']:
            D, dD = self.get_exp(mvals, AGrad, AHess)
        Answer['X'] = np.dot(D,D) / self.denom**2 / (np.sum(self.whfe) if self.normalize else 1)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(D, dD[p,:]) / self.denom**2 / (np.sum(self.whfe) if self.normalize else 1)
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dD[p,:], dD[q,:]) / self.denom**2 / (np.sum(self.whfe) if self.normalize else 1)
        if not in_fd():
            self.calc = self.hfe_dict
            if hasattr(self, 'hfe_err'):
                self.calc_err = self.hfe_err
            self.objective = Answer['X']
        return Answer
