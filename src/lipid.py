""" @package forcebalance.lipid Matching of lipid bulk properties.  Under development.

author Lee-Ping Wang
@date 04/2012
"""

import abc
import os
import shutil
from forcebalance.finite_difference import *
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from forcebalance.target import Target
import numpy as np
from forcebalance.molecule import Molecule
from re import match, sub
import subprocess
from subprocess import PIPE
try:
    from lxml import etree
except: pass
from pymbar import pymbar
import itertools
from collections import defaultdict, namedtuple, OrderedDict
import csv

from forcebalance.output import getLogger
logger = getLogger(__name__)

def weight_info(W, PT, N_k, verbose=True):
    C = []
    N = 0
    W += 1.0e-300
    I = np.exp(-1*np.sum((W*np.log(W))))
    for ns in N_k:
        C.append(sum(W[N:N+ns]))
        N += ns
    C = np.array(C)
    if verbose:
        logger.info("MBAR Results for Phase Point %s, Box, Contributions:\n" % str(PT))
        logger.info(str(C) + '\n')
        logger.info("InfoContent: % .2f snapshots (%.2f %%)\n" % (I, 100*I/len(W)))
    return C

# NPT_Trajectory = namedtuple('NPT_Trajectory', ['fnm', 'Rhos', 'pVs', 'Energies', 'Grads', 'mEnergies', 'mGrads', 'Rho_errs', 'Hvap_errs'])

class Lipid(Target):
    
    """ Subclass of Target for lipid property matching."""

    def __init__(self,options,tgt_opts,forcefield):
        # Initialize base class
        super(Lipid,self).__init__(options,tgt_opts,forcefield)
        # Weight of the density
        self.set_option(tgt_opts,'w_rho',forceprint=True)
        # Weight of the thermal expansion coefficient
        self.set_option(tgt_opts,'w_alpha',forceprint=True)
        # Weight of the isothermal compressibility
        self.set_option(tgt_opts,'w_kappa',forceprint=True)
        # Weight of the isobaric heat capacity
        self.set_option(tgt_opts,'w_cp',forceprint=True)
        # Weight of the dielectric constant
        self.set_option(tgt_opts,'w_eps0',forceprint=True)
        # Weight of the area per lipid
        self.set_option(tgt_opts,'w_al',forceprint=True)
        # Weight of the bilayer isothermal compressibility
        self.set_option(tgt_opts,'w_lkappa',forceprint=True)
        # Weight of the deuterium order parameter
        self.set_option(tgt_opts,'w_scd',forceprint=True)
        # Normalize the property contributions to the objective function
        self.set_option(tgt_opts,'w_normalize',forceprint=True)
        # Optionally pause on the zeroth step
        self.set_option(tgt_opts,'manual')
        # Number of time steps in the lipid "equilibration" run
        self.set_option(tgt_opts,'lipid_eq_steps',forceprint=True)
        # Number of time steps in the lipid "production" run
        self.set_option(tgt_opts,'lipid_md_steps',forceprint=True)
        # Number of time steps in the gas "equilibration" run
        self.set_option(tgt_opts,'gas_eq_steps',forceprint=False)
        # Number of time steps in the gas "production" run
        self.set_option(tgt_opts,'gas_md_steps',forceprint=False)
        # Cutoff for nonbonded interactions in the liquid
        if tgt_opts['nonbonded_cutoff'] is not None:
            self.set_option(tgt_opts,'nonbonded_cutoff')
        # Cutoff for vdW interactions if different from other nonbonded interactions
        if tgt_opts['vdw_cutoff'] is not None:
            self.set_option(tgt_opts,'vdw_cutoff')
        # Time step length (in fs) for the lipid production run
        self.set_option(tgt_opts,'lipid_timestep',forceprint=True)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'lipid_interval',forceprint=True)
        # Time step length (in fs) for the gas production run
        self.set_option(tgt_opts,'gas_timestep',forceprint=True)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'gas_interval',forceprint=True)
        # Minimize the energy prior to running any dynamics
        self.set_option(tgt_opts,'minimize_energy',forceprint=True)
        # Isolated dipole (debye) for analytic self-polarization correction.
        self.set_option(tgt_opts,'self_pol_mu0',forceprint=True)
        # Molecular polarizability (ang**3) for analytic self-polarization correction.
        self.set_option(tgt_opts,'self_pol_alpha',forceprint=True)
        # Set up the simulation object for self-polarization correction.
        self.do_self_pol = (self.self_pol_mu0 > 0.0 and self.self_pol_alpha > 0.0)
        # Enable anisotropic periodic box
        self.set_option(tgt_opts,'anisotropic_box',forceprint=True)
        # Whether to save trajectories (0 = never, 1 = delete after good step, 2 = keep all)
        self.set_option(tgt_opts,'save_traj')

        #======================================#
        #     Variables which are set here     #
        #======================================#
        # List of trajectory files that may be deleted if self.save_traj == 1.
        self.last_traj = []
        # Extra files to be copied back at the end of a run.
        self.extra_output = []
        # Read the reference data
        self.read_data()
        # Read in lipid starting coordinates.
        if 'n_ic' in self.RefData:
            # Linked IC folder into the temp-directory.
            self.nptfiles += ["IC"]
            # Store IC frames in a dictionary.
            self.lipid_mols = OrderedDict()
            self.lipid_mols_new = OrderedDict()
            for pt in self.PhasePoints:
                pt_label = "IC/%sK-%s%s" % (pt[0], pt[1], pt[2])
                if not os.path.exists(os.path.join(self.root, self.tgtdir, pt_label, self.lipid_coords)):
                    raise RuntimeError("Initial condition files don't exist; please provide IC directory")
                # Create molecule object for each IC.
                all_ic = Molecule(os.path.join(self.root, self.tgtdir, pt_label, self.lipid_coords))
                self.lipid_mols[pt] = []
                n_uniq_ic = int(self.RefData['n_ic'][pt])
                if n_uniq_ic > len(all_ic):
                    raise RuntimeError("Number of frames in initial conditions .gro file is less than the number of parallel simulations requested in data.csv")
                # Index ICs by pressure and temperature in a dictionary.
                for ic in range(n_uniq_ic):
                    self.lipid_mols[pt].append(all_ic[ic])
        else:
            # Read in lipid starting coordinates.
            if not os.path.exists(os.path.join(self.root, self.tgtdir, self.lipid_coords)): 
                logger.error("%s doesn't exist; please provide lipid_coords option\n" % self.lipid_coords)
                raise RuntimeError
            self.lipid_mol = Molecule(os.path.join(self.root, self.tgtdir, self.lipid_coords), toppbc=True)
            # Extra files to be linked into the temp-directory.
            self.nptfiles += [self.lipid_coords]
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts += ['npt_lipid.py']
        # Prepare the temporary directory.
        self.prepare_temp_directory()
        # Build keyword dictionary to pass to engine.
        if self.do_self_pol:
            self.gas_engine_args.update(self.OptionDict)
            self.gas_engine_args.update(options)
            del self.gas_engine_args['name']
            # Create engine object for gas molecule to do the polarization correction.
            self.gas_engine = self.engine_(target=self, mol=self.gas_mol, name="selfpol", **self.gas_engine_args)
        # Don't read indicate.log when calling meta_indicate()
        self.read_indicate = False
        self.write_indicate = False
        # Don't read objective.p when calling meta_get()
        self.read_objective = False
        #======================================#
        #          UNDER DEVELOPMENT           #
        #======================================#
        # Put stuff here that I'm not sure about. :)
        np.set_printoptions(precision=4, linewidth=100)
        np.seterr(under='ignore')
        ## Saved force field mvals for all iterations
        self.SavedMVal = {}
        ## Saved trajectories for all iterations and all temperatures
        self.SavedTraj = defaultdict(dict)
        ## Evaluated energies for all trajectories (i.e. all iterations and all temperatures), using all mvals
        self.MBarEnergy = defaultdict(lambda:defaultdict(dict))

    def prepare_temp_directory(self):
        """ Prepare the temporary directory by copying in important files. """
        abstempdir = os.path.join(self.root,self.tempdir)
        for f in self.nptfiles:
            LinkFile(os.path.join(self.root, self.tgtdir, f), os.path.join(abstempdir, f))
        for f in self.scripts:
            LinkFile(os.path.join(os.path.split(__file__)[0],"data",f),os.path.join(abstempdir,f))

    def read_data(self):
        # Read the 'data.csv' file. The file should contain guidelines.
        with open(os.path.join(self.tgtdir,'data.csv'),'rU') as f: R0 = list(csv.reader(f))
        # All comments are erased.
        R1 = [[sub('#.*$','',word) for word in line] for line in R0 if len(line[0]) > 0 and line[0][0] != "#"]
        # All empty lines are deleted and words are converted to lowercase.
        R = [[wrd.lower() for wrd in line] for line in R1 if any([len(wrd) for wrd in line]) > 0]
        global_opts = OrderedDict()
        found_headings = False
        known_vars = ['mbar','rho','hvap','alpha','kappa','cp','eps0','cvib_intra',
                      'cvib_inter','cni','devib_intra','devib_inter', 'al', 'scd', 'n_ic', 'lkappa']
        self.RefData = OrderedDict()
        for line in R:
            if line[0] == "global":
                # Global options are mainly denominators for the different observables.
                if isfloat(line[2]):
                    global_opts[line[1]] = float(line[2])
                elif line[2].lower() == 'false':
                    global_opts[line[1]] = False
                elif line[2].lower() == 'true':
                    global_opts[line[1]] = True
            elif not found_headings:
                found_headings = True
                headings = line
                if len(set(headings)) != len(headings):
                    logger.error('Column headings in data.csv must be unique\n')
                    raise RuntimeError
                if 'p' not in headings:
                    logger.error('There must be a pressure column heading labeled by "p" in data.csv\n')
                    raise RuntimeError
                if 't' not in headings:
                    logger.error('There must be a temperature column heading labeled by "t" in data.csv\n')
                    raise RuntimeError
            elif found_headings:
                try:
                    # Temperatures are in kelvin.
                    t     = [float(val) for head, val in zip(headings,line) if head == 't'][0]
                    # For convenience, users may input the pressure in atmosphere or bar.
                    pval  = [float(val.split()[0]) for head, val in zip(headings,line) if head == 'p'][0]
                    punit = [val.split()[1] if len(val.split()) >= 1 else "atm" for head, val in zip(headings,line) if head == 'p'][0]
                    unrec = set([punit]).difference(['atm','bar']) 
                    if len(unrec) > 0:
                        logger.error('The pressure unit %s is not recognized, please use bar or atm\n' % unrec[0])
                        raise RuntimeError
                    # This line actually reads the reference data and inserts it into the RefData dictionary of dictionaries.
                    for head, val in zip(headings,line):
                        if head == 't' or head == 'p' : continue
                        if isfloat(val):
                            self.RefData.setdefault(head,OrderedDict([]))[(t,pval,punit)] = float(val.strip())
                        elif val.lower() == 'true':
                            self.RefData.setdefault(head,OrderedDict([]))[(t,pval,punit)] = True
                        elif val.lower() == 'false':
                            self.RefData.setdefault(head,OrderedDict([]))[(t,pval,punit)] = False
                        elif head == 'scd':
                            self.RefData.setdefault(head,OrderedDict([]))[(t,pval,punit)] = np.array(map(float, val.split()))
                except:
                    logger.error(line + '\n')
                    logger.error('Encountered an error reading this line!\n')
                    raise RuntimeError
            else:
                logger.error(line + '\n')
                logger.error('I did not recognize this line!\n')
                raise RuntimeError
        # Check the reference data table for validity.
        default_denoms = defaultdict(int)
        PhasePoints = None
        for head in self.RefData:
            if head == 'n_ic':
                continue
            if head not in known_vars+[i+"_wt" for i in known_vars]:
                # Only hard-coded properties may be recognized.
                logger.error("The column heading %s is not recognized in data.csv\n" % head)
                raise RuntimeError
            if head in known_vars:
                if head+"_wt" not in self.RefData:
                    # If the phase-point weights are not specified in the reference data file, initialize them all to one.
                    self.RefData[head+"_wt"] = OrderedDict([(key, 1.0) for key in self.RefData[head]])
                wts = np.array(self.RefData[head+"_wt"].values())
                dat = np.array(self.RefData[head].values())
                # S_cd specifies an array of averages (one for each tail node).  Find avg over axis 0.
                avg = np.average(dat, weights=wts, axis=0)
                if len(wts) > 1:
                    # If there is more than one data point, then the default denominator is the
                    # standard deviation of the experimental values.
                    if head == 'scd':
                        default_denoms[head+"_denom"] = np.average(np.sqrt(np.dot(wts, (dat-avg)**2)/wts.sum()))
                    else:
                        default_denoms[head+"_denom"] = np.sqrt(np.dot(wts, (dat-avg)**2)/wts.sum())
                else:
                    # If there is only one data point, then the denominator is just the single
                    # data point itself.
                    if head == 'scd':
                        default_denoms[head+"_denom"] = np.average(np.sqrt(np.abs(dat[0])))
                    else:
                        default_denoms[head+"_denom"] = np.sqrt(np.abs(dat[0]))
            self.PhasePoints = self.RefData[head].keys()
            # This prints out all of the reference data.
            # printcool_dictionary(self.RefData[head],head)
        # Create labels for the directories.
        self.Labels = ["%.2fK-%.1f%s" % i for i in self.PhasePoints]
        logger.debug("global_opts:\n%s\n" % str(global_opts))
        logger.debug("default_denoms:\n%s\n" % str(default_denoms))
        for opt in global_opts:
            if "_denom" in opt:
                # Record entries from the global_opts dictionary so they can be retrieved from other methods.
                self.set_option(global_opts,opt,default=default_denoms[opt])
            else:
                self.set_option(global_opts,opt)

    def check_files(self, there):
        there = os.path.abspath(there)
        havepts = 0
        if all([i in os.listdir(there) for i in self.Labels]):
            for d in os.listdir(there):
                if d in self.Labels:
                    if os.path.exists(os.path.join(there, d, 'npt_result.p')):
                        havepts += 1
        if (float(havepts)/len(self.Labels)) > 0.75:
            return 1
        else:
            return 0
    def npt_simulation(self, temperature, pressure, simnum):
        """ Submit a NPT simulation to the Work Queue. """
        wq = getWorkQueue()
        if not os.path.exists('npt_result.p'):
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            self.last_traj += [os.path.join(os.getcwd(), i) for i in self.extra_output]
            self.lipid_mol[simnum%len(self.lipid_mol)].write(self.lipid_coords, ftype='tinker' if self.engname == 'tinker' else None)
            cmdstr = '%s python npt_lipid.py %s %.3f %.3f' % (self.nptpfx, self.engname, temperature, pressure)
            if wq is None:
                logger.info("Running condensed phase simulation locally.\n")
                logger.info("You may tail -f %s/npt.out in another terminal window\n" % os.getcwd())
                _exec(cmdstr, copy_stderr=True, outfnm='npt.out')
            else:
                queue_up(wq, command = cmdstr+' &> npt.out',
                         input_files = self.nptfiles + self.scripts + ['forcebalance.p'],
                         output_files = ['npt_result.p', 'npt.out'] + self.extra_output, tgt=self)

    def polarization_correction(self,mvals):
        d = self.gas_engine.get_multipole_moments(optimize=True)['dipole']
        if not in_fd():
            logger.info("The molecular dipole moment is % .3f debye\n" % np.linalg.norm(d))
        # Taken from the original OpenMM interface code, this is how we calculate the conversion factor.
        # dd2 = ((np.linalg.norm(d)-self.self_pol_mu0)*debye)**2
        # eps0 = 8.854187817620e-12 * coulomb**2 / newton / meter**2
        # epol = 0.5*dd2/(self.self_pol_alpha*angstrom**3*4*np.pi*eps0)/(kilojoule_per_mole/AVOGADRO_CONSTANT_NA)
        # In [2]: eps0 = 8.854187817620e-12 * coulomb**2 / newton / meter**2
        # In [7]: 1.0 * debye ** 2 / (1.0 * angstrom**3*4*np.pi*eps0) / (kilojoule_per_mole/AVOGADRO_CONSTANT_NA)
        # Out[7]: 60.240179789402056
        convert = 60.240179789402056
        dd2 = (np.linalg.norm(d)-self.self_pol_mu0)**2
        epol = 0.5*convert*dd2/self.self_pol_alpha
        return epol

    def indicate(self): 
        AGrad = hasattr(self, 'Gp')
        PrintDict = OrderedDict()
        def print_item(key, heading, physunit):
            if self.Xp[key] > 0:
                printcool_dictionary(self.Pp[key], title='%s %s%s\nTemperature  Pressure  Reference  Calculated +- Stdev     Delta    Weight    Term   ' % 
                                     (self.name, heading, " (%s) " % physunit if physunit else ""), bold=True, color=4, keywidth=15)
                bar = printcool("%s objective function: % .3f%s" % (heading, self.Xp[key], ", Derivative:" if AGrad else ""))
                if AGrad:
                    self.FF.print_map(vals=self.Gp[key])
                    logger.info(bar)
                PrintDict[heading] = "% 10.5f % 8.3f % 14.5e" % (self.Xp[key], self.Wp[key], self.Xp[key]*self.Wp[key])

        print_item("Rho", "Density", "kg m^-3")
        print_item("Alpha", "Thermal Expansion Coefficient", "10^-4 K^-1")
        print_item("Kappa", "Isothermal Compressibility", "10^-6 bar^-1")
        print_item("Cp", "Isobaric Heat Capacity", "cal mol^-1 K^-1")
        print_item("Eps0", "Dielectric Constant", None)
        print_item("Al", "Average Area per Lipid", "nm^2")
        print_item("Scd", "Deuterium Order Parameter", None)
        print_item("LKappa", "Bilayer Isothermal Compressibility", "mN/m")

        PrintDict['Total'] = "% 10s % 8s % 14.5e" % ("","",self.Objective)

        Title = "%s Condensed Phase Properties:\n %-20s %40s" % (self.name, "Property Name", "Residual x Weight = Contribution")
        printcool_dictionary(PrintDict,color=4,title=Title,keywidth=31)
        return

    def objective_term(self, points, expname, calc, err, grad, name="Quantity", SubAverage=False):
        if expname in self.RefData:
            exp = self.RefData[expname]
            Weights = self.RefData[expname+"_wt"]
            Denom = getattr(self,expname+"_denom")
        else:
            # If the reference data doesn't exist then return nothing.
            return 0.0, np.zeros(self.FF.np), np.zeros((self.FF.np,self.FF.np)), None
            
        Sum = sum(Weights.values())
        for i in Weights:
            Weights[i] /= Sum
        logger.info("Weights have been renormalized to " + str(sum(Weights.values())) + "\n")
        # Use least-squares or hyperbolic (experimental) objective.
        LeastSquares = True

        logger.info("Physical quantity %s uses denominator = % .4f\n" % (name, Denom))
        if not LeastSquares:
            # If using a hyperbolic functional form
            # we still want the contribution to the 
            # objective function to be the same when
            # Delta = Denom.
            Denom /= 3 ** 0.5
        
        Objective = 0.0
        Gradient = np.zeros(self.FF.np)
        Hessian = np.zeros((self.FF.np,self.FF.np))
        Objs = {}
        GradMap = []
        avgCalc = 0.0
        avgExp  = 0.0
        avgGrad = np.zeros(self.FF.np)
        for i, PT in enumerate(points):
            avgCalc += Weights[PT]*calc[PT]
            avgExp  += Weights[PT]*exp[PT]
            avgGrad += Weights[PT]*grad[PT]
        for i, PT in enumerate(points):
            if SubAverage:
                G = grad[PT]-avgGrad
                Delta = calc[PT] - exp[PT] - avgCalc + avgExp
            else:
                G = grad[PT]
                Delta = calc[PT] - exp[PT]
            if hasattr(Delta, "__len__"):
                Delta = np.average(Delta)
            if LeastSquares:
                # Least-squares objective function.
                ThisObj = Weights[PT] * Delta ** 2 / Denom**2
                Objs[PT] = ThisObj
                ThisGrad = 2.0 * Weights[PT] * Delta * G / Denom**2
                GradMap.append(G)
                Objective += ThisObj
                Gradient += ThisGrad
                # Gauss-Newton approximation to the Hessian.
                Hessian += 2.0 * Weights[PT] * (np.outer(G, G)) / Denom**2
            else:
                # L1-like objective function.
                D = Denom
                S = Delta**2 + D**2
                ThisObj  = Weights[PT] * (S**0.5-D) / Denom
                ThisGrad = Weights[PT] * (Delta/S**0.5) * G / Denom
                ThisHess = Weights[PT] * (1/S**0.5-Delta**2/S**1.5) * np.outer(G,G) / Denom
                Objs[PT] = ThisObj
                GradMap.append(G)
                Objective += ThisObj
                Gradient += ThisGrad
                Hessian += ThisHess
        GradMapPrint = [["#PhasePoint"] + self.FF.plist]
        for PT, g in zip(points,GradMap):
            GradMapPrint.append([' %8.2f %8.1f %3s' % PT] + ["% 9.3e" % i for i in g])
        o = wopen('gradient_%s.dat' % name)
        for line in GradMapPrint:
            print >> o, ' '.join(line)
        o.close()
            
        Delta = np.array([calc[PT] - exp[PT] for PT in points])
        delt = {PT : r for PT, r in zip(points,Delta)}
        if expname == 'scd': 
            print_out = OrderedDict([('    %8.2f %8.1f %3s' % PT, '\n %s' % (' '.join('\t \t \t %9.6f    %9.6f +- %-7.6f % 7.6f \n' % F for F in zip(exp[PT], calc[PT], flat(err[PT]), delt[PT])))) for PT in calc])
        else:
            print_out = OrderedDict([('    %8.2f %8.1f %3s' % PT, "%9.3f    %9.3f +- %-7.3f % 7.3f % 9.5f % 9.5f" % (exp[PT],calc[PT],err[PT],delt[PT],Weights[PT],Objs[PT])) for PT in calc])

        return Objective, Gradient, Hessian, print_out

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        # This routine is called by Objective.stage() will run before "get".
        # It submits the jobs to the Work Queue and the stage() function will wait for jobs to complete.
        #
        # First dump the force field to a pickle file
        lp_dump((self.FF,mvals,self.OptionDict,AGrad),'forcebalance.p')

        # Give the user an opportunity to copy over data from a previous (perhaps failed) run.
        if (not self.evaluated) and self.manual:
            warn_press_key("Now's our chance to fill the temp directory up with data!\n(Considering using 'read' or 'continue' for better checkpointing)", timeout=7200)

        # If self.save_traj == 1, delete the trajectory files from a previous good optimization step.
        if self.evaluated and self.goodstep and self.save_traj < 2:
            for fn in self.last_traj:
                if os.path.exists(fn):
                    os.remove(fn)
        self.last_traj = []

        # Set up and run the NPT simulations.
        snum = 0
        for label, pt in zip(self.Labels, self.PhasePoints):
            T = pt[0]
            P = pt[1]
            Punit = pt[2]
            if Punit == 'bar':
                P *= 1.0 / 1.01325
            if not os.path.exists(label):
                os.makedirs(label)
                os.chdir(label)
                if 'n_ic' in self.RefData:
                    n_uniq_ic = int(self.RefData['n_ic'][pt])
                    # Loop over parallel trajectories.
                    for trj in range(n_uniq_ic):
                        rel_trj = "trj_%i" % trj
                        # Create directories for each parallel simulation.
                        if not os.path.exists(rel_trj):
                            os.makedirs(rel_trj)
                            os.chdir(rel_trj)
                            # Pull each simulation molecule from the lipid_mols dictionary.
                            # lipid_mols is a dictionary of paths to either the initial 
                            # geometry files, or the geometries from the final frame of the 
                            # previous iteration.
                            self.lipid_mol = self.lipid_mols[pt][trj]
                            self.lipid_mol.write(self.lipid_coords)
                            if not self.lipid_coords in self.nptfiles:
                                self.nptfiles += [self.lipid_coords]
                            self.npt_simulation(T,P,snum)
                        os.chdir('..')
                else:
                    self.npt_simulation(T,P,snum)
                os.chdir('..')
                snum += 1

    def get(self, mvals, AGrad=True, AHess=True):
        
        """
        Fitting of lipid bulk properties.  This is the current major
        direction of development for ForceBalance.  Basically, fitting
        the QM energies / forces alone does not always give us the
        best simulation behavior.  In many cases it makes more sense
        to try and reproduce some experimentally known data as well.

        In order to reproduce experimentally known data, we need to
        run a simulation and compare the simulation result to
        experiment.  The main challenge here is that the simulations
        are computationally intensive (i.e. they require energy and
        force evaluations), and furthermore the results are noisy.  We
        need to run the simulations automatically and remotely
        (i.e. on clusters) and a good way to calculate the derivatives
        of the simulation results with respect to the parameter values.

        This function contains some experimentally known values of the
        density and enthalpy of vaporization (Hvap) of lipid water.
        It launches the density and Hvap calculations on the cluster,
        and gathers the results / derivatives.  The actual calculation
        of results / derivatives is done in a separate file.

        After the results come back, they are gathered together to form
        an objective function.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        
        """

        mbar_verbose = False

        Answer = {}

        Results = {}
        Points = []  # These are the phase points for which data exists.
        BPoints = [] # These are the phase points for which we are doing MBAR for the condensed phase.
        tt = 0
        for label, PT in zip(self.Labels, self.PhasePoints):
            if 'n_ic' in self.RefData:
                self.lipid_mols[PT] = [Molecule(last_frame) for last_frame in self.lipid_mols[PT]]
                n_uniq_ic = int(self.RefData['n_ic'][PT])
                for ic in range(n_uniq_ic):
                    if os.path.exists('./%s/trj_%s/npt_result.p' % (label, ic)):
                        # Read in each each parallel simulation's data, and concatenate each property time series.
                        ts = lp_load('./%s/trj_%s/npt_result.p' % (label, ic))
                        if ic == 0:
                            ts_concat = list(ts)
                        else:
                            for d_arr in range(len(ts)):
                                if isinstance(ts[d_arr], np.ndarray):
                                    # Gradients need a unique append format.
                                    if d_arr == 5:
                                        ts_concat[d_arr] = np.append(ts_concat[d_arr], ts[d_arr], axis = 1)
                                    else:
                                        ts_concat[d_arr] = np.append(ts_concat[d_arr], ts[d_arr], axis = 0)
                                if isinstance(ts_concat[d_arr], list):
                                    ts_concat[d_arr] = [np.append(ts_concat[d_arr][i], ts[d_arr][i], axis = 1) for i in range(len(ts_concat[d_arr]))]
                        # Write concatenated time series to a pickle file.
                        if ic == (int(n_uniq_ic) - 1):
                            lp_dump((ts_concat), './%s/npt_result.p' % label)
            if os.path.exists('./%s/npt_result.p' % label):
                logger.info('Reading information from ./%s/npt_result.p\n' % label)
                Points.append(PT)
                Results[tt] = lp_load('./%s/npt_result.p' % label)
                tt += 1
            else:
                logger.warning('The file ./%s/npt_result.p does not exist so we cannot read it\n' % label)
                pass
                # for obs in self.RefData:
                #     del self.RefData[obs][PT]
        if len(Points) == 0:
            logger.error('The lipid simulations have terminated with \x1b[1;91mno readable data\x1b[0m - this is a problem!\n')
            raise RuntimeError

        # Assign variable names to all the stuff in npt_result.p
        Rhos, Vols, Potentials, Energies, Dips, Grads, GDips, \
            Rho_errs, Alpha_errs, Kappa_errs, Cp_errs, Eps0_errs, NMols, Als, Al_errs, Scds, Scd_errs, LKappa_errs = ([Results[t][i] for t in range(len(Points))] for i in range(18))
        # Determine the number of molecules
        if len(set(NMols)) != 1:
            logger.error(str(NMols))
            logger.error('The above list should only contain one number - the number of molecules\n')
            raise RuntimeError
        else:
            NMol = list(set(NMols))[0]
    
        R  = np.array(list(itertools.chain(*list(Rhos))))
        V  = np.array(list(itertools.chain(*list(Vols))))
        E  = np.array(list(itertools.chain(*list(Energies))))
        Dx = np.array(list(itertools.chain(*list(d[:,0] for d in Dips))))
        Dy = np.array(list(itertools.chain(*list(d[:,1] for d in Dips))))
        Dz = np.array(list(itertools.chain(*list(d[:,2] for d in Dips))))
        G  = np.hstack(tuple(Grads))
        GDx = np.hstack(tuple(gd[0] for gd in GDips))
        GDy = np.hstack(tuple(gd[1] for gd in GDips))
        GDz = np.hstack(tuple(gd[2] for gd in GDips))
        A  = np.array(list(itertools.chain(*list(Als))))
        S  = np.array(list(itertools.chain(*list(Scds))))

        Rho_calc = OrderedDict([])
        Rho_grad = OrderedDict([])
        Rho_std  = OrderedDict([])
        Alpha_calc = OrderedDict([])
        Alpha_grad = OrderedDict([])
        Alpha_std  = OrderedDict([])
        Kappa_calc = OrderedDict([])
        Kappa_grad = OrderedDict([])
        Kappa_std  = OrderedDict([])
        Cp_calc = OrderedDict([])
        Cp_grad = OrderedDict([])
        Cp_std  = OrderedDict([])
        Eps0_calc = OrderedDict([])
        Eps0_grad = OrderedDict([])
        Eps0_std  = OrderedDict([])
        Al_calc = OrderedDict([])
        Al_grad = OrderedDict([])
        Al_std  = OrderedDict([])
        LKappa_calc = OrderedDict([])
        LKappa_grad = OrderedDict([])
        LKappa_std  = OrderedDict([])
        Scd_calc = OrderedDict([])
        Scd_grad = OrderedDict([])
        Scd_std  = OrderedDict([])

        # The unit that converts atmospheres * nm**3 into kj/mol :)
        pvkj=0.061019351687175
 
        # Run MBAR using the total energies. Required for estimates that use the kinetic energy.
        BSims = len(BPoints)
        Shots = len(Energies[0])
        Shots_m = [len(i) for i in Energies]
        N_k = np.ones(BSims)*Shots
        # Use the value of the energy for snapshot t from simulation k at potential m
        U_kln = np.zeros([BSims,BSims,Shots])
        for m, PT in enumerate(BPoints):
            T = PT[0]
            P = PT[1] / 1.01325 if PT[2] == 'bar' else PT[1]
            beta = 1. / (kb * T)
            for k in range(BSims):
                # The correct Boltzmann factors include PV.
                # Note that because the Boltzmann factors are computed from the conditions at simulation "m",
                # the pV terms must be rescaled to the pressure at simulation "m".
                kk = Points.index(BPoints[k])
                U_kln[k, m, :]   = Energies[kk] + P*Vols[kk]*pvkj
                U_kln[k, m, :]  *= beta
        W1 = None
        if len(BPoints) > 1:
            logger.info("Running MBAR analysis on %i states...\n" % len(BPoints))
            mbar = pymbar.MBAR(U_kln, N_k, verbose=mbar_verbose, relative_tolerance=5.0e-8)
            W1 = mbar.getWeights()
            logger.info("Done\n")
        elif len(BPoints) == 1:
            W1 = np.ones((BPoints*Shots,BPoints))
            W1 /= BPoints*Shots
        
        def fill_weights(weights, phase_points, mbar_points, snapshots):
            """ Fill in the weight matrix with MBAR weights where MBAR was run, 
            and equal weights otherwise. """
            new_weights = np.zeros([len(phase_points)*snapshots,len(phase_points)])
            for m, PT in enumerate(phase_points):
                if PT in mbar_points:
                    mm = mbar_points.index(PT)
                    for kk, PT1 in enumerate(mbar_points):
                        k = phase_points.index(PT1)
                        logger.debug("Will fill W2[%i:%i,%i] with W1[%i:%i,%i]\n" % (k*snapshots,k*snapshots+snapshots,m,kk*snapshots,kk*snapshots+snapshots,mm))
                        new_weights[k*snapshots:(k+1)*snapshots,m] = weights[kk*snapshots:(kk+1)*snapshots,mm]
                else:
                    logger.debug("Will fill W2[%i:%i,%i] with equal weights\n" % (m*snapshots,(m+1)*snapshots,m))
                    new_weights[m*snapshots:(m+1)*snapshots,m] = 1.0/snapshots
            return new_weights
        
        W2 = fill_weights(W1, Points, BPoints, Shots)

        if self.do_self_pol:
            EPol = self.polarization_correction(mvals)
            GEPol = np.array([(f12d3p(fdwrap(self.polarization_correction, mvals, p), h = self.h, f0 = EPol)[0] if p in self.pgrad else 0.0) for p in range(self.FF.np)])
            bar = printcool("Self-polarization correction to \nenthalpy of vaporization is % .3f kJ/mol%s" % (EPol, ", Derivative:" if AGrad else ""))
            if AGrad:
                self.FF.print_map(vals=GEPol)
                logger.info(bar)
            
        for i, PT in enumerate(Points):
            T = PT[0]
            P = PT[1] / 1.01325 if PT[2] == 'bar' else PT[1]
            PV = P*V*pvkj
            H = E + PV
            # The weights that we want are the last ones.
            W = flat(W2[:,i])
            C = weight_info(W, PT, np.ones(len(Points))*Shots, verbose=mbar_verbose)
            Gbar = flat(np.mat(G)*col(W))
            mBeta = -1/kb/T
            Beta  = 1/kb/T
            kT    = kb*T
            # Define some things to make the analytic derivatives easier.
            def avg(vec):
                return np.dot(W,vec)
            def covde(vec):
                return flat(np.mat(G)*col(W*vec)) - avg(vec)*Gbar
            def deprod(vec):
                return flat(np.mat(G)*col(W*vec))
            ## Density.
            Rho_calc[PT]   = np.dot(W,R)
            Rho_grad[PT]   = mBeta*(flat(np.mat(G)*col(W*R)) - np.dot(W,R)*Gbar)
            ## Ignore enthalpy.
            ## Thermal expansion coefficient.
            Alpha_calc[PT] = 1e4 * (avg(H*V)-avg(H)*avg(V))/avg(V)/(kT*T)
            GAlpha1 = -1 * Beta * deprod(H*V) * avg(V) / avg(V)**2
            GAlpha2 = +1 * Beta * avg(H*V) * deprod(V) / avg(V)**2
            GAlpha3 = deprod(V)/avg(V) - Gbar
            GAlpha4 = Beta * covde(H)
            Alpha_grad[PT] = 1e4 * (GAlpha1 + GAlpha2 + GAlpha3 + GAlpha4)/(kT*T)
            ## Isothermal compressibility.
            bar_unit = 0.06022141793 * 1e6
            Kappa_calc[PT] = bar_unit / kT * (avg(V**2)-avg(V)**2)/avg(V)
            GKappa1 = +1 * Beta**2 * avg(V**2) * deprod(V) / avg(V)**2
            GKappa2 = -1 * Beta**2 * avg(V) * deprod(V**2) / avg(V)**2
            GKappa3 = +1 * Beta**2 * covde(V)
            Kappa_grad[PT] = bar_unit*(GKappa1 + GKappa2 + GKappa3)
            ## Isobaric heat capacity.
            Cp_calc[PT] = 1000/(4.184*NMol*kT*T) * (avg(H**2) - avg(H)**2)
            if hasattr(self,'use_cvib_intra') and self.use_cvib_intra:
                logger.debug("Adding " + str(self.RefData['devib_intra'][PT]) + " to the heat capacity\n")
                Cp_calc[PT] += self.RefData['devib_intra'][PT]
            if hasattr(self,'use_cvib_inter') and self.use_cvib_inter:
                logger.debug("Adding " + str(self.RefData['devib_inter'][PT]) + " to the heat capacity\n")
                Cp_calc[PT] += self.RefData['devib_inter'][PT]
            GCp1 = 2*covde(H) * 1000 / 4.184 / (NMol*kT*T)
            GCp2 = mBeta*covde(H**2) * 1000 / 4.184 / (NMol*kT*T)
            GCp3 = 2*Beta*avg(H)*covde(H) * 1000 / 4.184 / (NMol*kT*T)
            Cp_grad[PT] = GCp1 + GCp2 + GCp3
            ## Static dielectric constant.
            prefactor = 30.348705333964077
            D2 = avg(Dx**2)+avg(Dy**2)+avg(Dz**2)-avg(Dx)**2-avg(Dy)**2-avg(Dz)**2
            Eps0_calc[PT] = 1.0 + prefactor*(D2/avg(V))/T
            GD2  = 2*(flat(np.mat(GDx)*col(W*Dx)) - avg(Dx)*flat(np.mat(GDx)*col(W))) - Beta*(covde(Dx**2) - 2*avg(Dx)*covde(Dx))
            GD2 += 2*(flat(np.mat(GDy)*col(W*Dy)) - avg(Dy)*flat(np.mat(GDy)*col(W))) - Beta*(covde(Dy**2) - 2*avg(Dy)*covde(Dy))
            GD2 += 2*(flat(np.mat(GDz)*col(W*Dz)) - avg(Dz)*flat(np.mat(GDz)*col(W))) - Beta*(covde(Dz**2) - 2*avg(Dz)*covde(Dz))
            Eps0_grad[PT] = prefactor*(GD2/avg(V) - mBeta*covde(V)*D2/avg(V)**2)/T
            ## Average area per lipid
            Al_calc[PT]   = np.dot(W,A)
            Al_grad[PT]   = mBeta*(flat(np.mat(G)*col(W*A)) - np.dot(W,A)*Gbar)
            ## Bilayer Isothermal compressibility.
            A_m2 = A * 1e-18
            kbT = 1.3806488e-23 * T
            LKappa_calc[PT] = (1e3 * 2 * kbT / 128) * (avg(A_m2) / (avg(A_m2**2)-avg(A_m2)**2))
            al_avg = avg(A_m2)
            al_sq_avg = avg(A_m2**2)
            al_avg_sq = al_avg**2
            al_var = al_sq_avg - al_avg_sq
            GLKappa1 = covde(A_m2) / al_var
            GLKappa2 = (al_avg / al_var**2) * (covde(A_m2**2) - (2 * al_avg * covde(A)))
            LKappa_grad[PT] = (1e3 * 2 * kbT / 128) * (GLKappa1 - GLKappa2)
            ## Deuterium order parameter
            Scd_calc[PT]   = np.dot(W,S)
            Scd_grad[PT]   = mBeta * (flat(np.average(np.mat(G) * (S * W[:, np.newaxis]), axis = 1)) - np.average(np.average(S * W[:, np.newaxis], axis = 0), axis = 0) * Gbar) 
            ## Estimation of errors.
            Rho_std[PT]    = np.sqrt(sum(C**2 * np.array(Rho_errs)**2))
            Alpha_std[PT]   = np.sqrt(sum(C**2 * np.array(Alpha_errs)**2)) * 1e4
            Kappa_std[PT]   = np.sqrt(sum(C**2 * np.array(Kappa_errs)**2)) * 1e6
            Cp_std[PT]   = np.sqrt(sum(C**2 * np.array(Cp_errs)**2))
            Eps0_std[PT]   = np.sqrt(sum(C**2 * np.array(Eps0_errs)**2))
            Al_std[PT]    = np.sqrt(sum(C**2 * np.array(Al_errs)**2))
            Scd_std[PT]    = np.sqrt(sum(np.mat(C**2) * np.array(Scd_errs)**2))
            LKappa_std[PT]   = np.sqrt(sum(C**2 * np.array(LKappa_errs)**2)) * 1e6

        # Get contributions to the objective function
        X_Rho, G_Rho, H_Rho, RhoPrint = self.objective_term(Points, 'rho', Rho_calc, Rho_std, Rho_grad, name="Density")
        X_Alpha, G_Alpha, H_Alpha, AlphaPrint = self.objective_term(Points, 'alpha', Alpha_calc, Alpha_std, Alpha_grad, name="Thermal Expansion")
        X_Kappa, G_Kappa, H_Kappa, KappaPrint = self.objective_term(Points, 'kappa', Kappa_calc, Kappa_std, Kappa_grad, name="Compressibility")
        X_Cp, G_Cp, H_Cp, CpPrint = self.objective_term(Points, 'cp', Cp_calc, Cp_std, Cp_grad, name="Heat Capacity")
        X_Eps0, G_Eps0, H_Eps0, Eps0Print = self.objective_term(Points, 'eps0', Eps0_calc, Eps0_std, Eps0_grad, name="Dielectric Constant")
        X_Al, G_Al, H_Al, AlPrint = self.objective_term(Points, 'al', Al_calc, Al_std, Al_grad, name="Avg Area per Lipid")
        X_Scd, G_Scd, H_Scd, ScdPrint = self.objective_term(Points, 'scd', Scd_calc, Scd_std, Scd_grad, name="Deuterium Order Parameter")
        X_LKappa, G_LKappa, H_LKappa, LKappaPrint = self.objective_term(Points, 'lkappa', LKappa_calc, LKappa_std, LKappa_grad, name="Bilayer Compressibility")

        Gradient = np.zeros(self.FF.np)
        Hessian = np.zeros((self.FF.np,self.FF.np))

        if X_Rho == 0: self.w_rho = 0.0
        if X_Alpha == 0: self.w_alpha = 0.0
        if X_Kappa == 0: self.w_kappa = 0.0
        if X_Cp == 0: self.w_cp = 0.0
        if X_Eps0 == 0: self.w_eps0 = 0.0
        if X_Al == 0: self.w_al = 0.0
        if X_Scd == 0: self.w_scd = 0.0
        if X_LKappa == 0: self.w_lkappa = 0.0

        if self.w_normalize:
            w_tot = self.w_rho + self.w_alpha + self.w_kappa + self.w_cp + self.w_eps0 + self.w_al + self.w_scd + self.w_lkappa
        else:
            w_tot = 1.0
        w_1 = self.w_rho / w_tot
        w_3 = self.w_alpha / w_tot
        w_4 = self.w_kappa / w_tot
        w_5 = self.w_cp / w_tot
        w_6 = self.w_eps0 / w_tot
        w_7 = self.w_al / w_tot
        w_8 = self.w_scd / w_tot
        w_9 = self.w_lkappa / w_tot

        Objective    = w_1 * X_Rho + w_3 * X_Alpha + w_4 * X_Kappa + w_5 * X_Cp + w_6 * X_Eps0 + w_7 * X_Al + w_8 * X_Scd + w_9 * X_LKappa
        if AGrad:
            Gradient = w_1 * G_Rho + w_3 * G_Alpha + w_4 * G_Kappa + w_5 * G_Cp + w_6 * G_Eps0 + w_7 * G_Al + w_8 * G_Scd + w_9 * G_LKappa
        if AHess:
            Hessian  = w_1 * H_Rho + w_3 * H_Alpha + w_4 * H_Kappa + w_5 * H_Cp + w_6 * H_Eps0 + w_7 * H_Al + w_8 * H_Scd + w_9 * H_LKappa

        if not in_fd():
            self.Xp = {"Rho" : X_Rho, "Alpha" : X_Alpha, 
                           "Kappa" : X_Kappa, "Cp" : X_Cp, "Eps0" : X_Eps0, "Al" : X_Al, "Scd" : X_Scd, "LKappa" : X_LKappa}
            self.Wp = {"Rho" : w_1, "Alpha" : w_3, 
                           "Kappa" : w_4, "Cp" : w_5, "Eps0" : w_6, "Al" : w_7, "Scd" : w_8, "LKappa" : w_9}
            self.Pp = {"Rho" : RhoPrint, "Alpha" : AlphaPrint, 
                           "Kappa" : KappaPrint, "Cp" : CpPrint, "Eps0" : Eps0Print, "Al" : AlPrint, "Scd" : ScdPrint, "LKappa": LKappaPrint}
            if AGrad:
                self.Gp = {"Rho" : G_Rho, "Alpha" : G_Alpha, 
                               "Kappa" : G_Kappa, "Cp" : G_Cp, "Eps0" : G_Eps0, "Al" : G_Al, "Scd" : G_Scd, "LKappa" : G_LKappa}
            self.Objective = Objective

        Answer = {'X':Objective, 'G':Gradient, 'H':Hessian}
        return Answer
