""" @package forcebalance.parser Input file parser for ForceBalance jobs.  Additionally, the location for all default options.

Although I will do my best to write good documentation,
for many programs the input parser becomes the most up-to-date
source for documentation.  So this is a great place to write
lots of comments for those who implement new functionality.

There are two types of sections for options - GENERAL and TARGET.
Since there can be many fitting targets within a single job (i.e. we
may wish to fit water trimers and hexamers, which constitutes two
fitting targets) the input is organized into sections, like so:

$options\n
gen_option_1 Big\n
gen_option_2 Mao\n
$target\n
tgt_option_1 Sniffy\n
tgt_option_2 Schmao\n
$target\n
tgt_option_1 Nifty\n
tgt_option_2 Jiffy\n
$end

In this case, two sets of target options are generated in addition to the general option.

(Note: "Target" used to be called "Simulation".  Backwards compatibility is maintained.)

Each option is meant to be parsed as a certain variable type.

- String option values are read in directly; note that only the first two words in the line are processed
- Some strings are capitalized when they are read in; this is mainly for function tables like OptTab and TgtTab
- List option types will pick up all of the words on the line and use them as values,
plus if the option occurs more than once it will aggregate all of the values.
- Integer and float option types are read in a pretty straightforward way
- Boolean option types are always set to true, unless the second word is '0', 'no', or 'false' (not case sensitive)
- Section option types are meant to treat more elaborate inputs, such
as the user pasting in output parameters from a previous job as input,
or a specification of internal coordinate system.  I imagine that for
every section type I would have to write my own parser.  Maybe a
ParsTab of parsing functions would work. :)

To add a new option, simply add it to the dictionaries below and give it a default value if desired.
If you add an entirely new type, make sure to implement the interpretation of that type in the parse_inputs function.

@author Lee-Ping Wang
@date 11/2012
"""

import os
import re
import sys
import itertools
import traceback
from nifty import printcool, printcool_dictionary, which, isfloat
from copy import deepcopy
from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

## Default general options.
## Note that the documentation is included in part of the key; this will aid in automatic doc-extraction. :)
## In the 5-tuple we have: Default value, priority (larger number means printed first), short docstring, description of scope, list of filter strings for pulling out pertinent targets (MakeInputFile.py)
gen_opts_types = {
    'strings' : {"gmxpath"      : (which('mdrun'), 60, 'Path for GROMACS executables (if not the default)', 'All targets that use GROMACS', ['GMX']),
                 "gmxsuffix"    : ('', 60, 'The suffix of GROMACS executables', 'All targets that use GROMACS', ['GMX']),
                 "tinkerpath"   : (which('testgrad'), 60, 'Path for TINKER executables (if not the default)', 'All targets that use TINKER', ['TINKER']),
                 "penalty_type" : ("L2", 100, 'Type of the penalty, L2 or Hyp in the optimizer', 'All optimizations'),
                 "scan_vals"    : (None, -100, 'Values to scan in the parameter space, given like this: -0.1:0.1:11', 'Job types scan_mvals and scan_pvals'),
                 "readchk"      : (None, -50, 'Name of the restart file we read from', 'Restart jobtype "newton" with "writechk" set'),
                 "writechk"     : (None, -50, 'Name of the restart file we write to (can be same as readchk)', 'Main optimizer'),
                 "ffdir"        : ('forcefield', 100, 'Directory containing force fields, relative to project directory', 'All'),
                 "amoeba_pol"   : (None, 0, 'The AMOEBA polarization type, either direct, mutual, or nonpolarizable.', 'Targets in OpenMM / TINKER that use the AMOEBA force field', ['OPENMM','TINKER']),
                 "amberhome"    : (None, -10, 'Path to AMBER installation directory (leave blank to use AMBERHOME environment variable.', 'Targets that use AMBER', 'AMBER'),
                 },
    'allcaps' : {"jobtype"      : ("single", 200, 'The calculation type, defaults to a single-point evaluation of objective function.',
                                   'All (important); choose "single", "gradient", "hessian", "newton" (Main Optimizer), "bfgs", "powell", "simplex", "anneal", "genetic", "conjugategradient", "scan_mvals", "scan_pvals", "fdcheck[gh]"'),
                 },
    'lists'   : {"forcefield"     : ([],  200, 'The names of force fields, corresponding to directory forcefields/file_name.(itp,xml,prm,frcmod,mol2)', 'All (important)'),
                 "scanindex_num"  : ([], -100, 'Numerical index of the parameter to scan over', 'Job types scan_mvals and scan_pvals'),
                 "scanindex_name" : ([], -100, 'Parameter name to scan over (should convert to a numerical index)', 'Job types scan_mvals and scan_pvals')
                 },
    'ints'    : {"maxstep"      : (100, 50, 'Maximum number of steps in an optimization', 'Main Optimizer'),
                 "objective_history"  : (2, 20, 'Number of good optimization steps to average over when checking the objective convergence criterion', 'Main Optimizer (jobtype "newton")'),
                 "wq_port"   : (0, 0, 'The port number to use for Work Queue', 'Targets that use Work Queue (advanced usage)'),
                 "criteria"   : (1, 160, 'The number of convergence criteria that must be met for main optimizer to converge', 'Main Optimizer'),
                 "rpmd_beads"       : (0, -160, 'Number of beads in ring polymer MD (zero to disable)', 'Condensed phase property targets (advanced usage)', 'liquid_openmm'),
                 "zerograd"         : (-1, 0, 'Set to a nonnegative number to turn on zero gradient skipping at that optimization step.', 'All'),
                 "amber_nbcut"            : (9999, -20, 'Specify the nonbonded cutoff for AMBER engine in Angstrom (I should port this to other engines too.)', 'AMBER targets, especially large nonperiodic systems', ['AMBER'])
                 },
    'bools'   : {"backup"           : (1,  10,  'Write temp directories to backup before wiping them'),
                 "writechk_step"    : (1, -50,  'Write the checkpoint file at every optimization step'),
                 "lq_converge"      : (1, -50,  'Allow convergence on "low quality" steps'),
                 "have_vsite"       : (0, -150, 'Specify whether there are virtual sites in the simulation (being fitted or not).  Enforces calculation of vsite positions.', 'Experimental feature in ESP fitting', ['ABINITIO']),
                 "constrain_charge" : (0,  10,  'Specify whether to constrain the charges on the molecules.', 'Printing the force field (all calculations)'),
                 "print_gradient"   : (1,  20,  'Print the objective function gradient at every step', 'Main Optimizer'),
                 "logarithmic_map"  : (0, -150, 'Optimize in the space of log-variables', 'Creating the force field (all calculations, advanced usage)'),
                 "print_hessian"    : (0,  20, 'Print the objective function Hessian at every step', 'Main Optimizer'),
                 "print_parameters" : (1,  20, 'Print the mathematical and physical parameters at every step', 'Main Optimizer'),
                 "normalize_weights": (1, 100, 'Normalize the weights for the fitting targets', 'Objective function (all calculations)'),
                 "verbose_options"  : (0, 150, 'Set to false to suppress printing options that are equal to their defaults', 'Printing output'),
                 "rigid_water"      : (0, -150, 'Perform calculations using rigid water molecules.', 'Currently used in AMOEBA parameterization (advanced usage)', ['OPENMM','TINKER']),
                 "vsite_bonds"      : (0, -150, 'Generate bonds from virtual sites to host atom bonded atoms.', 'Currently used in AMOEBA parameterization (advanced usage)', ['OPENMM','TINKER']),
                 "use_pvals"        : (0, -150, 'Bypass the transformation matrix and use the physical parameters directly', 'Creating the force field; advanced usage, be careful.'),
                 "asynchronous"     : (0, 0, 'Execute Work Queue tasks and local calculations asynchronously for improved speed', 'Targets that use Work Queue (advanced usage)'),
                 "reevaluate"       : (None, 0, 'Re-evaluate the objective function and gradients when the step is rejected (for noisy objective functions).', 'Main Optimizer'),
                 "continue"         : (0, 140, 'Continue the current run from where we left off (supports mid-iteration recovery).', 'Main Optimizer'),
                 "duplicate_pnames" : (0, -150, 'Allow duplicate parameter names (only if you know what you are doing!', 'Force Field Parser'),
                 },
    'floats'  : {"trust0"                 : (1e-1, 100, 'Levenberg-Marquardt trust radius; set to negative for nonlinear search', 'Main Optimizer'),
                 "mintrust"               : (0.0,   10, 'Minimum trust radius (if the trust radius is tiny, then noisy optimizations become really gnarly)', 'Main Optimizer'),
                 "convergence_objective"  : (1e-4, 100, 'Convergence criterion of objective function (in MainOptimizer this is the stdev of X2 over [objective_history] steps)', 'Main Optimizer'),
                 "convergence_gradient"   : (1e-3, 100, 'Convergence criterion of gradient norm', 'Main Optimizer'),
                 "convergence_step"       : (1e-4, 100, 'Convergence criterion of step size (just needs to fall below this threshold)', 'Main Optimizer'),
                 "eig_lowerbound"         : (1e-4,  10, 'Minimum eigenvalue for applying steepest descent correction', 'Main Optimizer'),
                 "step_lowerbound"        : (1e-6,  10, 'Optimization will "fail" if step falls below this size', 'Main Optimizer'),
                 "lm_guess"               : (1.0,    9, 'Guess value for bracketing line search in trust radius algorithm', 'Main Optimizer'),
                 "finite_difference_h"    : (1e-3,  50, 'Step size for finite difference derivatives in many functions', 'pretty much everywhere'),
                 "finite_difference_factor" : (0.1, 40, 'Make sure that the finite difference step size does not exceed this multiple of the trust radius.', 'Main Optimizer'),
                 "penalty_additive"       : (0.0,   55, 'Factor for additive penalty function in objective function', 'Objective function, all penalty types'),
                 "penalty_multiplicative" : (0.0,   55, 'Factor for multiplicative penalty function in objective function', 'Objective function, all penalty types'),
                 "penalty_alpha"          : (1e-3,  53, 'Extra parameter for fusion penalty function.  Dictates position of log barrier or L1-L0 switch distance',
                                             'Objective function, FUSION_BARRIER or FUSION_L0 penalty type, advanced usage in basis set optimizations'),
                 "penalty_hyperbolic_b"   : (1e-6,  54, 'Cusp region for hyperbolic constraint; for x=0, the Hessian is a/2b', 'Penalty type L1'),
                 "adaptive_factor"        : (0.25,  10, 'The step size is increased / decreased by up to this much in the event of a good / bad step; increase for a more variable step size.', 'Main Optimizer'),
                 "adaptive_damping"       : (0.5,   10, 'Damping factor that ties down the trust radius to trust0; decrease for a more variable step size.', 'Main Optimizer'),
                 "error_tolerance"        : (0.0,   10, 'Error tolerance; the optimizer will only reject steps that increase the objective function by more than this number.', 'Main Optimizer'),
                 "search_tolerance"       : (1e-4, -10, 'Search tolerance; used only when trust radius is negative, dictates convergence threshold of nonlinear search.', 'Main Optimizer with negative mintrust; advanced usage'),
                 "amoeba_eps"             : (None, -10, 'The AMOEBA mutual polarization criterion.', 'Targets in OpenMM / TINKER that use the AMOEBA force field', ['OPENMM','TINKER']),
                 },
    'sections': {"read_mvals" : (None, 100, 'Paste mathematical parameters into the input file for them to be read in directly', 'Restarting an optimization'),
                 "read_pvals" : (None, 100, 'Paste physical parameters into the input file for them to be read in directly', 'Restarting an optimization (recommend use_mvals instead)'),
                 "priors"     : (OrderedDict(), 150, 'Paste priors into the input file for them to be read in directly', 'Scaling and regularization of parameters (important)')
                 }
    }

## Default fitting target options.
tgt_opts_types = {
    'strings' : {"force_map" : ('residue', 0, 'The resolution of mapping interactions to net forces and torques for groups of atoms.  In order of resolution: molecule > residue > charge-group', 'Force Matching', 'AbInitio'),
                 "fragment1" : ('', 0, 'Interaction fragment 1: a selection of atoms specified using atoms and dashes, e.g. 1-6 to select the first through sixth atom (i.e. list numbering starts from 1)', 'Interaction energies', 'Interaction'),
                 "fragment2" : ('', 0, 'Interaction fragment 2: a selection of atoms specified using atoms and dashes, e.g. 7-11 to select atoms 7 through 11.', 'Interaction energies', 'Interaction'),
                 "openmm_precision" : (None, -10, 'Precision of OpenMM calculation if using CUDA or OpenCL platform.  Choose either single, double or mixed ; defaults to the OpenMM default.', 'Targets that use OpenMM', 'OpenMM'),
                 "openmm_platform" : (None, -10, 'OpenMM platform.  Choose either Reference, CUDA or OpenCL.  AMOEBA is on Reference or CUDA only.', 'Targets that use OpenMM', 'OpenMM'),
                 "qdata_txt"             : (None, -10, 'Text file containing quantum data.  If not provided, will search for a default (qdata.txt).', 'Energy/force matching, ESP evaluations, interaction energies', 'TINKER'),
                 "inter_txt"             : ('interactions.txt', 0, 'Text file containing interacting systems.  If not provided, will search for a default.', 'Binding energy target', 'BindingEnergy'),
                 "reassign_modes"        : (None, -180, 'Reassign modes before fitting frequencies, using either linear assignment "permute" or maximum overlap "overlap".', 'Vibrational frequency targets', 'vibration'),
                 "liquid_coords"         : (None, 0, 'Provide file name for condensed phase coordinates.', 'Condensed phase properties', 'Liquid'),
                 "gas_coords"            : (None, 0, 'Provide file name for gas phase coordinates.', 'Condensed phase properties', 'Liquid'),
                 "nvt_coords"         : (None, 0, 'Provide file name for condensed phase NVT coordinates.', 'Condensed phase properties', 'Liquid'),
                 "lipid_coords"         : (None, 0, 'Provide file name for lipid coordinates.', 'Condensed phase properties', 'Lipid'),
                 "coords"                : (None, -10, 'Coordinates for single point evaluation; if not provided, will search for a default.', 'Energy/force matching, ESP evaluations, interaction energies'),
                 "pdb"                   : (None, -10, 'PDB file mainly used for building OpenMM systems but can also contain coordinates.', 'Targets that use OpenMM', 'OpenMM'),
                 "gmx_mdp"               : (None, -10, 'Gromacs .mdp files.  If not provided, will search for default.', 'Targets that use GROMACS', 'GMX'),
                 "gmx_top"               : (None, -10, 'Gromacs .top files.  If not provided, will search for default.', 'Targets that use GROMACS', 'GMX'),
                 "gmx_ndx"               : (None, -10, 'Gromacs .ndx files.  If not provided, will search for default.', 'Targets that use GROMACS', 'GMX'),
                 "amber_mol2"            : (None, -10, 'Name of mol2 file to pass to tleap when setting up AMBER simulations.', 'Targets that use AMBER', 'AMBER'),
                 "amber_frcmod"          : (None, -10, 'Name of frcmod file to pass to tleap when setting up AMBER simulations.', 'Targets that use AMBER', 'AMBER'),
                 "amber_leapcmd"         : (None, -10, 'File containing commands for "tleap" when setting up AMBER simulations.', 'Targets that use AMBER', 'AMBER'),
                 "tinker_key"            : (None, -10, 'TINKER .key files.  If not provided, will search for default.', 'Targets that use TINKER', 'TINKER'),
                 "expdata_txt"           : ('expset.txt', 0, 'Text file containing experimental data.', 'Thermodynamic properties target', 'thermo'),
                 "hfedata_txt"           : ('hfedata.txt', 0, 'Text file containing experimental data.', 'Hydration free energy target', 'hydration'),
                 "hfemode"               : ('single', 0, 'Method for calculating hydration energies (single point, FEP, TI).', 'Hydration free energy target', 'hydration'),
                 "read"                  : (None, 50, 'Provide a temporary directory ".tmp" to read data from a previous calculation on the initial iteration (for instance, to restart an aborted run).', 'Liquid and Remote targets', 'Liquid, Remote'),
                 "remote_prefix"         : ('', 50, 'Specify an optional prefix script to run in front of rtarget.py, for loading environment variables', 'Remote targets', 'Remote'),
                 "fitatoms"              : ('0', 0, 'Number of fitting atoms; defaults to all of them.  Use a comma and dash style list (1,2-5), atoms numbered from one, inclusive', 'Energy + Force Matching', 'AbInitio'),
                 "subset"                : (None, 0, 'Specify a subset of molecules to fit.  The rest are used for cross-validation.', 'Hydration free energy target', 'Hydration'),
                 "gmx_eq_barostat"       : ('berendsen', 0, 'Name of the barostat to use for equilibration.', 'Condensed phase property targets, Gromacs only', 'liquid, lipid'),
                 },
    'allcaps' : {"type"   : (None, 200, 'The type of fitting target, for instance AbInitio_GMX ; this must correspond to the name of a Target subclass.', 'All targets (important)' ,''),
                 "engine" : (None, 180, 'The external code used to execute the simulations (GMX, TINKER, AMBER, OpenMM)', 'All targets (important)', '')
                 },
    'lists'   : {"name"      : ([], 200, 'The name of the target, corresponding to the directory targets/name ; may provide a list if multiple targets have the same settings', 'All targets (important)'),
                 "fd_ptypes" : ([], -100, 'The parameter types that are differentiated using finite difference', 'In conjunction with fdgrad, fdhess, fdhessdiag; usually not needed'),
                 "quantities" : ([], 100, 'List of quantities to be fitted, each must have corresponding Quantity subclass', 'Thermodynamic properties target', 'thermo'),
                 },
    'ints'    : {"shots"              : (-1, 0, 'Number of snapshots; defaults to all of the snapshots', 'Energy + Force Matching', 'AbInitio'),
                 "sleepy"             : (0, -50, 'Wait a number of seconds every time this target is visited (gives me a chance to ctrl+C)', 'All targets (advanced usage)'),
                 "liquid_md_steps"    : (10000, 0, 'Number of time steps for the liquid production run.', 'Condensed phase property targets', 'liquid'),
                 "liquid_eq_steps"    : (1000, 0, 'Number of time steps for the liquid equilibration run.', 'Condensed phase property targets', 'liquid'),
                 "lipid_md_steps"     : (10000, 0, 'Number of time steps for the lipid production run.', 'Condensed phase property targets', 'lipid'),
                 "lipid_eq_steps"     : (1000, 0, 'Number of time steps for the lipid equilibration run.', 'Condensed phase property targets', 'lipid'),
                 "n_mcbarostat"       : (25, 0, 'Number of steps in the liquid simulation between MC barostat volume adjustments.', 'Liquid properties in OpenMM', 'Liquid_OpenMM'),
                 "gas_md_steps"       : (100000, 0, 'Number of time steps for the gas production run, if different from default.', 'Condensed phase property targets', 'liquid'),
                 "gas_eq_steps"       : (10000, 0, 'Number of time steps for the gas equilibration run, if different from default.', 'Condensed phase property targets', 'liquid'),
                 "nvt_md_steps"       : (100000, 0, 'Number of time steps for the liquid NVT production run.', 'Condensed phase property targets', 'liquid'),
                 "nvt_eq_steps"       : (10000, 0, 'Number of time steps for the liquid NVT equilibration run.', 'Condensed phase property targets', 'liquid'),
                 "writelevel"         : (0, 0, 'Affects the amount of data being printed to the temp directory.', 'Energy + Force Matching', 'AbInitio'),
                 "md_threads"         : (1, 0, 'Set the number of threads used by Gromacs or TINKER processes in MD simulations', 'Condensed phase properties in GROMACS and TINKER', 'Liquid_GMX, Lipid_GMX, Liquid_TINKER'),
                 "save_traj"          : (0, -10, 'Whether to save trajectories.  0 = Never save; 1 = Delete if optimization step is good; 2 = Always save', 'Condensed phase properties', 'Liquid, Lipid'),
                 "eq_steps"           : (20000, 0, 'Number of time steps for the equilibration run.', 'Thermodynamic property targets', 'thermo'),
                 "md_steps"           : (50000, 0, 'Number of time steps for the production run.', 'Thermodynamic property targets', 'thermo'),
                 "n_sim_chain"        : (1, 0, 'Number of simulations required to calculate quantities.', 'Thermodynamic property targets', 'thermo'),
                 "n_molecules"        : (-1, 0, 'Provide the number of molecules in the structure (defaults to auto-detect).', 'Condensed phase properties', 'Liquid'),
                 },
    'bools'   : {"fdgrad"           : (0, -100, 'Finite difference gradient of objective function w/r.t. specified parameters', 'Use together with fd_ptypes (advanced usage)'),
                 "fdhess"           : (0, -100, 'Finite difference Hessian of objective function w/r.t. specified parameters', 'Use together with fd_ptypes (advanced usage)'),
                 "fdhessdiag"       : (0, -100, 'Finite difference Hessian diagonals w/r.t. specified parameters (costs 2np times a objective calculation)', 'Use together with fd_ptypes (advanced usage)'),
                 "all_at_once"      : (1, -50, 'Compute all energies and forces in one fell swoop where possible(as opposed to calling the simulation code once per snapshot)', 'Various QM targets and MD codes', 'AbInitio'),
                 "run_internal"     : (1, -50, 'For OpenMM or other codes with Python interface: Compute energies and forces internally', 'OpenMM interface', 'OpenMM'),
                 "energy"           : (1, 0, 'Enable the energy objective function', 'All ab initio targets', 'AbInitio'),
                 "force"            : (1, 0, 'Enable the force objective function', 'All ab initio targets', 'AbInitio'),
                 "resp"             : (0, -150, 'Enable the RESP objective function', 'Ab initio targets with RESP; experimental (remember to set espweight)'),
                 "do_cosmo"         : (0, -150, 'Call Q-Chem to do MM COSMO on MM snapshots.', 'Currently unused, but possible in AbInitio target'),
                 "optimize_geometry": (1, 0, 'Perform a geometry optimization before computing properties', 'Monomer properties', 'moments'),
                 "absolute"         : (0, -150, 'When matching energies in AbInitio, do not subtract the mean energy gap.', 'Energy matching (advanced usage)', 'abinitio'),
                 "cauchy"           : (0, 0, 'Normalize interaction energies each using 1/(denom**2 + reference**2) which resembles a Cauchy distribution', 'Interaction energy targets', 'interaction'),
                 "attenuate"        : (0, 0, 'Normalize interaction energies using 1/(denom**2 + reference**2) only for repulsive interactions greater than denom.', 'Interaction energy targets', 'interaction'),
                 "normalize"        : (0, -150, 'Divide objective function by the number of snapshots / vibrations', 'Interaction energy / vibrational mode targets', 'interaction, vibration'),
                 "w_normalize"      : (0, 0, 'Normalize the condensed phase property contributions to the liquid / lipid property target', 'Condensed phase property targets', 'liquid, lipid'),
                 "manual"           : (0, -150, 'Give the user a chance to fill in condensed phase stuff on the zeroth step', 'Condensed phase property targets (advanced usage)', 'liquid'),
                 "hvap_subaverage"  : (0, -150, 'Don\'t target the average enthalpy of vaporization and allow it to freely float (experimental)', 'Condensed phase property targets (advanced usage)', 'liquid'),
                 "force_cuda"       : (0, -150, 'Force the external npt.py script to crash if CUDA Platform not available', 'Condensed phase property targets (advanced usage)', 'liquid_openmm'),
                 "anisotropic_box"  : (0, -150, 'Enable anisotropic box scaling (e.g. for crystals or two-phase simulations) in external npt.py script', 'Condensed phase property targets (advanced usage)', 'liquid_openmm, liquid_tinker'),
                 "mts_integrator"   : (0, -150, 'Enable multiple-timestep integrator in external npt.py script', 'Condensed phase property targets (advanced usage)', 'liquid_openmm'),
                 "minimize_energy"  : (1, 0, 'Minimize the energy of the system prior to running dynamics', 'Condensed phase property targets (advanced usage)', 'liquid_openmm', 'liquid_tinker'),
                 "remote"           : (0, 50, 'Evaluate target as a remote work_queue task', 'All targets (optional)'),
                 "adapt_errors"     : (0, 50, 'Adapt to simulation uncertainty by combining property estimations and adjusting simulation length.', 'Condensed phase property targets', 'liquid'),
                 "force_average"    : (0, -50, 'Average over all atoms when normalizing force errors.', 'Force matching', 'abinitio'),
                 "remote_backup"    : (0, -50, 'When running remote target, back up files at the remote location.', 'Liquid, lipid and remote targets', 'liquid, lipid, remote'),
                 "pure_num_grad"    : (0, -50, 'Pure numerical gradients -- launch two additional simulations for each perturbed forcefield parameter, and compute derivatives using 3-point formula. (This is very expensive and should only serve as a sanity check)')
                 },
    'floats'  : {"weight"       : (1.0, 150, 'Weight of the target (determines its importance vs. other targets)', 'All targets (important)'),
                 "w_rho"        : (1.0, 0, 'Weight of experimental density', 'Condensed phase property targets', 'liquid, lipid'),
                 "w_hvap"       : (1.0, 0, 'Weight of enthalpy of vaporization', 'Condensed phase property targets', 'liquid, lipid'),
                 "w_alpha"      : (1.0, 0, 'Weight of thermal expansion coefficient', 'Condensed phase property targets', 'liquid, lipid'),
                 "w_kappa"      : (1.0, 0, 'Weight of isothermal compressibility', 'Condensed phase property targets', 'liquid, lipid'),
                 "w_cp"         : (1.0, 0, 'Weight of isobaric heat capacity', 'Condensed phase property targets', 'liquid, lipid'),
                 "w_eps0"       : (1.0, 0, 'Weight of dielectric constant', 'Condensed phase property targets', 'liquid, lipid'),
                 "w_al"         : (1.0, 0, 'Weight of average area per lipid', 'Lipid property targets', 'lipid'),
                 "w_scd"        : (1.0, 0, 'Weight of deuterium order parameter', 'Lipid property targets', 'lipid'),
                 "w_energy"     : (1.0, 0, 'Weight of energy', 'Ab initio targets', 'liquid'),
                 "w_force"      : (1.0, 0, 'Weight of atomistic forces', 'Ab initio targets', 'liquid'),
                 "w_surf_ten"   : (0.0, 0, 'Weight of surface tension', 'Condensed phase property targets', 'liquid'),
                 "w_netforce"   : (0.0, 0, 'Weight of net forces (condensed to molecules, residues, or charge groups)', 'Ab initio targets', 'abinitio'),
                 "w_torque"     : (0.0, 0, 'Weight of torques (condensed to molecules, residues, or charge groups)', 'Ab initio targets', 'abinitio'),
                 "w_resp"       : (0.0, -150, 'Weight of RESP', 'Ab initio targets with RESP (advanced usage)', 'abinitio'),
                 "resp_a"       : (0.001, -150, 'RESP "a" parameter for strength of penalty; 0.001 is strong, 0.0005 is weak', 'Ab initio targets with RESP (advanced usage)', 'abinitio'),
                 "resp_b"       : (0.1, -150, 'RESP "b" parameter for hyperbolic behavior; 0.1 is recommended', 'Ab initio targets with RESP (advanced usage)', 'abinitio'),
                 "energy_upper" : (30.0, 0, 'Upper energy cutoff (in kcal/mol); super-repulsive interactions are given zero weight', 'Interaction energy targets', 'interaction'),
                 "hfe_temperature"  : (298.15, -100, 'Simulation temperature for hydration free energies (Kelvin)', 'Hydration free energy using molecular dynamics', 'hydration'),
                 "hfe_pressure"   : (1.0, -100, 'Simulation temperature for hydration free energies (atm)', 'Hydration free energy using molecular dynamics', 'hydration'),
                 "energy_denom"   : (1.0, 0, 'Energy normalization for binding energies in kcal/mol (default is to use stdev)', 'Binding energy targets', 'binding'),
                 "rmsd_denom"     : (0.1, 0, 'RMSD normalization for optimized geometries in Angstrom', 'Binding energy targets', 'binding'),
                 "wavenumber_tol" : (10.0, 0, 'Frequency normalization (in wavenumber) for vibrational frequencies', 'Vibrational frequency targets', 'vibration'),
                 "dipole_denom"   : (1.0, 0, 'Dipole normalization (Debye) ; set to 0 if a zero weight is desired', 'Monomer property targets', 'monomer'),
                 "quadrupole_denom"   : (1.0, 0, 'Quadrupole normalization (Buckingham) ; set to 0 if a zero weight is desired', 'Monomer property targets', 'monomer'),
                 "polarizability_denom"   : (1.0, 0, 'Dipole polarizability tensor normalization (cubic Angstrom) ; set to 0 if a zero weight is desired', 'Monomer property targets with polarizability', 'monomer'),
                 "liquid_timestep"  : (1.0, 0, 'Time step size for the liquid simulation.', 'Condensed phase property targets', 'liquid'),
                 "liquid_interval"  : (0.1, 0, 'Time interval for saving coordinates for the liquid production run.', 'Condensed phase property targets', 'liquid'),
                 "gas_timestep"  : (1.0, 0, 'Time step size for the gas simulation (if zero, use default in external script.).', 'Condensed phase property targets', 'liquid'),
                 "gas_interval"  : (0.1, 0, 'Time interval for saving coordinates for the gas production run (if zero, use default in external script.)', 'Condensed phase property targets', 'liquid'),
                 "lipid_timestep"  : (1.0, 0, 'Time step size for the lipid simulation.', 'Lipid property targets', 'lipid'),
                 "lipid_interval"  : (0.1, 0, 'Time interval for saving coordinates for the lipid production run.', 'Lipid property targets', 'lipid'),
                 "nvt_timestep"  : (1.0, 0, 'Time step size for the NVT simulation.', 'Condensed phase property targets', 'liquid'),
                 "nvt_interval"  : (0.1, 0, 'Time interval for saving coordinates for the NVT simulation production run.', 'Condensed phase property targets', 'liquid'),
                 "self_pol_mu0"  : (0.0, -150, 'Gas-phase dipole parameter for self-polarization correction (in debye).', 'Condensed phase property targets', 'liquid'),
                 "self_pol_alpha"  : (0.0, -150, 'Polarizability parameter for self-polarization correction (in debye).', 'Condensed phase property targets', 'liquid'),
                 "epsgrad"         : (0.0, -150, 'Gradient below this threshold will be set to zero.', 'All targets'),
                 "energy_asymmetry": (1.0, -150, 'Snapshots with (E_MM - E_QM) < 0.0 will have their weights increased by this factor.', 'Ab initio targets'),
                 "nonbonded_cutoff"  : (None, -1, 'Cutoff for nonbonded interactions (passed to engines).', 'Condensed phase property targets', 'liquid'),
                 "vdw_cutoff"        : (None, -2, 'Cutoff for vdW interactions if different from other nonbonded interactions', 'Condensed phase property targets', 'liquid'),
                 "liquid_fdiff_h" : (1e-2, 0, 'Step size for finite difference derivatives for liquid targets in pure_num_grad', 'Condensed phase property targets', 'liquid'),
                 },
    'sections': {}
    }

all_opts_names = list(itertools.chain(*[i.keys() for i in gen_opts_types.values()])) + list(itertools.chain(*[i.keys() for i in tgt_opts_types.values()]))
## Check for uniqueness of option names.
for i in all_opts_names:
    iocc = []
    for typ, dct in gen_opts_types.items():
        if i in dct:
            iocc.append("gen_opt_types %s" % typ)
    for typ, dct in tgt_opts_types.items():
        if i in dct:
            iocc.append("gen_opt_types %s" % typ)
    if len(iocc) != 1:
        logger.error("CODING ERROR: ForceBalance option %s occurs in more than one place (%s)\n" % (i, str(iocc)))
        raise RuntimeError

## Default general options - basically a collapsed veresion of gen_opts_types.
gen_opts_defaults = {}
for t in gen_opts_types:
    subdict = {}
    for i in gen_opts_types[t]:
        subdict[i] = gen_opts_types[t][i][0]
    gen_opts_defaults.update(subdict)

## Default target options - basically a collapsed version of tgt_opts_types.
tgt_opts_defaults = {}
for t in tgt_opts_types:
    subdict = {}
    for i in tgt_opts_types[t]:
        subdict[i] = tgt_opts_types[t][i][0]
    tgt_opts_defaults.update(subdict)

## Option maps for maintaining backward compatibility.
bkwd = {"simtype" : "type",
        "masterfile" : "inter_txt",
        "openmm_cuda_precision" : "openmm_precision",
        "mdrun_threads" : "md_threads",
        "mts_vvvr" : "mts_integrator",
        "amoeba_polarization" : "amoeba_pol",
        "liquid_prod_steps" : "liquid_md_steps",
        "gas_prod_steps" : "gas_md_steps",
        "liquid_equ_steps" : "liquid_eq_steps",
        "gas_equ_steps" : "gas_eq_steps",
        "lipid_prod_steps" : "lipid_md_steps",
        "lipid_equ_steps" : "lipid_eq_steps",
        }

## Listing of sections in the input file.
mainsections = ["SIMULATION","TARGET","OPTIONS","END","NONE"]

def read_mvals(fobj):
    Answer = []
    for line in fobj:
        if re.match("(/read_mvals)|(^\$end)",line):
            break
        Answer.append(float(line.split('[')[-1].split(']')[0].split()[-1]))
    return Answer

def read_pvals(fobj):
    Answer = []
    for line in fobj:
        if re.match("(/read_pvals)|(^\$end)",line):
            break
        Answer.append(float(line.split('[')[-1].split(']')[0].split()[-1]))
    return Answer

def read_priors(fobj):
    Answer = OrderedDict()
    for line in fobj:
        line = line.split("#")[0]
        if re.match("(/priors)|(^\$end)",line):
            break
        Answer[line.split()[0]] = float(line.split()[-1])
    return Answer

def read_internals(fobj):
    return

## ParsTab that refers to subsection parsers.
ParsTab  = {"read_mvals" : read_mvals,
            "read_pvals" : read_pvals,
            "priors"     : read_priors,
            "internal"   : read_internals
            }

def printsection(heading,optdict,typedict):
    """ Print out a section of the input file in a parser-compliant and readable format.

    At the time of writing of this function, it's mainly intended to be called by MakeInputFile.py.
    The heading is printed first (it is something like $options or $target).  Then it loops
    through the variable types (strings, allcaps, etc...) and the keys in each variable type.
    The one-line description of each key is printed out as a comment, and then the key itself is
    printed out along with the value provided in optdict.  If optdict is None, then the default
    value is printed out instead.

    @param[in] heading Heading, either $options or $target
    @param[in] optdict Options dictionary or None.
    @param[in] typedict Option type dictionary, either gen_opts_types or tgt_opts_types specified in this file.
    @return Answer List of strings for the section that we are printing out.

    """
    from forcebalance.objective import Implemented_Targets
    from forcebalance.optimizer import Optimizer

    def FilterTargets(search):
        if type(search) == str:
            search = [search]
        list_out = []
        for key in sorted(Implemented_Targets.keys()):
            if any([i.lower() in key.lower() for i in search]):
                list_out.append(Implemented_Targets[key].__name__)
        return ', '.join(sorted(list_out))

    Answer = [heading]
    firstentry = 1
    Options = []
    for i in ['strings','allcaps','lists','ints','bools','floats','sections']:
        vartype = re.sub('s$','',i)
        for j in typedict[i]:
            Option = []
            val = optdict[j] if optdict is not None else typedict[i][j][0]
            if firstentry:
                firstentry = 0
            else:
                Option.append("")
            Priority = typedict[i][j][1]
            Option.append("# (%s) %s" % (vartype, typedict[i][j][2]))
            if len(typedict[i][j]) >= 4:
                Relevance = typedict[i][j][3]
                str2 = "# used in: %s" % Relevance
                if len(typedict[i][j]) >= 5:
                    TargetName = FilterTargets(typedict[i][j][4])
                    str2 += " (%s)" % TargetName
                else:
                    TargetName = "None"
                Option.append(str2)
            else:
                Relevance = "None"
            Option.append("%s %s" % (str(j),str(val)))
            Options.append((Option, Priority, TargetName, j))
    def key1(o):
        return o[1]
    def key2(o):
        return o[2]
    def key3(o):
        return o[3]
    Options.sort(key=key3)
    Options.sort(key=key2)
    Options.sort(key=key1, reverse=True)
    for o in Options:
        Answer += o[0]

    # PriSet = sorted(list(set(Priorities)))[::-1]
    # TgtSet = sorted(list(set(TargetNames)))
    # RelSet = sorted(list(set(Relevances)))
    # for p0 in PriSet:
    #     ogrp = []
    #     rgrp = []
    #     tgrp = []
    #     for o, p, r, t in zip(Options, Priorities, Relevances, TargetNames):
    #         if p == p0:
    #             ogrp.append(o)
    #             rgrp.append(r)
    #             tgrp.append(t)
    #     ogrp2 = []
    #     rgrp2 = []
    #     for t0 in TgtSet:
    #         for o, r, t in zip(ogrp, rgrp, tgrp):
    #             if t == t0:
    #                 ogrp2.append(

    Answer.append("$end")
    return Answer

def parse_inputs(input_file=None):
    """ Parse through the input file and read all user-supplied options.

    This is usually the first thing that happens when an executable script is called.
    Our parser first loads the default options, and then updates these options as it
    encounters keywords.

    Each keyword corresponds to a variable type; each variable type (e.g. string,
    integer, float, boolean) is treated differently.  For more elaborate inputs,
    there is a 'section' variable type.

    There is only one set of general options, but multiple sets of target options.
    Each target has its own section delimited by the \em $target keyword,
    and we build a list of target options.

    @param[in]  input_file The name of the input file.
    @return     options    General options.
    @return     tgt_opts   List of fitting target options.

    @todo Implement internal coordinates.
    @todo Implement sampling correction.
    @todo Implement charge groups.
    """

    logger.info("Reading options from file: %s\n" % input_file)
    section = "NONE"
    # First load in all of the default options.
    options = deepcopy(gen_opts_defaults) # deepcopy to make sure options doesn't make changes to gen_opts_defaults
    options['root'] = os.getcwd()
    options['input_file'] = input_file
    tgt_opts = []
    this_tgt_opt = deepcopy(tgt_opts_defaults)
    # Give back a bunch of default options if input file isn't specified.
    if input_file is None:
        return (options, [this_tgt_opt])
    fobj = open(input_file)
    for line in fobj:
        try:
            # Anything after "#" is a comment
            line = line.split("#")[0].strip()
            s = line.split()
            # Skip over blank lines
            if len(s) == 0:
                continue
            key = s[0].lower()
            if key in bkwd: # Do option replacement for backward compatibility.
                key = bkwd[key]
            # If line starts with a $, this signifies that we're in a new section.
            if re.match('^\$',line):
                newsection = re.sub('^\$','',line).upper()
                if section in ["SIMULATION","TARGET"] and newsection in mainsections:
                    tgt_opts.append(this_tgt_opt)
                    this_tgt_opt = deepcopy(tgt_opts_defaults)
                if newsection == "END": newsection = "NONE"
                section = newsection
            elif section in ["OPTIONS","SIMULATION","TARGET"]:
                ## Depending on which section we are in, we choose the correct type dictionary
                ## and add stuff to 'options' and 'this_tgt_opt'
                (this_opt, opts_types) = (options, gen_opts_types) if section == "OPTIONS" else (this_tgt_opt, tgt_opts_types)
                ## Note that "None" is a special keyword!  The variable will ACTUALLY be set to None.
                if len(s) > 1 and s[1].upper() == "NONE":
                    this_opt[key] = None
                elif key in opts_types['strings']:
                    this_opt[key] = s[1]
                elif key in opts_types['allcaps']:
                    this_opt[key] = s[1].upper()
                elif key in opts_types['lists']:
                    for word in s[1:]:
                        this_opt.setdefault(key,[]).append(word)
                elif key in opts_types['ints']:
                    if isfloat(s[1]):
                        this_opt[key] = int(float(s[1]))
                    else:
                        this_opt[key] = int(s[1])
                elif key in opts_types['bools']:
                    if len(s) == 1:
                        this_opt[key] = True
                    elif s[1].upper() in ["0", "NO", "FALSE", "OFF"]:
                        this_opt[key] = False
                    elif isfloat(s[1]) and int(float(s[1])) == 0:
                        this_opt[key] = False
                    elif s[1].upper() in ["1", "YES", "TRUE", "ON"]:
                        this_opt[key] = True
                    elif isfloat(s[1]) and int(float(s[1])) == 1:
                        this_opt[key] = True
                    else:
                        logger.error('%s is a true/false option but you provided %s; to enable, provide ["1", "yes", "true", "on" or <no value>].  To disable, provide ["0", "no", "false", or "off"].\n' % (key, s[1]))
                        raise RuntimeError
                elif key in opts_types['floats']:
                    this_opt[key] = float(s[1])
                elif key in opts_types['sections']:
                    this_opt[key] = ParsTab[key](fobj)
                else:
                    logger.error("Unrecognized keyword: --- \x1b[1;91m%s\x1b[0m --- in %s section\n" \
                          % (key, section))
                    logger.error("Perhaps this option actually belongs in %s section?\n" \
                          % (section == "OPTIONS" and "a TARGET" or "the OPTIONS"))
                    raise RuntimeError
            elif section == "NONE" and len(s) > 0:
                logger.error("Encountered a non-comment line outside of a section\n")
                raise RuntimeError
            elif section not in mainsections:
                logger.error("Unrecognized section: %s\n" % section)
                raise RuntimeError
        except:
            # traceback.print_exc()
            logger.exception("Failed to read in this line! Check your input file.\n")
            logger.exception('\x1b[91m' + line + '\x1b[0m\n')
            raise RuntimeError
    if section == "SIMULATION" or section == "TARGET":
        tgt_opts.append(this_tgt_opt)
    if not options['verbose_options']:
        printcool("Options at their default values are not printed\n Use 'verbose_options True' to Enable", color=5)
    # Expand target options (i.e. create multiple tgt_opts dictionaries if multiple target names are specified)
    tgt_opts_x = []
    for topt in tgt_opts:
        for name in topt['name']:
            toptx = deepcopy(topt)
            toptx['name'] = name
            tgt_opts_x.append(toptx)
    return options, tgt_opts_x
