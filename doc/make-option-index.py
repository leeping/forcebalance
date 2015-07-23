#!/usr/bin/env python

from forcebalance import parser, optimizer
from forcebalance.objective import Implemented_Targets
import re

""" The OptionDoc dictionary is a key : dictionary dictionary.
Each variable can have several parts of the documentation: 'pert' = 
"""

GenOptionDoc = {"gmxpath" : {"scope" : "Targets that use GROMACS",
                             "required" : True,
                             "long" : """Specify the path where GROMACS executables are installed, most likely ending in 'bin'.
                             Note that executables are only installed 'bin' if the program is installed using 'make install';
                             this will NOT be the case if you simply ran 'make'.""",
                             "recommend" : "Depends on your local installation and environment."
                             },
                "tinkerpath" : {"scope" : "Targets that use TINKER",
                                "required" : True,
                                "recommend" : "Depends on your local installation and environment."
                                },
                "gmxsuffix" : {"scope" : "Targets that use GROMACS",
                               "required" : False,
                               "long" : """Depending on how GROMACS is configured and installed, a suffix may be appended to executable
                               names.  If there is a suffix, it needs to be specified here (or else ForceBalance will not find the
                               GROMACS executable and it will crash.""",
                               "recommend" : "Depends on your local installation and environment."
                               },
                "penalty_type" : {"scope" : "All force field optimizations",
                                  "required" : False,
                                  "long" : """To prevent the optimization from changing the parameters too much, an additional penalty
                                  is applied to the objective function that depends linearly (L1) or quadratically (L2) on the norm
                                  of the parameter displacement vector.  L1 corresponds to LASSO regularization while L2 is known as
                                  Tikhonov regularization or ridge regression.""",
                                  "recommend" : "L2; tested and known to be working.  Implementation of L1 in progress."
                                  },
                "scan_vals" : {"scope" : "scan_mvals and scan_pvals job types",
                               "required" : False,
                               "long" : """This specifies a range of parameter values to scan in a uniform grid.  scan_mvals works in
                               the mathematical parameter space while scan_pvals works in the physical parameter space.  The syntax
                               is lower:upper:nsteps (just like numpy.linspace) .  Both lower and upper limits are included in the range.""",
                               "recommend" : "For scan_mvals, a range of values between -1 and +1 is recommended; for scan_pvals, choose values close to the physical parameter value."
                               },
                "scanindex_num" : {"scope" : "scan_mvals and scan_pvals job types",
                                   "required" : False,
                                   "long" : """ForceBalance assigns to each adjustable parameter a 'parameter number' corresponding to
                                   its position in the parameter vector.  This tells the parameter scanner which number to scan over.""",
                                   "recommend" : "Look at the printout from a single-point job to decide which parameter number you wish to scan over."
                                   },
                "scanindex_name" : {"scope" : "scan_mvals and scan_pvals job types",
                                    "required" : False,
                                    "long" : """ForceBalance assigns to each adjustable parameter a 'parameter name'.  By specifying
                                    this option, this tells the parameter scanner to locate the correct parameter with the specified name and then
                                    scan over it.""",
                                    "recommend" : "Look at the printout from a single-point job to determine the parameter names."
                                    },
                "maxstep" : {"scope" : "All iterative optimization jobs",
                             "required" : False,
                             "recommend" : "At least 100 optimization steps are recommended."
                             },
                "readchk" : {"scope" : "Main optimizer",
                             "required" : False,
                             "long" : """The main optimizer has the ability to pick up where it left off by reading / writing checkpoint
                             files.  Here you may specify the checkpoint file to read in from a previous optimization run.  This is
                             equivalent to reading in stored parameter values, except the gradient and Hessian (which contains memory from previous
                             steps) is recorded too."""
                             },
                "writechk" : {"scope" : "Main optimizer",
                              "required" : False,
                              "long" : """The main optimizer has the ability to pick up where it left off by reading / writing checkpoint
                              files.  Here you may specify the checkpoint file to write after the job is finished.""",
                              "recommend" : "Writing the checkpoint file is highly recommended."
                              },
                "writechk_step" : {"scope" : "Main optimizer when 'writechk' is turned on",
                                   "required" : False,
                                   "long" : """Write a checkpoint file every single step, not just after the job is finished.""",
                                   "recommend" : "Useful if you want to quit an optimization before it finishes and restart, but make sure you don't overwrite existing checkpoint files by accident."
                                   },
                "ffdir" : {"scope" : "All force field optimizations",
                           "required" : False,
                           "recommend" : "Unless you're using a nonstandard location for force field files, you probably shouldn't change this."
                           },
                "jobtype" : {"scope" : "All force field optimizations",
                             "required" : True,
                             "long" : """Here you may specify the type of ForceBalance job.  This ranges from gradient-based and stochastic
                             optimizations to simple scans over the parameter space and finite difference checking of gradients.""",
                             "recommend" : "See the Optimizer class documentation for which optimizer is best suited for you."
                             #"recommend" : "See the Optimizer class documentation for which optimizer is best suited for you.  Available options are: %s" % ', '.join(sorted([i for i in optimizer.Optimizer.OptTab]))
                             },
                "forcefield" : {"scope" : "All force field optimizations",
                                "required" : True
                                },
                "backup" : {"scope" : "All force field optimizations",
                            "required" : False
                            },
                "trust0" : {"scope" : "Main optimizer",
                            "required" : False,
                            "long" : """The main optimizer uses a trust radius which 'adapts' (i.e. increases or decreases) based on whether the last step
                            was a good or bad step.  'trust0' provides the starting trust radius, and the trust radius is not allowed to increase too much
                            from trust0.""",
                            "recommend" : "Increase from the default if the optimizer takes many good steps but takes too long; decrease if the optimizer takes many bad steps."
                            },
                "convergence_objective" : {"scope" : "Main optimizer",
                                           "required" : False,
                                           "long" : """The main optimizer will quit when the last ten good values of the objective function have a
                                           standard deviation that falls below this number.  We use the last ten good values (instead of the latest
                                           change in the objective function), otherwise this condition would be triggered by taking tiny steps.""",
                                           "recommend" : "Decrease this value if it's being triggered by small step sizes."
                                           },
                "convergence_step" : {"scope" : "Main optimizer",
                                      "required" : False,
                                      "long" : """The main optimizer will quit when the step size falls below this number.  This happens if we are
                                      approaching a local minimum, or if the optimizer is constantly taking bad steps and the trust radius is
                                      reduced until it falls below this number.  In the latter case, this usually means that the derivatives are
                                      wrong.""",
                                      "recommend" : "Make sure that this value is much smaller than trust0."
                                      },
                "convergence_gradient" : {"scope" : "Main optimizer",
                                          "required" : False,
                                          "long" : """The main optimizer will quit when the objective function gradient falls below this number.
                                          Since this is a newly implemented option, I can't say when this option will fail.""",
                                          "recommend" : "Leave at the default, or set to several orders of magnitude below a typical value of the gradient (perhaps the gradient at the start of the optimization.)"
                                          },
                "eig_lowerbound" : {"scope" : "Main optimizer",
                                    "required" : False,
                                    "long" : """The main optimizer will misbehave if there are negative or very small eigenvalues in the
                                    objective function Hessian.  In the former case the optimizer will travel toward a saddle point (or
                                    local maximum), and in the latter case the matrix inversion will fail because of the matrix singularity.
                                    If the smallest eigenvalue is below this value, then a multiple of the identity matrix is added to the
                                    Hessian to increase the smallest eigenvalue to at least this value.""",
                                    "recommend" : "Shouldn't have to worry about this setting, unless the optimizer appears to be taking bad steps or inverting nearly singular matrices."
                                    },
                
                "finite_difference_h" : {"scope" : "fdcheck_G or fdcheck_H job types, or whenever the objective function is evaluated using finite difference",
                                         "required" : False,
                                         "long" : """When the objective function derivatives are checked using finite difference, or when the objective function derivative
                                         requires finite difference, this is the step size that is used (in the mathematical space).  The actual parameter in the
                                         force field is changed by this amount times the rescaling factor.""",
                                         "recommend" : """1e-2 to 1e-4; run FDCheckG to see if derivatives are accurate; if derivatives are inaccurate then adjust accordingly.
                                         If the objective function itself requires finite difference, there will still be a difference because FDCheckG(H) uses an accurate
                                         seven-point (five-point) stencil.  Make sure that the derivatives agree before settling on a value to use."""
                                         },
                
                "penalty_additive" : {"scope" : "Objective function",
                                      "required" : False,
                                      "long" : """Add a penalty to the objective function (e.g. L2 or L1 norm) with this prefactor.
                                      Using an additive penalty requires an assessment of the order of magnitude of the objective function,
                                      but it is closer to the statistical concept of ridge or LASSO regression.""",
                                      "recommend" : """No recommendation; run a single-point calculation to choose a prefactor.  Consider 0.01
                                      for an objective function of order 1."""
                                      },
                
                "penalty_multiplicative" : {"scope" : "Objective function",
                                            "required" : False,
                                            "long" : """Multiply the objective function by (1+X) where X is this value.
                                            Using an multiplicative penalty works well for objective functions of any size but it is not
                                            equivalent to statistical regularization methods.""",
                                            "recommend" : """A value of 0.01 tends to keep the length of the parameter vector from exceeding 1."""
                                            },
                
                "amoeba_polarization" : {"scope" : "Optimizations of the AMOEBA polarizable force field",
                                         "required" : False,
                                         "long" : """When optimizing a force field with the AMOEBA functional form, 
                                         set this variable to 'direct', 'mutual', or 'nonpolarizable'.  At present 
                                         this variable affects OpenMM API calls, but not TINKER input files.""",
                                         "recommend" : """No recommendation, depends on the application."""
                                         },

                "adaptive_factor" : {"scope" : "Main optimizer",
                                      "required" : False,
                                      "long" : """Adaptive adjustment of the step size in trust-radius Newton Raphson.
                                      If the optimizer takes a good step, the step is increased as follows:
@verbatim
trust += adaptive_factor*trust*np.exp(-adaptive_damping*(trust/self.trust0 - 1))
@endverbatim
                                      Note that the adaptive_damping option makes sure that the trust radius increases by a smaller factor
                                      the further it deviates from the original trust radius (trust0).
                                      On the other hand, if the optimizer takes a bad step, the step is reduced as follows:
@verbatim
trust = max(ndx*(1./(1+adaptive_factor)), self.mintrust)
@endverbatim
""",
                                      "recommend" : """0.2 is a conservative value, 0.5 for big step size adjustments."""
                                      },

                "adaptive_damping" : {"scope" : "Main optimizer",
                                      "required" : False,
                                      "long" : """See documentation for adaptive_factor.""",
                                      "recommend" : """A larger value will ensure that the trust radius never exceeds the original value by more
                                      than a small percentage.  0.5 is a reasonable value to start from."""
                                      },

                "asynchronous" : {"scope" : "Targets that use Work Queue",
                                  "required" : False,
                                  "long" : """When using Work Queue to distribute computationally intensive tasks (e.g. condensed phase simulations), 
                                  it is computationally efficient to run the local jobs concurrently rather than wait for the tasks to finish.  Setting
                                  this flag allows local evaluation of targets to proceed while the Work Queue runs in the background, which speeds up
                                  the calculation compared to waiting idly for the Work Queue tasks to complete.""",
                                  "recommend" : """If using Work Queue to distribute tasks for some targets, set to True."""
                                  },

                "constrain_charge" : {"scope" : "Force fields with point charges",
                                      "required" : False,
                                      "long" : """It is important for force fields with point charges to not change the overall charge on the molecule or ion.  Setting this option
                                      will activate a linear transformation which projects out the direction in parameter space that changes the net charge.""",
                                      "recommend" : """Either set to true and check your output carefully, or use "eval" statements in the force field file for finer control."""
                                  },

                "error_tolerance" : {"scope" : "Main optimizer",
                                     "required" : False,
                                     "long" : """In some targets (e.g. condensed phase properties), the contribution to the objective function may contain statistical noise
                                     and cause the optimization step to be rejected.  Introducing an error tolerance allows the optimization to continue despite some apparent
                                     roughness in the objective function surface.""",
                                     "recommend" : """Set to zero for targets that don't have statistical noise.  
                                     Otherwise, choose a value based on the rough size of the objective function and the weight of the statistically noisy targets."""
                                  },
                
                "read_mvals" : {"scope" : "All force field optimizations",
                                "required" : False,
                                "long" : """Read in mathematical parameters before starting the optimization.  There is a standardized syntax, given by:
@verbatim read_mvals
0 [ -2.9766e-01 ] : VDWSOW
1 [  2.2283e-01 ] : VDWTOW
2 [ -1.1138e-03 ] : BONDSBHWOW
3 [ -9.0883e-02 ] : BONDSKHWOW
\\read_mvals @endverbatim""",
                                "recommend" : """If you run the main optimizer, it will print out this block at the very end for you to use and/or modify."""
                                },
                
                "read_pvals" : {"scope" : "All force field optimizations",
                                "required" : False,
                                "long" : """Read in physical parameters before starting the optimization.  There is a standardized
                                syntax, given by:
@verbatim read_pvals
 0 [  2.9961e-01 ] : VDWSOW
 1 [  1.2009e+00 ] : VDWTOW
 2 [  9.5661e-02 ] : BONDSBHWOW
 3 [  4.1721e+05 ] : BONDSKHWOW
 \\read_pvals @endverbatim
                                These are the actual numbers that go into the force field file, so note the large changes in magnitude.""",
                                "recommend" : """If you run the main optimizer, it will print out this block at the very end for you to use and/or modify."""
                                },
                }

TgtOptionDoc = {"name" : {"scope" : "All targets",
                          "required" : True,
                          "recommend" : """Choose a descriptive name and make sure all targets have different names."""
                          },
                "type" : {"scope" : "All targets",
                             "required" : True,
                             "long" : """This is the type of target that you are running.  The current accepted values for the target type
                             are given in the beginning of the objective.py file: %s.""" % ', '.join([i for i in Implemented_Targets]),
                             "recommend" : "Choose the appropriate type, and if the target type is missing, feel free to implement your own (or ask me for help)."
                             },
                "fd_ptypes" : {"scope" : "All target types",
                             "required" : False,
                               "long" : """To compute the objective function derivatives, some components may require numerical finite difference in the derivatives.
                             Here you may specify the parameter types that finite difference is applied to,
                             or write 'ALL' to take finite-difference derivatives in all parameter types.""",
                             "recommend" : """If you aren't sure, either use 'ALL' to do finite difference in each component (this is costly), or run a fdcheckG(H)
                             job with this option set to 'NONE' to check which analytic derivatives are missing."""
                             },
                "shots" : {"scope" : "Force and energy matching",
                           "required" : False,
                           "long" : """This option allows you to choose a subset from the snapshots available in the force matching 'targets' directory.
                           The subset is simply taken from the front of the trajectory.
                           In the future this option will be expanded to allow a random selection of snapshots, or a specific selection""",
                           "recommend" : """100-10,000 snapshots are recommended.  Note that you need at least 3x (number of atoms) if
                           the covariance matrix is turned on."""
                           },
                "fitatoms" : {"scope" : "Force and energy matching",
                              "required" : False,
                              "long" : """Choose a subset of atoms to fit forces to.  This is useful in situations where
                              it is undesirable to fit the forces on part of the system (e.g. the part that is described by another force field.)
                              Currently, you are only allowed to choose from the atoms in the front of the trajectory;
                              soon this will be expanded for random flexibility (see 'shots').  However, random coordinate selections are not allowed. ;)""",
                              "recommend" : """Situation-dependent; this should be based on the part of the system that you're fitting, or leave blank
                              if you're fitting the whole system."""
                              },
                "whamboltz" : {"scope" : "Force and energy matching",
                               "required" : False,
                               "long" : """In self-consistent energy/force matching projects, the data from previous cycles can be reused by applying the
                               Weighted Histogram Analysis Method (WHAM).  However, the WHAM data is currently generated by external scripts that
                               haven't made it into this distribution yet.  In the future, generation of WHAM data will be incorporated into this
                               program automatically.""",
                               "recommend" : """Leave off unless you have an externally generated wham-master.txt and wham-weights.txt files."""
                               },
                "covariance" : {"scope" : "Force and energy matching",
                                "required" : False,
                                "long" : """The components of the energy and force contribution to the objective function are rescaled to be on the 
                                same footing when the objective function is optimized.  This can be done by dividing each component by its variance,
                                or by multiplying the energy-force polytensor by the inverse of the quantum energy-force covariance matrix.  The
                                latter method was proposed as a way to emphasize intermolecular interactions but it is unproven.""",
                                "recommend" : """No recommendation; turn the covariance off if the number of snapshots is not much larger than
                                the number of coordinates."""
                                },
                "batch_fd" : {"scope" : "All target types",
                              "required" : False,
                              "long" : """This is a stub for future functionality.  When the flag is switched on, the jobs corresponding to finite
                              difference derivatives are evaluated in parallel on a distributed computing platform."""
                              },
                "fdgrad" : {"scope" : "All target types",
                            "required" : False,
                            "long" : """When this option is enabled, finite difference gradients will be enabled for selected parameter types 
                            (using the fd_ptypes option).  Gradients are computed using two-point finite difference of the objective function.""",
                            "recommend" : """If analytic derivatives are implemented (and correct), then they are much faster than finite difference
                            derivatives.  Run the 'fdcheckG' routine with this option set to Off to check which finite difference derivatives you need."""
                            },
                "fdhess" : {"scope" : "All target types",
                            "required" : False,
                            "long" : """When this option is enabled, finite difference Hessians will be enabled for selected parameter types 
                            (using the fd_ptypes option).  Hessians are computed using two-point finite difference of the gradient.""",
                            "recommend" : """Run the 'fdcheckH' routine with this option set to Off to check which finite difference Hessian elements you need.
                            Note that this requires a very large number of objective function evaluations, so use sparingly."""
                            },
                "fdhessdiag" : {"scope" : "All target types",
                                "required" : False,
                                "long" : """When this option is enabled, finite difference gradients and Hessian diagonal elements will be enabled 
                                for selected parameter types (using the fd_ptypes option).  This is done using a three-point finite difference of
                                the objective function.""",
                                "recommend" : """Use this as a substitute for 'fdgrad'; it doubles the cost but provides more accurate derivatives
                                plus the Hessian diagonal values (these are very nice for quasi-Newton optimizers like BFGS)."""
                                },
                "use_pvals" : {"scope" : "All target types",
                               "required" : False,
                               "long" : """When this option is enabled, the coordinate transformation in parameter space will be bypassed, and 
                               parameters passed into the 'get' subroutines will be plugged directly into the force field files.  This
                               option is turned on automatically if we are running a 'scan_pvals' job.  Note that the coordinate transformation
                               is essential in multi-parameter optimizations and preserves the condition number of the Hessian, so turning it off
                               should generally be avoided.""",
                               "recommend" : """This option should almost always be off unless the user really knows what he/she is doing."""
                               },
                "weight" : {"scope" : "All target types",
                            "required" : False,
                            "long" : """This option specifies the weight that the target will contribute to the objective function.
                            A larger weight for a given target means that the optimizer will prioritize it over the others.
                            When several targets are used, the weight should be chosen carefully such that all targets
                            contribute a finite amount to the objective function.  Note that the choice of weight determines the final outcome
                            of the force field, although we hope not by too much.""",
                            "recommend" : """It is important to specify something here (giving everything equal weight is unlikely to work.)  
                            Run a single-point objective function evaluation with all weights set to one to get a handle on
                            the natural size of each target's contribution, and then add weights accordingly."""
                            },
                "efweight" : {"scope" : "Force and energy matching",
                              "required" : False,
                              "long" : """Energies and forces are evaluated together in a force/energy matching target, and this option
                              specifies the relative weight of the energy and force contributions.""",
                              "recommend" : """Based on experience, it should be okay to leave this option at its default value, unless you wish
                              to emphasize only the force (then choose 0.0) or only the energy (then choose 1.0)."""
                              },
                "qmboltz" : {"scope" : "Force and energy matching",
                             "required" : False,
                             "long" : """When Boltzmann sampling is used to gather snapshots for force/energy matching, there is a potential
                             ambiguity regarding which ensemble one should sample from (either the force field's ensemble or the QM calculation's
                             ensemble.  The QM ensemble may be sampled using MM-sampled snapshots by reweighting; this tuning parameter specifies
                             the fraction of QM Boltzmann weight to include.  Note that when two ensembles are different, reweighting will decrease
                             the statistical significance of the number of snapshots (i.e. there is less InfoContent).""",
                             "recommend" : """If you want to reweight your snapshots entirely to the QM ensemble, choose 1.0; for hybrid weights,
                             use 0.5.  Avoid if the there is a very large RMS energy difference between QM and MM."""
                             },
                "qmboltztemp" : {"scope" : "Force and energy matching",
                                 "required" : False,
                                 "long" : """The reweighting of an ensemble involves an exponential of (DE)/kT, so there is a massive degradation of sample
                                 quality if (DE) is large.  This option allows you to change the temperature in the denominator, which is unphysical (but
                                 it does decrease the effect of moving toward the QM ensemble.""",
                                 "recommend" : """Irrelevant if 'qmboltz' is set to zero.  Leave at the default value unless you're performing experiments."""
                                 }
                
                }

def create_index(optiondoc,opt_types):
    Glossary = {}
    for i in ['strings','allcaps','lists','ints','bools','floats','sections']:
        vartype = re.sub('s$','',i)
        for j in opt_types[i]:
            val =  opt_types[i][j][0]
            #print j
            Glossary[j] = ["@li <b> %s </b> (%s)" % (j.upper(), vartype.capitalize())]
            Glossary[j].append("\\n<b> One-line description </b>: %s" % opt_types[i][j][2])
            Glossary[j].append("\\n<b> Default Value </b>: %s" % val)
            if j not in optiondoc:
                Glossary[j].append("\\n(Needs full documentation)")
                continue
            if "scope" in optiondoc[j] and "required" in optiondoc[j]:
                Glossary[j].append("\\n<b> Scope </b>: %s (%s)" % (optiondoc[j]["scope"],"<b><em>Required</em></b>" if optiondoc[j]["required"] else "Optional"))
            if "long" in optiondoc[j]:
                Glossary[j].append("\\n<b> Full description </b>: %s" % optiondoc[j]["long"])
            if "recommend" in optiondoc[j]:
                Glossary[j].append("\\n<b> Recommendation </b>: %s" % optiondoc[j]["recommend"])
    return Glossary


def main():


    Answer = ["""\section gen_option_index Option index: General options

This section contains a listing of the general options available when running
a ForceBalance job, which go into the $options section.  The general options
are global for the ForceBalance job, in contrast to 'Target options' which apply to one
target within a job (described in the next section).
The option index is generated by running make-option-index.py.
"""]
    GenIndex = create_index(GenOptionDoc,parser.gen_opts_types)

    for i in sorted([j for j in GenIndex]):
        Answer += GenIndex[i]
        Answer.append("")

    Answer.append("""\section tgt_option_index Option index: Target options

This section contains a listing of the target options available when running
a ForceBalance job, which go into the $tgt_opts section.  There can be multiple 
$tgt_opts sections in a ForceBalance input file, one for each target.
""")

    TgtIndex = create_index(TgtOptionDoc,parser.tgt_opts_types)

    for i in sorted([j for j in TgtIndex]):
        Answer += TgtIndex[i]
        Answer.append("")
    

#[i.replace('\n','') for i in GenIndex[i]]
        #for line in GenIndex[i]:
            
                
                # # and GenOptionDoc[j]["required"] == True:

                # "Scope: %s" % GenOptionDoc[j]["scope"],
                #            "Full Description: %s" % GenOptionDoc[j]["long"],
                #            "Recommendation: %s" % GenOptionDoc[j]["recommend"]]
            #Answer.append("%s (%s):%s" % (str(j), vartype, parser.gen_opts_types[i][j][1]))
            #Answer.append("%s %s" % (str(j),str(val)))
    #Answer.append("$end")
    print '\n'.join(Answer)



if __name__ == "__main__":
    main()
