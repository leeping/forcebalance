#!/usr/bin/env python

from forcebalance import parser, optimizer
from simtab import SimTab

""" The OptionDoc dictionary is a key : dictionary dictionary.
Each variable can have several parts of the documentation: 'pert' = 
"""

GenOptionDoc = {"gmxpath" : {"scope" : "Fitting simulations that use GROMACS (GROMACS-X2 for ForceEnergyMatch_GMX)",
                             "required" : True,
                             "long" : "Specify the path where GROMACS executables are installed, most likely ending in 'bin'.
                             Note that executables are only installed 'bin' if the program is installed using 'make install';
                             this will NOT be the case if you simply ran 'make'.",
                             "recommend" : "Depends on your local installation and environment."
                             },
                "tinkerpath" : {"scope" : "Fitting simulations that use TINKER",
                                "required" : True,
                                "recommend" : "Depends on your local installation and environment."
                                },
                "gmxsuffix" : {"scope" : "Fitting simulations that use GROMACS",
                               "required" : False,
                               "long" : "Depending on how GROMACS is configured and installed, a suffix may be appended to executable
                               names.  If there is a suffix, it needs to be specified here (or else ForceBalance will not find the
                               GROMACS executable and it will crash.",
                               "recommend" : "Depends on your local installation and environment."
                               },
                "penalty_type" : {"scope" : "All force field optimizations",
                                  "required" : False,
                                  "long" : "To prevent the optimization from changing the parameters too much, an additional penalty
                                  is applied to the objective function that depends linearly (L1) or quadratically (L2) on the norm
                                  of the parameter displacement vector.  L1 corresponds to LASSO regularization while L2 is known as
                                  Tikhonov regularization or ridge regression."
                                  "recommend" : "L2; tested and known to be working.  Implementation of L1 in progress."
                                  },
                "scan_vals" : {"scope" : "scan_mvals and scan_pvals job types",
                               "required" : True,
                               "long" : "This specifies a range of parameter values to scan in a uniform grid.  scan_mvals works in
                               the mathematical parameter space while scan_pvals works in the physical parameter space.  The syntax
                               is lower:step:upper .  Both lower and upper limits are included in the range."
                               "recommend" : "For scan_mvals, a range of values between -1 and +1 is recommended; for scan_pvals, choose
                               values close to the physical parameter value."
                               },
                "scanindex_num" : {"scope" : "scan_mvals and scan_pvals job types",
                                   "required" : True,
                                   "long" : "ForceBalance assigns to each adjustable parameter a 'parameter number' corresponding to
                                   its position in the parameter vector.  This tells the parameter scanner which number to scan over."
                                   "recommend" : "Look at the printout from a single-point job to decide which parameter number
                                   you wish to scan over."
                                   },
                "scanindex_name" : {"scope" : "scan_mvals and scan_pvals job types",
                                    "required" : True,
                                    "long" : "ForceBalance assigns to each adjustable parameter a 'parameter name'.  By specifying
                                    this option, this tells the parameter scanner to locate the correct parameter with the specified name and then
                                    scan over it."
                                    "recommend" : "Look at the printout from a single-point job to determine the parameter names."
                                    },
                "maxstep" : {"scope" : "All iterative optimization jobs",
                             "required" : False,
                             "recommend" : "At least 100 optimization steps are recommended."
                             },
                "readchk" : {"scope" : "Main optimizer",
                             "required" : False,
                             "long" : "The main optimizer has the ability to pick up where it left off by reading / writing checkpoint
                             files.  Here you may specify the checkpoint file to read in from a previous optimization run.  This is
                             equivalent to reading in stored parameter values, except the gradient and Hessian (which contains memory from previous
                             steps) is recorded too."
                             },
                "writechk" : {"scope" : "Main optimizer",
                              "required" : False,
                              "long" : "The main optimizer has the ability to pick up where it left off by reading / writing checkpoint
                              files.  Here you may specify the checkpoint file to write after the job is finished.",
                              "recommend" : "Writing the checkpoint file is highly recommended."
                              },
                "writechk_step" : {"scope" : "Main optimizer when 'writechk' is turned on",
                                   "required" : False,
                                   "long" : "Write a checkpoint file every single step, not just after the job is finished.",
                                   "recommend" : "Useful if you want to quit an optimization before it finishes and restart,
                                   but make sure you don't overwrite existing checkpoint files by accident."
                                   },
                "ffdir" : {"scope" : "All force field optimizations",
                           "required" : False,
                           "recommend" : "Unless you're using a nonstandard location for force field files, you probably shouldn't change this."
                           },
                "jobtype" : {"scope" : "All force field optimizations",
                             "required" : True,
                             "long" : "Here you may specify the type of ForceBalance job.  This ranges from gradient-based and stochastic
                             optimizations to simple scans over the parameter space and finite difference checking of gradients.",
                             "recommend" : "See the Optimizer class documentation for which optimizer is best suited for you.  Available
                             options are: %s" % ', '.join(sorted([keys(optimizer.OptTab)]))
                             },
                "forcefield" : {"scope" : "All force field optimizations",
                                "required" : True
                                },
                "trust0" : {"scope" : "Main optimizer",
                            "required" : False,
                            "long" : "The main optimizer uses a trust radius which 'adapts' (i.e. increases or decreases) based on whether the last step
                            was a good or bad step.  'trust0' provides the starting trust radius, and the trust radius is not allowed to increase too much
                            from trust0.",
                            "recommend" : "Increase from the default if the optimizer takes many good steps but takes too long; decrease if the
                            optimizer takes many bad steps."
                            },
                "convergence_objective" : {"scope" : "Main optimizer",
                                           "required" : False,
                                           "long" : "The main optimizer will quit when the last ten good values of the objective function have a
                                           standard deviation that falls below this number.  We use the last ten good values (instead of the latest
                                           change in the objective function), otherwise this condition would be triggered by taking tiny steps.",
                                           "recommend" : "Decrease this value if it's being triggered by small step sizes."
                                           },
                "convergence_step" : {"scope" : "Main optimizer",
                                      "required" : False,
                                      "long" : "The main optimizer will quit when the step size falls below this number.  This happens if we are
                                      approaching a local minimum, or if the optimizer is constantly taking bad steps and the trust radius is
                                      reduced until it falls below this number.  In the latter case, this usually means that the derivatives are
                                      wrong.",
                                      "recommend" : "Make sure that this value is much smaller than trust0."
                                      },
                "convergence_gradient" : {"scope" : "Main optimizer",
                                          "required" : False,
                                          "long" : "The main optimizer will quit when the objective function gradient falls below this number.
                                          Since this is a newly implemented option, I can't say when this option will fail.",
                                          "recommend" : "Leave at the default, or set to several orders of magnitude below a typical value of the gradient
                                          (perhaps the gradient at the start of the optimization.)"
                                          },
                "eig_lowerbound" : {"scope" : "Main optimizer",
                                    "required" : False,
                                    "long" : "The main optimizer will misbehave if there are negative or very small eigenvalues in the
                                    objective function Hessian.  In the former case the optimizer will travel toward a saddle point (or
                                    local maximum), and in the latter case the matrix inversion will fail because of the matrix singularity.
                                    If the smallest eigenvalue is below this value, then a multiple of the identity matrix is added to the
                                    Hessian to increase the smallest eigenvalue to at least this value.",
                                    "recommend" : "Shouldn't have to worry about this setting, unless the optimizer appears to be taking
                                    bad steps or inverting nearly singular matrices."
                                    },
                
                "finite_difference_h" : {"scope" : "fdcheck_G or fdcheck_H job types, or whenever the objective function is evaluated using finite difference",
                                         "required" : False,
                                         "long" : "When the objective function derivatives are checked using finite difference, or when the objective function derivative
                                         requires finite difference, this is the step size that is used (in the mathematical space).  The actual parameter in the
                                         force field is changed by this amount times the rescaling factor.",
                                         "recommend" : "1e-2 to 1e-4; run FDCheckG to see if derivatives are accurate; if derivatives are inaccurate then adjust accordingly.
                                         If the objective function itself requires finite difference, there will still be a difference because FDCheckG(H) uses an accurate
                                         seven-point (five-point) stencil.  Make sure that the derivatives agree before settling on a value to use."
                                         },
                
                "penalty_additive" : {"scope" : "Objective function",
                                      "required" : False,
                                      "long" : "Add a penalty to the objective function (e.g. L2 or L1 norm) with this prefactor.
                                      Using an additive penalty requires an assessment of the order of magnitude of the objective function,
                                      but it is closer to the statistical concept of ridge or LASSO regression.",
                                      "recommend" : "No recommendation; run a single-point calculation to choose a prefactor.  Consider 0.01
                                      for an objective function of order 1."
                                      },
                
                "penalty_multiplicative" : {"scope" : "Objective function",
                                            "required" : False,
                                            "long" : "Multiply the objective function by (1+X) where X is this value.
                                            Using an multiplicative penalty works well for objective functions of any size but it is not
                                            equivalent to statistical regularization methods.",
                                            "recommend" : "A value of 0.01 tends to keep the length of the parameter vector from exceeding 1."
                                            },
                
                "read_mvals" : {"scope" : "All force field optimizations",
                                "required" : False,
                                "long" : "Read in mathematical parameters before starting the optimization.  There is a standardized
                                syntax, given by:
                                
                                read_mvals
                                0 [ -2.9766e-01 ] : VDWSOW
                                1 [  2.2283e-01 ] : VDWTOW
                                2 [ -1.1138e-03 ] : BONDSBHWOW
                                3 [ -9.0883e-02 ] : BONDSKHWOW
                                \read_mvals
                                ",
                                "recommend" : "If you run the main optimizer, it will print out this block at the very end for you to use and/or modify."
                                },
                
                "read_pvals" : {"scope" : "All force field optimizations",
                                "required" : False,
                                "long" : "Read in physical parameters before starting the optimization.  There is a standardized
                                syntax, given by:
                                
                                read_pvals
                                0 [  2.9961e-01 ] : VDWSOW
                                1 [  1.2009e+00 ] : VDWTOW
                                2 [  9.5661e-02 ] : BONDSBHWOW
                                3 [  4.1721e+05 ] : BONDSKHWOW
                                \read_pvals
                                
                                These are the actual numbers that go into the force field file, so note the large changes in magnitude.
                                ",
                                "recommend" : "If you run the main optimizer, it will print out this block at the very end for you to use and/or modify."
                                },
                }

SimOptionDoc = {"name" : {"scope" : "All fitting simulations",
                          "required" : True,
                          "recommend" : "Choose a descriptive name and make sure all fitting simulations have different names."
                          },
                "simtype" : {"scope" : "All fitting simulations",
                             "required" : True,
                             "long" : "This is the type of fitting simulation that you are running.  The current accepted values for the fitting simulation are given in the SimTab.py file: %s." % ', '.join([i for i in SimTab]),
                             "recommend" : "Choose the appropriate type, and if the fitting simulation is missing, feel free to implement your own (or ask me for help)."
                             },
                "fd_ptypes" : {"scope" : "All fitting simulations",
                             "required" : False,
                             "long" : "To compute the objective function derivatives, some components may require numerical finite difference in the derivatives.  Here you may specify the parameter types that finite difference is applied to,
                             or write 'ALL' to take finite-difference derivatives in all parameter types.",
                             "recommend" : "If you aren't sure, either use 'ALL' to do finite difference in each component (this is costly), or run a fdcheckG(H) job with this option set to 'NONE' to check which analytic derivatives are missing.
                             Usually analytic derivatives will be missing in anything but FORCEENERGYMATCH_GMXX2 jobs."
                             },
                "shots" : {"scope" : "Force and energy matching simulations",
                           "required" : False,
                           "long" : "This option allows you to choose a subset from the snapshots available in the force matching 'simulations' directory.
                           The subset is simply taken from the front of the trajectory.
                           In the future this option will be expanded to allow a random selection of snapshots, or a specific selection",
                           "recommend" : "100-10,000 snapshots are recommended.  Note that you need at least 3x (number of atoms) if
                           the covariance matrix is turned on."
                           },
                "fitatoms" : {"scope" : "Force and energy matching simulations",
                              "required" : False,
                              "long" : "Choose a subset of atoms from the force matching simulation to fit forces to.  This is useful in situations where
                              it is undesirable to fit the forces on part of the system (e.g. the part that is described by another force field.)
                              Currently, you are only allowed to choose from the atoms in the front of the trajectory;
                              soon this will be expanded for random flexibility (see 'shots').  However, random coordinate selections are not allowed. ;)",
                              "recommend" : "Situation-dependent; this should be based on the part of the simulation that you're fitting, or leave blank
                              if you're fitting the whole system."
                              },
                "whamboltz" : {"scope" : "Force and energy matching simulations",
                               "required" : False,
                               "long" : "In self-consistent energy/force matching projects, the data from previous cycles can be reused by applying the
                               Weighted Histogram Analysis Method (WHAM).  However, the WHAM data is currently generated by external scripts that
                               haven't made it into this distribution yet.  In the future, generation of WHAM data will be incorporated into this
                               program automatically.",
                               "recommend" : "Leave off unless you have an externally generated wham-master.txt and wham-weights.txt files."
                               },
                "sampcorr" : {"scope" : "Force and energy matching simulations that use GROMACS-X2",
                              "required" : False,
                              "long" : "Every time the force field parameters are updated, the ensemble is different.  In principle this applies
                              to not only the self-consistent optimization cycles (which include re-running dynamics, QM calculations etc) but also
                              the numerical optimization itself.  When this option is turned on, the Boltzmann weights of the snapshots are updated
                              in every step of the optimization and the derivatives are modified accordingly.  My investigations reveal that this makes
                              the force field more accurate in energy minima and less accurate for barriers, which was not very useful.  I haven't touched
                              the 'sampling corrected' code in a long time; thus this option is vestigial and may be removed in the future.",
                              "recommend" : "Off."
                              },
                "covariance" : {"scope" : "Force and energy matching simulations that use GROMACS",
                                "required" : False,
                                "long" : "The components of the energy and force are rescaled to be on the same footing when the objective function is
                                optimized.  This can be done by ",
                                "recommend" : "No recommendation; ."
                                },
                
    }

def main():
    #for t in gen_opts_types:
        
    for i in ['strings','allcaps','lists','ints','bools','floats','sections']:
        vartype = re.sub('s$','',i)
        for j in typedict[i]:
            val = optdict[j] if optdict != None else typedict[i][j][0]
            if firstentry:
                firstentry = 0
            else:
                Answer.append("")
            Answer.append("# (%s) %s" % (vartype, typedict[i][j][1]))
            Answer.append("%s %s" % (str(j),str(val)))
    Answer.append("$end")
    return Answer


if __name__ == "__main__":
    main()
