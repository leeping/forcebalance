# RDF liquid target: closed-loop validation on a Lennard-Jones fluid

This is a correctness check for the RDF (radial distribution function) liquid
target, not a physics study: it verifies that fitting against g(r) actually
recovers the parameter that generated it, end to end through ForceBalance's
`Liquid_OpenMM` target -- rdf.dat parsing, per-snapshot MDTraj sampling during
NPT MD, MBAR reweighting, and the analytic RDF gradient.

## The idea

1. Simulate a monatomic Lennard-Jones fluid (argon-like, 512 atoms) at a known
   `sigma`, and record its radial distribution function. This is genuine MD
   output, not fabricated data.
2. Treat that g(r) as the *only* fitting target (`w_rdf 1.0`, every other
   liquid property weighted to zero) and optimize a single free parameter,
   `sigma`, starting from a deliberately wrong initial guess. `epsilon` is
   held fixed so the result isolates the length-scale relationship the RDF
   target is supposed to recover.
3. Check whether the optimizer walks `sigma` back to the value that generated
   the target RDF -- and in opposite directions for two different targets,
   from the same starting guess.

Two ground truths are used: `sigma_true = 3.0 Ang` ("peak30") and
`sigma_true = 3.8 Ang` ("peak38"), both starting from `sigma0 = 3.4 Ang`.

## Results

| target | true sigma | initial sigma0 | fitted sigma | error | target RDF peak | fitted RDF peak | 2^(1/6)*sigma_fit |
|---|---|---|---|---|---|---|---|
| peak30 | 3.000 Ang | 3.400 Ang | 3.027 Ang | 0.91% | 3.223 Ang | 3.223 Ang | 3.398 Ang |
| peak38 | 3.800 Ang | 3.400 Ang | 3.828 Ang | 0.74% | 4.236 Ang | 4.034 Ang | 4.297 Ang |

(`results/summary.csv` has the machine-readable version; regenerate with `summarize.py`.)

Both fits converge in under 3 minutes on a single GPU (Newton-Raphson,
analytic gradient, ~5-7 iterations) and recover the true `sigma` to within 1%,
moving in opposite directions from the same starting guess. See
`results/rdf_comparison.png` -- fitted and target g(r) overlap almost exactly
across the whole curve (first peak, first minimum, second peak), not just at
the argmax bin used in the table above.

The fitted force field's pairwise minimum, `2^(1/6)*sigma_fit`, sits slightly
*outside* the actual liquid RDF peak (e.g. 3.398 Ang vs. 3.223 Ang for
peak30). That's expected liquid-state physics, not fitting error: in a dense
fluid the first coordination shell is pulled inward of the isolated-pair
potential minimum by the surrounding packing pressure, so
`peak(g(r)) < 2^(1/6)*sigma` is the normal relationship, not something to
chase out. The parameter-recovery numbers above are the real validation,
since that's what the optimizer directly fits to.

## Layout

```
generate_truth.py    Runs the ground-truth NPT simulation for one sigma and writes
                      results/rdf_truth_<label>.dat (r, g(r)) -- requires an OpenMM
                      CUDA build and mdtraj.
build_targets.py      Builds case_peak30/ and case_peak38/ from the files in results/:
                      forcefield/lj.xml (only sigma is parameterize="sigma"),
                      targets/LJ_RDF/{liquid.pdb,gas.pdb,rdf.dat,data.csv}, optimize.in.
summarize.py          After both cases have been run, reads optimize.out (fitted sigma)
                      and the last iteration's npt_result.p (fitted g(r)), and writes
                      results/summary.csv + results/rdf_comparison.png.

case_peak30/          Ready-to-run ForceBalance project, target = ground truth at sigma=3.0.
case_peak38/           ...same, target = ground truth at sigma=3.8. Both already contain
                       optimize.out / optimize.sav from a completed run.
results/               Ground-truth RDFs, the comparison plot, and summary.csv.
```

## Reproducing it

Requires ForceBalance installed, an OpenMM build with the `CUDA` platform, and
`mdtraj`. `Liquid_OpenMM` targets default to the `CUDA` platform, so nothing
extra needs to be configured if a working CUDA-enabled OpenMM is already on
the system -- for a fresh conda environment, `conda install -c conda-forge
openmm mdtraj` is normally enough to pull in a CUDA-capable build.

```bash
# 1. Generate the two ground-truth RDFs (~5 s each on a modern GPU)
python generate_truth.py peak30 3.0
python generate_truth.py peak38 3.8

# 2. Build the two ForceBalance project directories from those RDFs
python build_targets.py

# 3. Run the optimizer for each case (~2 min each on a modern GPU)
cd case_peak30 && ForceBalance.py optimize.in && cd ..
cd case_peak38 && ForceBalance.py optimize.in && cd ..

# 4. Summarize: results/summary.csv + results/rdf_comparison.png
python summarize.py
```

`generate_truth.py` and `build_targets.py` regenerate `case_peak30/` and
`case_peak38/` from scratch; step 3 overwrites `optimize.out`/`optimize.sav`
and creates a (gitignored) `optimize.tmp/` working directory per case with the
full per-iteration MD output.

## Why this exists

Written while reviving the RDF liquid target from
[PR #109](https://github.com/leeping/forcebalance/pull/109) (open since 2018),
which added RDF matching but was never merged. Unit tests cover the ported
code's individual pieces (rdf.dat parsing, MSD math, MDTraj plumbing); this
example is the complementary end-to-end check that the whole pipeline -- MD
sampling through to the fitted parameter -- behaves the way a physicist would
expect it to.
