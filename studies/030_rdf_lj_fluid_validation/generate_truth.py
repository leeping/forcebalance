""" Generate "ground truth" RDFs for the LJ-fluid validation.

Runs a short NPT simulation of a monatomic Lennard-Jones fluid directly with
OpenMM at a given sigma, and computes g(r) with MDTraj.  The resulting r/g(r)
table is what build_targets.py turns into a ForceBalance rdf.dat target -- it
is real simulation output, not synthetic/fabricated data.

Usage:
    python generate_truth.py <label> <sigma_angstrom>

Requires an OpenMM CUDA-capable build and mdtraj.  Takes a few seconds on a
modern GPU (~60000 steps).
"""
import os
import sys
import numpy as np
import openmm as mm
from openmm import app, unit
import mdtraj as md

N_PER_SIDE = 8              # 8^3 = 512 particles
MASS = 39.948                # amu, argon-like
EPSILON = 0.996 * unit.kilojoule_per_mole   # argon-like well depth
TEMP = 100 * unit.kelvin
PRESSURE = 1 * unit.atmosphere
LATTICE_SPACING = 0.4        # nm, initial cubic lattice spacing (relaxed by the barostat)
EQUIL_STEPS = 20000
PROD_STEPS = 40000
TIMESTEP = 2 * unit.femtosecond
SAVE_INTERVAL = 100          # steps between RDF snapshots

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')


def _lattice_positions():
    n = N_PER_SIDE
    return np.array([[i * LATTICE_SPACING, j * LATTICE_SPACING, k * LATTICE_SPACING]
                      for i in range(n) for j in range(n) for k in range(n)])


def _dummy_topology(n):
    top = app.Topology()
    chain = top.addChain()
    for _ in range(n):
        res = top.addResidue('LJ', chain)
        top.addAtom('Ar', app.element.argon, res)
    return top


def build_system(sigma_nm):
    positions = _lattice_positions()
    box = N_PER_SIDE * LATTICE_SPACING

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*(np.eye(3) * box))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    for _ in positions:
        system.addParticle(MASS)
        nb.addParticle(0.0, sigma_nm * unit.nanometer, EPSILON)
    system.addForce(nb)
    system.addForce(mm.MonteCarloBarostat(PRESSURE, TEMP, 25))

    integrator = mm.LangevinMiddleIntegrator(TEMP, 1 / unit.picosecond, TIMESTEP)
    platform = mm.Platform.getPlatformByName('CUDA')
    sim = app.Simulation(_dummy_topology(len(positions)), system, integrator, platform)
    sim.context.setPositions(positions * unit.nanometer)
    return sim


def run(sigma_ang, label):
    sim = build_system(sigma_ang / 10.0)
    print(f"[{label}] Minimizing...")
    sim.minimizeEnergy()
    print(f"[{label}] Equilibrating {EQUIL_STEPS} steps...")
    sim.step(EQUIL_STEPS)

    frames, box_lengths = [], []
    print(f"[{label}] Production {PROD_STEPS} steps, saving every {SAVE_INTERVAL}...")
    for _ in range(PROD_STEPS // SAVE_INTERVAL):
        sim.step(SAVE_INTERVAL)
        state = sim.context.getState(getPositions=True)
        frames.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        box_lengths.append([box[0][0], box[1][1], box[2][2]])

    xyz = np.array(frames, dtype=np.float32)
    unitcell_lengths = np.array(box_lengths, dtype=np.float32)
    unitcell_angles = np.tile([90., 90., 90.], (len(frames), 1)).astype(np.float32)

    top = md.Topology()
    chain = top.add_chain()
    for _ in range(xyz.shape[1]):
        res = top.add_residue('LJ', chain)
        top.add_atom('Ar', md.element.argon, res)
    traj = md.Trajectory(xyz=xyz, topology=top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

    pairs = top.select_pairs('name Ar', 'name Ar')
    r, gr = md.compute_rdf(traj, pairs=pairs, r_range=(0.15, 0.9), bin_width=0.02, periodic=True, opt=True)

    avg_box = unitcell_lengths.mean(axis=0)
    total_mass = len(frames[0]) * MASS * unit.amu / unit.AVOGADRO_CONSTANT_NA
    volume = avg_box[0] * avg_box[1] * avg_box[2] * unit.nanometer**3
    density = (total_mass / volume).in_units_of(unit.gram / unit.centimeter**3)
    peak_r = r[np.argmax(gr)]
    print(f"[{label}] sigma={sigma_ang} Ang -> density={density}, RDF peak at r={peak_r*10:.3f} Ang (g={gr.max():.2f})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"rdf_truth_{label}.dat")
    np.savetxt(out_path, np.column_stack([r * 10, gr]), header="r(Ang) g(r)")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    run(float(sys.argv[2]), sys.argv[1])
