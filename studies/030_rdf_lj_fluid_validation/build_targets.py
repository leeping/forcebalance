""" Build the two ForceBalance case directories (case_peak30/, case_peak38/) for the
LJ-fluid RDF-target validation, from the ground-truth RDFs in results/.

Each case directory is a self-contained ForceBalance project: a single-atom-type
Lennard-Jones force field (only sigma is marked parameterize="sigma"; epsilon is
held fixed), a 512-particle liquid box, a single-atom gas phase, and an rdf.dat
target built from the corresponding results/rdf_truth_<label>.dat.  Both cases
start from the same deliberately-wrong initial guess (SIGMA0_ANG) so that the
optimizer has to move in opposite directions to recover each ground truth.

Usage:
    python build_targets.py
"""
import os
import numpy as np
from openmm import app, unit
import generate_truth as truth

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')

SIGMA0_ANG = 3.4   # deliberately-wrong initial guess for both cases
CASES = [('peak30', 3.0), ('peak38', 3.8)]

OPTIMIZE_IN_TEMPLATE = """$options
ffdir forcefield
penalty_type L2
jobtype OPTIMIZE
forcefield lj.xml
maxstep 12
convergence_step 0.001
convergence_objective 0.01
convergence_gradient 0.001
trust0 0.1
mintrust 0.01
finite_difference_h 0.001
normalize_weights False
backup False
$end

$target
name LJ_RDF
type Liquid_OpenMM
weight 1.0

w_rho 0.0
w_hvap 0.0
w_alpha 0.0
w_kappa 0.0
w_cp 0.0
w_eps0 0.0
w_surf_ten 0.0
w_rdf 1.0

liquid_equ_steps {equ_steps}
liquid_prod_steps {prod_steps}
liquid_timestep {timestep}
liquid_interval {interval}
liquid_coords liquid.pdb

gas_equ_steps 5000
gas_prod_steps 5000
gas_timestep {timestep}
gas_interval {interval}
gas_coords gas.pdb
$end
"""


def write_ff_xml(path, sigma0_nm, epsilon_kjmol):
    with open(path, 'w') as f:
        f.write(f"""<ForceField>
 <AtomTypes>
  <Type name="LJ-Ar" class="Ar" element="Ar" mass="39.948"/>
 </AtomTypes>
 <Residues>
  <Residue name="LJ">
   <Atom name="Ar" type="LJ-Ar"/>
  </Residue>
 </Residues>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="LJ-Ar" charge="0.0" sigma="{sigma0_nm:.6e}" epsilon="{epsilon_kjmol:.6e}" parameterize="sigma"/>
 </NonbondedForce>
</ForceField>
""")


def write_liquid_pdb(path):
    positions = truth._lattice_positions() * unit.nanometer
    box = truth.N_PER_SIDE * truth.LATTICE_SPACING
    top = app.Topology()
    top.setPeriodicBoxVectors(np.eye(3) * box)
    chain = top.addChain()
    for _ in positions:
        res = top.addResidue('LJ', chain)
        top.addAtom('Ar', app.element.argon, res)
    with open(path, 'w') as f:
        app.PDBFile.writeFile(top, positions, f)


def write_gas_pdb(path):
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue('LJ', chain)
    top.addAtom('Ar', app.element.argon, res)
    with open(path, 'w') as f:
        app.PDBFile.writeFile(top, [[0.0, 0.0, 0.0]] * unit.nanometer, f)


def write_rdf_dat(path, truth_file, rdf_name='name Ar&name Ar', pt_index=0):
    data = np.loadtxt(truth_file)
    with open(path, 'w') as f:
        f.write("# Synthetic target RDF for the LJ-fluid ForceBalance RDF-target validation.\n")
        f.write(f"RDF {rdf_name}\n")
        f.write(f"@ PT {pt_index}\n")
        for r, gval in data:
            f.write(f"{r:.4f} {gval:.6f}\n")
        f.write("ENDPT\n")
        f.write("ENDRDF\n")


def write_data_csv(path, temp_k, pressure_atm):
    with open(path, 'w') as f:
        f.write("Global,rdf_denom,0.05,\n")
        f.write("T,P,rdf,rdf_wt\n")
        f.write(f"{temp_k},{pressure_atm} atm,0,1\n")


def build(case_dir, truth_file, sigma0_ang=SIGMA0_ANG, epsilon_kjmol=None):
    epsilon_kjmol = epsilon_kjmol or truth.EPSILON.value_in_unit(unit.kilojoule_per_mole)
    ff_dir = os.path.join(case_dir, 'forcefield')
    tgt_dir = os.path.join(case_dir, 'targets', 'LJ_RDF')
    os.makedirs(ff_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    write_ff_xml(os.path.join(ff_dir, 'lj.xml'), sigma0_ang / 10.0, epsilon_kjmol)
    write_liquid_pdb(os.path.join(tgt_dir, 'liquid.pdb'))
    write_gas_pdb(os.path.join(tgt_dir, 'gas.pdb'))
    write_rdf_dat(os.path.join(tgt_dir, 'rdf.dat'), truth_file)
    write_data_csv(os.path.join(tgt_dir, 'data.csv'),
                    temp_k=truth.TEMP.value_in_unit(unit.kelvin),
                    pressure_atm=truth.PRESSURE.value_in_unit(unit.atmosphere))
    with open(os.path.join(case_dir, 'optimize.in'), 'w') as f:
        f.write(OPTIMIZE_IN_TEMPLATE.format(
            equ_steps=truth.EQUIL_STEPS, prod_steps=truth.PROD_STEPS,
            timestep=truth.TIMESTEP.value_in_unit(unit.femtosecond),
            interval=truth.SAVE_INTERVAL * truth.TIMESTEP.value_in_unit(unit.picosecond)))
    print(f"Built {case_dir}: initial sigma0={sigma0_ang} Ang (fixed epsilon={epsilon_kjmol} kJ/mol), "
          f"target={os.path.basename(truth_file)}")


if __name__ == "__main__":
    for label, sigma_true in CASES:
        truth_file = os.path.join(RESULTS_DIR, f'rdf_truth_{label}.dat')
        if not os.path.exists(truth_file):
            raise SystemExit(f"Missing {truth_file} -- run `python generate_truth.py {label} {sigma_true}` first.")
        build(os.path.join(HERE, f'case_{label}'), truth_file)
