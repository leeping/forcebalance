
"""Build a refit-ready force field for a 4-site water model workflow.

This script starts from a small-molecule OpenFF force field and a separate
water-model force field, then:

1. Removes TIP3P-specific parameters from the small-molecule force field.
2. Marks vdW parameters for optimization based on dataset coverage.
3. Zeroes atom-centered water charges and reassigns charge to virtual sites.
4. Adds an auxiliary virtual-site type used for geometry optimization.
5. Merges selected handlers from the water model into the output force field.

Example:
    python generate-4-site-input-forcefield.py \
        --input-force-field openff-2.3.0.offxml \
        --water-model tip4p_fb \
        --input-dataset dataset.json \
        --output-force-field outputs/force-field.offxml \
        --n-data-points 3
"""

import click
import pathlib
from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import numpy as np
from openff.toolkit import ForceField, Molecule, Topology, unit
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet
from openff.interchange.models import (
    TopologyKey,
    VirtualSiteKey,
)

from loguru import logger


@click.command()
@click.option(
    "--input-force-field",
    "-i",
    default="openff-2.3.0.offxml",
    show_default=True,
    type=str,
    help="Input small-molecule OpenFF force field (.offxml).",
)
@click.option(
    "--water-model",
    "-w",
    default="tip4p_fb",
    show_default=True,
    type=str,
    help="Water model basename (expects <water-model>.offxml).",
)
@click.option(
    "--output-force-field",
    "-o",
    default="forcefield/force-field.offxml",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help="Path for the generated force field.",
)
def main(
    input_force_field: pathlib.Path,
    water_model: str,
    output_force_field: pathlib.Path,
):
    small_molecule_ff = ForceField(str(input_force_field))
    water_ff = ForceField(f"{water_model}.offxml")

    # === Constraints ===
    # Keep constraints fixed, but remove TIP3P-specific entries so we can
    # replace water terms with those from the selected 4-site model.
    constraints = small_molecule_ff.get_parameter_handler("Constraints")
    assert constraints.parameters[0].id == "c1", "Must use constrained FF"
    tip3p_parameter_indices = [
        i for i, p in enumerate(constraints._parameters[:3]) if "tip3p" in p.id
    ]
    for index in sorted(tip3p_parameter_indices, reverse=True):
        del constraints._parameters[index]

    # === vdW ===
    # Remove TIP3P atom types and mark only well-sampled vdW parameters for refit.
    vdWs = small_molecule_ff.get_parameter_handler("vdW")

    all_parameter_ids = [p.id for p in vdWs.parameters]
    tip3p_parameter_indices = [
        all_parameter_ids.index("n-tip3p-H"), all_parameter_ids.index("n-tip3p-O")
    ]
    for index in sorted(tip3p_parameter_indices, reverse=True):
        del vdWs._parameters[index]

    # parameterize another vdW type
    mol = Molecule.from_smiles("CC(C)O")
    labels = small_molecule_ff.label_molecules(mol.to_topology())[0]["vdW"]
    vdw_parameter = labels[(0,)]
    # small perturbation to make sure it's being updated during optimization
    print(f"Original CX4 vdW epsilon: {vdw_parameter.epsilon}, sigma: {vdw_parameter.sigma}")
    vdw_parameter.epsilon += 0.1 * unit.kilocalorie_per_mole
    vdw_parameter.sigma += 0.1 * unit.angstrom
    vdw_parameter.add_cosmetic_attribute("parameterize", "epsilon,sigma")

    # parameterize water O vdW
    water_o_vdw_parameter = None
    for parameter in water_ff.get_parameter_handler("vdW").parameters:
        if parameter.id and "tip4p" in parameter.id and parameter.id.endswith("-O"):
            water_o_vdw_parameter = parameter
            break
    if water_o_vdw_parameter is None:
        raise ValueError("Could not find water oxygen vdW parameter in water model")
    print(f"Original water O vdW epsilon: {water_o_vdw_parameter.epsilon}, sigma: {water_o_vdw_parameter.sigma}")
    water_o_vdw_parameter.add_cosmetic_attribute("parameterize", "epsilon,sigma")

    # === Electrostatics ===
    # Remove TIP3P library charges from the small-molecule force field.
    # Water charges are then taken from the selected water model and modified.

    library_charges = small_molecule_ff.get_parameter_handler("LibraryCharges")
    library_charges_parameters_ids = [p.id for p in library_charges.parameters]
    tip3p_parameter_indices = [
        library_charges_parameters_ids.index("q-tip3p-O"),
        library_charges_parameters_ids.index("q-tip3p-H")
    ]
    for index in sorted(tip3p_parameter_indices, reverse=True):
        del library_charges._parameters[index]

    h_librarycharge = None
    o_librarycharge = None
    library_charges_water = water_ff.get_parameter_handler("LibraryCharges")
    for parameter in library_charges_water.parameters:
        if "ion" in parameter.id:
            continue
        if parameter.id.endswith("-H"):
            h_librarycharge = parameter
        elif parameter.id.endswith("-O"):
            o_librarycharge = parameter

    if h_librarycharge is None or o_librarycharge is None:
        raise ValueError("Could not find water library charges in water model")

    # Neutralize atom-centered charges; charge is redistributed onto virtual sites.
    h_librarycharge.charge = [0 * unit.elementary_charge]
    o_librarycharge.charge = [0 * unit.elementary_charge]

    # === Virtual sites ===
    # Define two virtual-site categories:
    # 1) M site: carries the main negative charge, no LJ terms.
    # 2) L sites: auxiliary sites near hydrogens, used for geometry + LJ tuning.

    vsites = water_ff.get_parameter_handler("VirtualSites")
    assert len(vsites.parameters) == 1, "Expected exactly one virtual site parameter for 4-site water model"

    m_parameter = vsites.parameters[0]
    assert m_parameter.type == "DivalentLonePair"
    # Keep oxygen increment fixed; tie H increments together during optimization.
    m_parameter.add_cosmetic_attribute("parameterize", "distance,charge_increment2")
    m_chargeincrement2 = f"PRM['VirtualSites/VirtualSite/charge_increment2/{m_parameter.smirks}/{m_parameter.type}/{m_parameter.name}/{m_parameter.match}']"
    m_parameter.add_cosmetic_attribute(
        "parameter_eval",
        f"charge_increment3={m_chargeincrement2}"
    )
    m_parameter_charge = -sum(m_parameter.charge_increment)

    # arbitrarily alter charge increment to make sure it's being updated during optimization
    print(f"Original M-site charge increment: {m_parameter.charge_increment2}")
    m_parameter.charge_increment2 += 0.05 * unit.elementary_charge
    m_parameter.charge_increment3 += 0.05 * unit.elementary_charge

    # Build the L-site geometry from constrained O-H and H-H distances.
    angle_constraint = None
    bond_constraint = None
    water_constraints = water_ff.get_parameter_handler("Constraints")
    for parameter in water_constraints.parameters:
        if "H-O-H" in parameter.id:
            angle_constraint = parameter
        elif "O-H" in parameter.id or "H-O" in parameter.id:
            bond_constraint = parameter
    
    if angle_constraint is None or bond_constraint is None:
        raise ValueError("Could not find necessary constraints in water model for virtual site geometry")

    l_distance = bond_constraint.distance
    hh_distance = angle_constraint.distance.m_as(unit.angstrom)

    # Compute out-of-plane angle from a right triangle defined by H-H and O-H.
    out_of_plane_angle = np.rad2deg(
        np.arcsin(hh_distance / (2 * l_distance.m_as(unit.angstrom)))
    )

    l_parameter_kwargs = {
        "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
        "type": "DivalentLonePair",
        "name": "EP-L",
        "match": "all_permutations",
        "distance": l_distance,
        "outOfPlaneAngle": out_of_plane_angle * unit.degree,
        "charge_increment": [
            0 * unit.elementary_charge,
            -m_parameter.charge_increment2 / 2,
            -m_parameter.charge_increment3 / 2,
        ],
        "epsilon": 0.0 * unit.kilocalorie_per_mole,
        "sigma": 1.0 * unit.angstrom,
    }
    vsites.add_parameter(l_parameter_kwargs)
    # don't optimize this just to avoid confusing optimizer too much
    # l_parameter.add_cosmetic_attribute(
    #     "parameterize", "distance,outOfPlaneAngle"
    # )
    # l_parameter.add_cosmetic_attribute(
    #     "parameter_eval",
    #     f"charge_increment2=-0.5*{m_chargeincrement2}, charge_increment3=-0.5*{m_chargeincrement2}"
    # )

    molecule = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    molecule.generate_conformers(n_conformers=1)
    interchange = water_ff.create_interchange(molecule.to_topology())

    # Validate geometry of virtual sites; should match angle geometry.
    interchange.minimize()
    positions = interchange.get_positions(include_virtual_sites=True)

    rdmol = Chem.RWMol(molecule.to_rdkit())
    n_vsites = len(positions) - molecule.n_atoms
    for _ in range(n_vsites):
        rdmol.AddAtom(Chem.Atom(0))  # virtual site as dummy atom

    # Create conformer with virtual site positions
    conf = Chem.Conformer(len(positions))
    for i in range(len(positions)):
        pos = positions[i].m_as("angstrom")
        conf.SetAtomPosition(i, pos)
    rdmol.AddConformer(conf)

    lol_angle = rdMolTransforms.GetAngleDeg(conf, 4, 0, 5)
    ol_distance = rdMolTransforms.GetBondLength(conf, 0, 4)

    l_distance_ = l_distance.m_as("angstrom")

    assert np.isclose(ol_distance, l_distance_), f"Expected OL distance {l_distance_}, got {ol_distance}"
    assert np.isclose(lol_angle, 2 * out_of_plane_angle), f"Expected LOL angle {2 * out_of_plane_angle}, got {lol_angle}"
    
    # now combine force fields
    # Merge updated water handlers into the base small-molecule force field.
    for handler_name in ["Constraints", "vdW", "LibraryCharges", "VirtualSites"]:
        small_molecule_handler = small_molecule_ff.get_parameter_handler(handler_name)
        water_handler = water_ff.get_parameter_handler(handler_name)

        for parameter in water_handler.parameters:
            if parameter.id and "ion" in parameter.id:
                continue
            kwargs = parameter.to_dict()
            if "parameterize" in parameter._cosmetic_attribs or "parameter_eval" in parameter._cosmetic_attribs:
                kwargs["allow_cosmetic_attributes"] = True
            small_molecule_handler.add_parameter(kwargs)

    output_force_field.parent.mkdir(parents=True, exist_ok=True)

    small_molecule_ff.to_file(output_force_field)
    logger.info(f"Wrote modified force field to {output_force_field}")


if __name__ == "__main__":
    main()
