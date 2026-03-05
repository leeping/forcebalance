""" @package forcebalance.recharge
A target to train bond charge corrections (currently only the SMIRNOFF force
field format is supported) against electrostatic potential data.

author Simon Boothroyd
@date 07/2020
"""
from __future__ import division, print_function

import json
import os

import numpy as np

from forcebalance.nifty import printcool_dictionary, warn_once
from forcebalance.output import getLogger
from forcebalance.target import Target

try:
    from openff.recharge.charges import ChargeSettings
    from openff.recharge.esp.storage import MoleculeESPStore
    from openff.recharge.optimize import ElectricFieldObjective, ESPObjective # ElectricFieldOptimization, ESPOptimization
    from openff.recharge.charges.bcc import BCCCollection
    recharge_import_success = True
except ImportError:
    recharge_import_success = False

try:
    from openff.toolkit.typing.engines import smirnoff
    toolkit_import_success = True
except ImportError:
    toolkit_import_success = False

logger = getLogger(__name__)


class Recharge_SMIRNOFF(Target):
    """A custom optimisation target which employs the `openff-recharge`
    package to train bond charge correction parameters against QM derived
    electrostatic potential data."""

    def __init__(self, options, tgt_opts, forcefield):

        if not recharge_import_success:
            warn_once("Note: Failed to import the OpenFF Recharge package - FB Recharge_SMIRNOFF target will not work. ")

        if not toolkit_import_success:
            warn_once("Note: Failed to import the OpenFF Toolkit - FB Recharge_SMIRNOFF target will not work. ")

        super(Recharge_SMIRNOFF, self).__init__(options, tgt_opts, forcefield)

        # Store a mapping between the FB pval parameters and the expected recharge
        # ordering.
        self._parameter_to_bcc_map = None

        # Pre-calculate the expensive portion of the objective function.
        self._design_matrix = None
        self._target_residuals = None

        # Store a copy of the objective function details from the previous
        # optimisation cycle.
        self._molecule_residual_ranges = {}
        self._per_molecule_residuals = {}

        # Get the filename for the database which contains the ESP data
        # to train against.
        self.set_option(tgt_opts, "recharge_esp_store", forceprint=True)
        self.set_option(tgt_opts, "recharge_property", forceprint=True)

        assert self.recharge_property in ["esp", "electric-field"]

        # Initialize the target.
        self._initialize()

    def _initialize(self):
        """Initializes the target."""

        # Load in the ESP data store.
        esp_store = MoleculeESPStore(os.path.join(self.tgtdir, self.recharge_esp_store))

        # Define the molecules to include in the training set.
        smiles = [smiles_pattern for smiles_pattern in esp_store.list()]

        # Determine which BCC parameters are being optimized.
        force_field = smirnoff.ForceField(
            os.path.join(self.FF.ffdir, self.FF.offxml),
            allow_cosmetic_attributes=True,
            load_plugins=True
        )

        bcc_handler = force_field.get_parameter_handler("ChargeIncrementModel")

        if bcc_handler.partial_charge_method.lower() != "am1elf10":
            raise NotImplementedError()

        # TODO: it is assumed that the MDL aromaticity model should be used
        #       rather than the once specified in the FF as the model is not
        #       currently exposed. See OpenFF toolkit issue #663.
        bcc_collection = BCCCollection.from_smirnoff(bcc_handler)
        bcc_smirks = [bcc.smirks for bcc in bcc_collection.parameters]

        # Determine the indices of the BCC parameters being refit.
        bcc_to_parameter_index = {}

        for parameter_index, field_list in enumerate(self.FF.pfields):

            split_key = field_list[0].split("/")

            parameter_tag = split_key[0].strip()
            parameter_smirks = split_key[3].strip()

            if (
                parameter_tag != "ChargeIncrementModel"
                or field_list[3] != "charge_increment1"
            ):
                continue

            bcc_index = bcc_smirks.index(parameter_smirks)
            bcc_to_parameter_index[bcc_index] = parameter_index

        fixed_parameters = [
            i for i in range(len(bcc_smirks)) if i not in bcc_to_parameter_index
        ]

        self._parameter_to_bcc_map = np.array(
            [
                bcc_to_parameter_index[i]
                for i in range(len(bcc_collection.parameters))
                if i not in fixed_parameters
            ]
        )

        # TODO: Currently only AM1 is supported by the SMIRNOFF handler.
        charge_settings = ChargeSettings(theory="am1", symmetrize=True, optimize=True)

        # Pre-calculate the expensive operations which are needed to evaluate the
        # objective function, but do not depend on the current parameters.
        optimization_class = {
            "esp": ESPObjective,
            "electric-field": ElectricFieldObjective,
        }[self.recharge_property]

        objective_terms = [
            objective_term
            for objective_term in optimization_class.compute_objective_terms(
                smiles, esp_store, bcc_collection, fixed_parameters, charge_settings
            )
        ]

        self._design_matrix = np.vstack(
            [objective_term.design_matrix for objective_term in objective_terms]
        )
        self._target_residuals = np.vstack(
            [objective_term.target_residuals for objective_term in objective_terms]
        )

        # Track which residuals map to which molecule.
        residual_counter = 0

        for smiles_pattern in smiles:
            esp_records = esp_store.retrieve(smiles_pattern)

            n_residuals = sum(
                len(esp_record.grid_coordinates) for esp_record in esp_records
            )

            self._molecule_residual_ranges[smiles_pattern] = np.array(
                [i + residual_counter for i in range(n_residuals)]
            )

            residual_counter += len(self._molecule_residual_ranges[smiles_pattern])

    def _compute_gradient_jacobian(self, mvals, perturbation_amount=1.0e-4):
        """Build the matrix which maps the gradient w.r.t. physical parameters to
        a gradient w.r.t mathematical parameters.

        Parameters
        ----------
        mvals: np.ndarray
            The current force balance mathematical parameters.
        perturbation_amount: float
            The amount to perturb the mathematical parameters by
            when calculating the finite difference gradients.
        """

        jacobian_list = []

        for index in range(len(mvals)):

            reverse_mvals = mvals.copy()
            reverse_mvals[index] -= perturbation_amount
            reverse_pvals = np.array(self.FF.make(reverse_mvals))

            forward_mvals = mvals.copy()
            forward_mvals[index] += perturbation_amount
            forward_pvals = np.array(self.FF.make(forward_mvals))

            gradients = (forward_pvals - reverse_pvals) / (2.0 * perturbation_amount)
            jacobian_list.append(gradients)

        # Make sure to restore the FF object back to its original state.
        self.FF.make(mvals)

        jacobian = np.array(jacobian_list)
        return jacobian

    def wq_complete(self):
        return True

    def get(self, mvals, AGrad=True, AHess=True):
        """
        Get the objective function value, gradient, and hessian

        Parameters
        ----------
        mvals: np.ndarray
            mvals array containing the math values of the parameters
        AGrad: bool
            Flag for computing gradients of not
        AHess: bool
            Flag for computing hessian or not

        Returns
        -------
        Answer: dict
            Answer = {'X':obj_value, 'G':obj_grad, 'H':obj_hess}
            obj_value: float
            obj_grad: np.ndarray of shape (n_param, )
            obj_hess: np.ndarray of shape (n_param, n_param)

        Notes
        -----
        1. obj_grad is all zero when AGrad == False
        2. obj_hess is all zero when AHess == False or AGrad == False, because the
           hessian estimate depends on gradients
        """

        # Ensure the input flags are actual booleans.
        AGrad = bool(AGrad)
        AHess = bool(AHess)

        # Extract the current BCC values.
        parameter_values = np.array(self.FF.make(mvals))
        bcc_values = parameter_values[self._parameter_to_bcc_map].reshape(-1, 1)

        if self.recharge_property == "electric-field":
            # Flatten the charges to ensure correct shapes after tensor multiplication.
            bcc_values = bcc_values.flatten()

        # Compute the objective function
        delta = self._target_residuals - np.matmul(self._design_matrix, bcc_values)
        loss = (delta * delta).sum()

        loss_gradient = np.zeros(len(parameter_values))
        loss_hessian = np.zeros((len(parameter_values), len(parameter_values)))

        # Track the per molecule loss as the sum over all conformer
        # contributions
        self._per_molecule_residuals = {
            smiles: (
                delta[self._molecule_residual_ranges[smiles]]
                * delta[self._molecule_residual_ranges[smiles]]
            ).sum()
            for smiles in self._molecule_residual_ranges
        }

        # Save a copy of the per molecule residuals to the temporary directory
        residuals_path = os.path.join(self.root, self.rundir, "residuals.json")

        with open(residuals_path, "w") as file:
            json.dump(self._per_molecule_residuals, file)

        # Compute the objective gradient and hessian.
        if AGrad is True:

            if self.recharge_property == "esp":
                bcc_gradient = -2.0 * np.matmul(self._design_matrix.T, delta)

            elif self.recharge_property == "electric-field":

                bcc_gradient = -2.0 * np.einsum(
                    "ij,ijk->ijk", delta, self._design_matrix
                ).sum(0).sum(0)

            else:
                raise NotImplementedError()

            for bcc_index, parameter_index in enumerate(self._parameter_to_bcc_map):
                loss_gradient[parameter_index] = bcc_gradient[bcc_index]

            gradient_jacobian = self._compute_gradient_jacobian(mvals)
            loss_gradient = np.matmul(gradient_jacobian, loss_gradient)

        if AHess is True:
            loss_hessian = 2.0 * np.outer(loss_gradient * 0.5, loss_gradient * 0.5)

        return {"X": loss, "G": loss_gradient, "H": loss_hessian}

    def indicate(self):
        """Print information to the output file about the last epoch."""

        title = "SMILES\nX"

        dict_for_print = {
            smiles: "%9.3e" % loss
            for smiles, loss in self._per_molecule_residuals.items()
        }

        printcool_dictionary(
            dict_for_print, title=title, bold=True, color=4, keywidth=15
        )
