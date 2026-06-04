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
    from openff.recharge.charges.qc import QCChargeSettings
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
    electrostatic potential data.
    
    Note -- this has NOT been written to work with anything but BCCs. It will
    refuse a force field whose virtual sites carry charge (see
    ``_check_no_virtual_site_charges``); charge-free virtual sites are ignored.
    """

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
        self._reference_values = None

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

    @staticmethod
    def _check_no_virtual_site_charges(force_field):
        """Raise ``NotImplementedError`` if ``force_field`` defines virtual sites
        that carry charge.

        The target only places charge on atoms (AM1 base charges perturbed by bond
        charge corrections) and never passes a virtual site collection to
        ``openff-recharge``. A virtual site with a non-zero ``charge_increment``
        would contribute to the electrostatic potential in a way this target cannot
        represent, and would otherwise be silently ignored - biasing the fitted
        bond charge corrections. Virtual sites whose charge increments are all zero
        do not affect the ESP and are therefore permitted.
        """

        vsite_handler = force_field.get_parameter_handler("VirtualSites")

        for parameter in vsite_handler.parameters:
            if any(
                getattr(charge_increment, "m", charge_increment) != 0.0
                for charge_increment in parameter.charge_increment
            ):
                raise NotImplementedError(
                    "The Recharge_SMIRNOFF target does not support virtual sites "
                    "that carry charge (VirtualSites parameter '{}' has a non-zero "
                    "charge_increment). Only bond charge corrections on atoms can "
                    "be trained.".format(parameter.smirks)
                )

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
        
        # The target only models charge on atoms (AM1 base charges perturbed by
        # bond charge corrections); a virtual site that carries charge cannot be
        # represented and would otherwise be silently ignored, biasing the fit.
        self._check_no_virtual_site_charges(force_field)

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

        # The BCC parameters being optimized, identified by their SMIRKS pattern.
        # ``compute_objective_terms`` now expects the SMIRKS of the *trainable*
        # parameters (previously it expected the indices of the *fixed* ones). The
        # order of these keys defines the column ordering of the design matrix, so it
        # must match the ``_parameter_to_bcc_map`` used to map BCCs back to FB params.
        trainable_bcc_indices = [
            i for i in range(len(bcc_smirks)) if i in bcc_to_parameter_index
        ]
        bcc_parameter_keys = [bcc_smirks[i] for i in trainable_bcc_indices]

        self._parameter_to_bcc_map = np.array(
            [bcc_to_parameter_index[i] for i in trainable_bcc_indices]
        )

        # TODO: Currently only AM1 is supported by the SMIRNOFF handler.
        charge_settings = QCChargeSettings(
            theory="am1", symmetrize=True, optimize=True
        )

        # Pre-calculate the expensive operations which are needed to evaluate the
        # objective function, but do not depend on the current parameters.
        optimization_class = {
            "esp": ESPObjective,
            "electric-field": ElectricFieldObjective,
        }[self.recharge_property]

        # Retrieve the ESP records to train against. They are gathered per molecule so
        # that the ordering matches the per-molecule residual ranges computed below.
        esp_records = [
            esp_record
            for smiles_pattern in smiles
            for esp_record in esp_store.retrieve(smiles_pattern)
        ]

        objective_terms = [
            objective_term
            for objective_term in optimization_class.compute_objective_terms(
                esp_records,
                charge_collection=charge_settings,
                bcc_collection=bcc_collection,
                bcc_parameter_keys=bcc_parameter_keys,
            )
        ]

        self._design_matrix = np.vstack(
            [objective_term.atom_charge_design_matrix for objective_term in objective_terms]
        )
        self._reference_values = np.vstack(
            [objective_term.reference_values for objective_term in objective_terms]
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
        delta = self._reference_values - np.matmul(self._design_matrix, bcc_values)
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

            # Flatten to a 1-D vector of one gradient per BCC. The ESP branch yields
            # a column vector of shape (n_bcc, 1); older NumPy used to silently squeeze
            # the resulting (1,)-shaped slices into the scalar assignment below, but
            # newer NumPy raises instead, so squeeze explicitly here.
            bcc_gradient = np.asarray(bcc_gradient).reshape(-1)

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
