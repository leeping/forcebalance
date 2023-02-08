""" @package forcebalance.evaluator
A target which employs the OpenFF Evaluator framework to compute condensed
phase physical properties. Currently only force fields in the  OpenFF SMIRNOFF
format are supported.

author Yudong Qiu, Simon Boothroyd, Lee-Ping Wang
@date 06/2019
"""
from __future__ import division, print_function

import json
import os
import re
import tempfile

import numpy as np
from forcebalance.nifty import warn_once, printcool, printcool_dictionary
from forcebalance.output import getLogger
from forcebalance.target import Target

try:
    from openff.evaluator import unit
    from openff.evaluator.attributes import UNDEFINED
    from openff.evaluator.client import EvaluatorClient, ConnectionOptions, RequestOptions
    from openff.evaluator.datasets import PhysicalPropertyDataSet
    from openff.evaluator.utils.exceptions import EvaluatorException
    from openff.evaluator.utils.openmm import openmm_quantity_to_pint
    from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
    from openff.evaluator.forcefield import ParameterGradientKey
    evaluator_import_success = True
except ImportError:
    evaluator_import_success = False

try:
    from openff.toolkit.typing.engines import smirnoff
    toolkit_import_success = True
except ImportError:
    toolkit_import_success = False

logger = getLogger(__name__)


class Evaluator_SMIRNOFF(Target):
    """A custom optimisation target which employs the `openff-evaluator`
    package to rapidly estimate a collection of condensed phase physical
    properties at each optimisation epoch."""

    class OptionsFile:
        """Represents the set of options that a `Evaluator_SMIRNOFF`
        target will be run with.

        Attributes
        ----------
        connection_options: openff.evaluator.client.ConnectionOptions
            The options for how the `evaluator` client should
            connect to a running server instance.
        estimation_options: openff.evaluator.client.RequestOptions
            The options for how properties should be estimated by the
            `evaluator` (e.g. the uncertainties to which properties
            should be estimated).
        data_set_path: str
            The path to a JSON serialized PhysicalPropertyDataSet which
            contains those physical properties which will be optimised
            against.
        weights: dict of float
            The weighting of each property which will be optimised against.
        denominators: dict of str and unit.Quantity
            The denominators will be used to remove units from the properties
            and scale their values.
        polling_interval: float
            The time interval with which to check whether the evaluator has
            finished fulfilling the request (in seconds).
        """

        def __init__(self):

            self.connection_options = ConnectionOptions()
            self.estimation_options = RequestOptions()

            self.data_set_path = ""
            self.weights = {}
            self.denominators = {}

            self.polling_interval = 600

        def to_json(self):
            """Converts this class into a JSON string.

            Returns
            -------
            str
                The JSON representation of this class.
            """

            value = {
                "connection_options": self.connection_options.__getstate__(),
                "estimation_options": self.estimation_options.__getstate__(),
                "data_set_path": self.data_set_path,
                "weights": {
                    property_name: self.weights[property_name]
                    for property_name in self.weights
                },
                "denominators": {
                    property_name: self.denominators[property_name]
                    for property_name in self.denominators
                },
                "polling_interval": self.polling_interval
            }

            return json.dumps(
                value,
                sort_keys=True,
                indent=4,
                separators=(",", ": "),
                cls=TypedJSONEncoder,
            )

        @classmethod
        def from_json(cls, json_source):
            """Creates this class from a JSON string.

            Parameters
            -------
            json_source: str or file-like object
                The JSON representation of this class.
            """

            if isinstance(json_source, str):
                with open(json_source, "r") as file:
                    dictionary = json.load(file, cls=TypedJSONDecoder)
            else:
                dictionary = json.load(json_source, cls=TypedJSONDecoder)

            if "polling_interval" not in dictionary:
                dictionary["polling_interval"] = 600

            assert (
                "connection_options" in dictionary
                and "estimation_options" in dictionary
                and "data_set_path" in dictionary
                and "weights" in dictionary
                and "denominators" in dictionary
                and "polling_interval" in dictionary
            )

            value = cls()

            value.connection_options = ConnectionOptions()
            value.connection_options.__setstate__(dictionary["connection_options"])

            value.estimation_options = RequestOptions()
            value.estimation_options.__setstate__(dictionary["estimation_options"])

            value.data_set_path = dictionary["data_set_path"]

            value.weights = {
                property_name: dictionary["weights"][property_name]
                for property_name in dictionary["weights"]
            }
            value.denominators = {
                property_name: dictionary["denominators"][property_name]
                for property_name in dictionary["denominators"]
            }

            value.polling_interval = dictionary["polling_interval"]

            return value

    def __init__(self, options, tgt_opts, forcefield):

        if not evaluator_import_success:
            warn_once("Note: Failed to import the OpenFF Evaluator - FB Evaluator target will not work. ")

        if not toolkit_import_success:
            warn_once("Note: Failed to import the OpenFF Toolkit - FB Evaluator target will not work. ")

        super(Evaluator_SMIRNOFF, self).__init__(options, tgt_opts, forcefield)

        self._options = None  # The options for this target loaded from JSON.
        self._default_units = (
            {}
        )  # The default units to convert each type of property to.

        self._client = None  # The client object which will communicate with an already spun up server.

        self._reference_data_set = None  # The data set of properties to estimate.
        self._normalised_weights = (
            None  # The normalised weights of the different properties.
        )

        # Store a `Future` like object which can be queried for the results of
        # a property estimation.
        self._pending_estimate_request = None

        # Store a mapping between gradient keys and the force balance string representation.
        self._gradient_key_mappings = {}
        self._parameter_units = {}

        # Store a copy of the objective function details from the previous optimisation cycle.
        self._last_obj_details = {}

        # Get the filename for the evaluator input file.
        self.set_option(tgt_opts, "evaluator_input", forceprint=True)

        # Initialize the target.
        self._initialize()

    def _initialize(self):
        """Initializes the evaluator target from an input json file.

        1. Reads the user specified input file.
        2. Creates a `evaluator` client object.
        3. Loads in a reference experimental data set.
        4. Assigns and normalises weights for each property.
        """

        # Load in the options from a user provided JSON file.
        print(os.path.join(self.tgtdir, self.evaluator_input))
        options_file_path = os.path.join(self.tgtdir, self.evaluator_input)
        self._options = self.OptionsFile.from_json(options_file_path)

        for property_type, denominator in self._options.denominators.items():
            self._default_units[property_type] = denominator.units

        # Attempt to create an evaluator client object using the specified
        # connection options.
        self._client = EvaluatorClient(self._options.connection_options)

        # Load in the experimental data set.
        data_set_path = os.path.join(self.tgtdir, self._options.data_set_path)
        self._reference_data_set = PhysicalPropertyDataSet.from_json(data_set_path)

        if len(self._reference_data_set) == 0:

            raise ValueError(
                "The physical property data set to optimise against is empty."
            )

        # Print the reference data, and count the number of instances of
        # each property type.
        printcool("Loaded experimental data.")

        property_types = self._reference_data_set.property_types

        number_of_properties = {
            x: sum(1 for y in self._reference_data_set.properties_by_type(x))
            for x in property_types
        }

        for substance in self._reference_data_set.substances:

            dict_for_print = {}

            for physical_property in self._reference_data_set.properties_by_substance(
                substance
            ):

                property_type = physical_property.__class__.__name__

                value = physical_property.value.to(self._default_units[property_type])
                uncertainty = np.nan

                if physical_property.uncertainty != UNDEFINED:

                    uncertainty = physical_property.uncertainty.to(
                        self._default_units[property_type]
                    )

                tuple_key = (
                    property_type,
                    physical_property.thermodynamic_state.temperature,
                    physical_property.thermodynamic_state.pressure,
                )

                dict_for_print["%s %s-%s" % tuple_key] = "%s+/-%s" % (
                    value,
                    uncertainty,
                )

            printcool_dictionary(
                dict_for_print, title="Reference %s data" % substance.identifier,
            )

        # Assign and normalize weights for each phase point (average for now)
        self._normalised_weights = {}

        for property_type in self._reference_data_set.property_types:

            self._normalised_weights[property_type] = (
                self._options.weights[property_type]
                / number_of_properties[property_type]
            )

    def _parameter_value_from_gradient_key(self, gradient_key):
        """Extracts the value of the parameter in the current
        open force field object pointed to by a given
        `ParameterGradientKey` object.

        Parameters
        ----------
        gradient_key: openff.evaluator.forcefield.ParameterGradientKey
            The gradient key which points to the parameter of interest.

        Returns
        -------
        unit.Quantity
            The value of the parameter.
        bool
            Returns True if the parameter is a cosmetic one.
        """
        # try:
        #     import openmm.unit as simtk_unit
        # except ImportError:
        #     import simtk.unit as simtk_unit
        from openff.units import unit as openff_unit


        parameter_handler = self.FF.openff_forcefield.get_parameter_handler(
            gradient_key.tag
        )
        parameter = (
            parameter_handler if gradient_key.smirks is None
            else parameter_handler.parameters[gradient_key.smirks]
        )

        attribute_split = re.split(r"(\d+)", gradient_key.attribute)
        attribute_split = list(filter(None, attribute_split))

        parameter_attribute = None
        parameter_value = None

        if hasattr(parameter, gradient_key.attribute):

            parameter_attribute = gradient_key.attribute
            parameter_value = getattr(parameter, parameter_attribute)

        elif len(attribute_split) == 2:

            parameter_attribute = attribute_split[0]

            if hasattr(parameter, parameter_attribute):
                parameter_index = int(attribute_split[1]) - 1

                parameter_value_list = getattr(parameter, parameter_attribute)
                parameter_value = parameter_value_list[parameter_index]

        is_cosmetic = False

        if (
            parameter_attribute is None
            or parameter_attribute in parameter._cosmetic_attribs
        ):
            is_cosmetic = True

        if not isinstance(parameter_value, openff_unit.Quantity):
            parameter_value = parameter_value * openff_unit.dimensionless

        #return openmm_quantity_to_pint(parameter_value), is_cosmetic
        return parameter_value, is_cosmetic

    def _extract_physical_parameter_values(self):
        """Extracts an array of the values of the physical parameters
        (which are not cosmetic) from the current `FF.openff_forcefield`
        object.

        Returns
        -------
        np.ndarray
            The array of values of shape (len(self._gradient_key_mappings),)
        """

        parameter_values = np.zeros(len(self._gradient_key_mappings))

        for gradient_key, parameter_index in self._gradient_key_mappings.items():

            parameter_value, _ = self._parameter_value_from_gradient_key(gradient_key)
            expected_unit = self._parameter_units[gradient_key]

            parameter_values[parameter_index] = parameter_value.to(
                expected_unit
            ).magnitude

        return parameter_values

    def _build_pvals_jacobian(self, mvals, perturbation_amount=1.0e-4):
        """Build the matrix which maps the gradients of properties with
        respect to physical parameters to gradients with respect to
        force balance mathematical parameters.

        Parameters
        ----------
        mvals: np.ndarray
            The current force balance mathematical parameters.
        perturbation_amount: float
            The amount to perturb the mathematical parameters by
            when calculating the finite difference gradients.

        Returns
        -------
        np.ndarray
            A matrix of d(Physical Parameter)/d(Mathematical Parameter).
        """

        jacobian_list = []

        for index in range(len(mvals)):

            reverse_mvals = mvals.copy()
            reverse_mvals[index] -= perturbation_amount

            self.FF.make(reverse_mvals)
            reverse_physical_values = self._extract_physical_parameter_values()

            forward_mvals = mvals.copy()
            forward_mvals[index] += perturbation_amount

            self.FF.make(forward_mvals)
            forward_physical_values = self._extract_physical_parameter_values()

            gradients = (forward_physical_values - reverse_physical_values) / (
                2.0 * perturbation_amount
            )
            jacobian_list.append(gradients)

        # Make sure to restore the FF object back to its original state.
        self.FF.make(mvals)

        jacobian = np.array(jacobian_list)
        return jacobian

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        """
        Submit jobs for evaluating the objective function

        Parameters
        ----------
        mvals: np.ndarray
            mvals array containing the math values of the parameters
        AGrad: bool
            Flag for computing gradients of not
        AHess: bool
            Flag for computing hessian or not

        Notes
        -----
        1. This function is called before wq_complete() and get().
        2. This function should not block.
        """

        # Make the force field based on the current values of the parameters.
        self.FF.make(mvals)

        force_field = smirnoff.ForceField(
            self.FF.offxml, allow_cosmetic_attributes=True, load_plugins=True
        )

        # strip out cosmetic attributes
        with tempfile.NamedTemporaryFile(mode="w", suffix=".offxml") as file:
            force_field.to_file(file.name, discard_cosmetic_attributes=True)
            force_field = smirnoff.ForceField(file.name, load_plugins=True)

        # Determine which gradients (if any) we should be estimating.
        parameter_gradient_keys = []

        self._gradient_key_mappings = {}
        self._parameter_units = {}

        if AGrad is True:

            index_counter = 0

            for field_list in self.FF.pfields:

                string_key = field_list[0]
                key_split = string_key.split("/")

                if len(key_split) == 3 and key_split[0] == "":
                    parameter_tag = key_split[1].strip()
                    parameter_smirks = None
                    parameter_attribute = key_split[2].strip()
                elif len(key_split) == 4:
                    parameter_tag = key_split[0].strip()
                    parameter_smirks = key_split[3].strip()
                    parameter_attribute = key_split[2].strip()
                else:
                    raise NotImplementedError()

                # Use the full attribute name (e.g. k1) for the gradient key.
                parameter_gradient_key = ParameterGradientKey(
                    tag=parameter_tag,
                    smirks=parameter_smirks,
                    attribute=parameter_attribute,
                )

                # Find the unit of the gradient parameter.
                parameter_value, is_cosmetic = self._parameter_value_from_gradient_key(
                    parameter_gradient_key
                )

                if parameter_value is None or is_cosmetic:
                    # We don't wan't gradients w.r.t. cosmetic parameters.
                    continue

                parameter_unit = parameter_value.units
                parameter_gradient_keys.append(parameter_gradient_key)

                self._gradient_key_mappings[parameter_gradient_key] = index_counter
                self._parameter_units[parameter_gradient_key] = parameter_unit

                index_counter += 1

        # Submit the estimation request.
        self._pending_estimate_request, _ = self._client.request_estimate(
            property_set=self._reference_data_set,
            force_field_source=force_field,
            options=self._options.estimation_options,
            parameter_gradient_keys=parameter_gradient_keys,
        )

        logger.info(
            "Requesting the estimation of {} properties, and their "
            "gradients with respect to {} parameters.\n".format(
                len(self._reference_data_set), len(parameter_gradient_keys)
            )
        )

        if (
            self._pending_estimate_request.results(
                True, polling_interval=self._options.polling_interval
            )[0] is None
        ):

            raise RuntimeError(
                "No `EvaluatorServer` could be found to submit the calculations to. "
                "Please double check that a server is running, and that the connection "
                "settings specified in the input script are correct."
            )

    @staticmethod
    def _check_estimation_request(estimation_request):
        """Checks whether an estimation request has finished with any exceptions.

        Parameters
        ----------
        estimation_request: openff.evaluator.client.Request
            The request to check.
        """
        results, _ = estimation_request.results()

        if results is None:
            raise ValueError("Trying to extract the results of an unfinished request.")

        # Check for any exceptions that were raised while estimating
        # the properties.
        if isinstance(results, EvaluatorException):

            raise ValueError(
                "An uncaught exception occured within the evaluator "
                "framework: %s" % str(results)
            )

        if len(results.unsuccessful_properties) > 0:

            exceptions = "\n".join(str(result) for result in results.exceptions)

            raise ValueError(
                "Some properties could not be estimated:\n\n%s." % exceptions
            )

    def _extract_property_data(self, estimation_request, mvals, AGrad):
        """Extract the property estimates #and their gradients#
        from a relevant evaluator request object.

        Parameters
        ----------
        estimation_request: openff.evaluator.client.Request
            The request to extract the data from.

        Returns
        -------
        estimated_data: openff.evaluator.datasets.PhysicalPropertyDataSet
            The data set of estimated properties.
        estimated_gradients: dict of str and np.array
            The estimated gradients in a dictionary with keys of the estimated
            properties unique ids, and values of the properties gradients of shape
            (n_params,).
        """
        # Make sure the request actually finished and was error free.
        Evaluator_SMIRNOFF._check_estimation_request(estimation_request)

        # Extract the results from the request.
        results, _ = estimation_request.results()

        # Save a copy of the results to the temporary directory
        results_path = os.path.join(self.root, self.rundir, "results.json")
        results.json(results_path)

        # Print out some statistics about the calculation
        calculation_layer_counts = {}

        for physical_property in results.estimated_properties:

            calculation_layer = physical_property.source.fidelity

            if calculation_layer not in calculation_layer_counts:
                calculation_layer_counts[calculation_layer] = 0

            calculation_layer_counts[calculation_layer] += 1

        logger.info("\n")

        for layer_type in calculation_layer_counts:

            count = calculation_layer_counts[layer_type]

            logger.info(
                "{} properties were estimated using the {} layer.\n".format(
                    count, layer_type
                )
            )

        logger.info("\n")

        if len(results.exceptions) > 0:

            exceptions = "\n\n".join(str(result) for result in results.exceptions)
            exceptions = exceptions.replace("\\n", "\n")

            # In some cases, an exception will be raised when executing a property but
            # it will not stop the property from being estimated (e.g an error occured
            # while reweighting so a simulation was used to estimate the property
            # instead).
            exceptions_path = os.path.join(
                self.root, self.rundir, "non_fatal_exceptions.txt"
            )

            with open(exceptions_path, "w") as file:
                file.write(exceptions)

            logger.warning(
                "A number of non-fatal exceptions occurred. These were saved to "
                "the %s file." % exceptions_path
            )

        estimated_gradients = {}

        if AGrad is False:
            return results.estimated_properties, estimated_gradients

        jacobian = self._build_pvals_jacobian(mvals)

        # The below snippet will extract any evaluated gradients
        # and map them from gradients with respect to FF parameters,
        # to gradients with respect to FB mathematical parameters.
        for physical_property in results.estimated_properties:

            property_class = physical_property.__class__.__name__

            estimated_gradients[physical_property.id] = np.zeros(
                len(self._gradient_key_mappings)
            )

            for gradient in physical_property.gradients:

                parameter_index = self._gradient_key_mappings[gradient.key]

                gradient_unit = (
                    self._default_units[property_class]
                    / self._parameter_units[gradient.key]
                )

                if isinstance(gradient.value, unit.Quantity):
                    gradient_value = gradient.value.to(gradient_unit).magnitude
                else:
                    gradient_value = gradient.value
                    assert isinstance(gradient_value, float)

                estimated_gradients[physical_property.id][
                    parameter_index
                ] = gradient_value

        for property_id in estimated_gradients:

            pval_gradients = estimated_gradients[property_id]
            mval_gradients = np.matmul(jacobian, pval_gradients)

            estimated_gradients[property_id] = mval_gradients

        return results.estimated_properties, estimated_gradients

    def wq_complete(self):
        """
        Check if all jobs are finished
        This function should have a sleep in it if not finished.

        Returns
        -------
        finished: bool
            True if all jobs are finished, False if not
        """

        estimation_results, _ = self._pending_estimate_request.results()

        return (
            isinstance(estimation_results, EvaluatorException)
            or len(estimation_results.queued_properties) == 0
        )

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

        # Extract the properties estimated using the unperturbed parameters.
        estimated_data_set, estimated_gradients = self._extract_property_data(
            self._pending_estimate_request, mvals, AGrad
        )

        # compute objective value
        obj_value = 0.0
        obj_grad = np.zeros(self.FF.np)
        obj_hess = np.zeros((self.FF.np, self.FF.np))

        # store details for printing
        self._last_obj_details = {}

        for property_type in self._reference_data_set.property_types:

            self._last_obj_details[property_type] = []

            denominator = (
                self._options.denominators[property_type]
                .to(self._default_units[property_type])
                .magnitude
            )

            weight = self._normalised_weights[property_type]

            for reference_property in self._reference_data_set.properties_by_type(
                property_type
            ):

                reference_value = reference_property.value.to(
                    self._default_units[property_type]
                ).magnitude

                target_property = next(
                    x for x in estimated_data_set if x.id == reference_property.id
                )
                target_value = target_property.value.to(
                    self._default_units[property_type]
                ).magnitude

                target_error = np.nan

                if target_property.uncertainty != UNDEFINED:

                    target_error = target_property.uncertainty.to(
                        self._default_units[property_type]
                    ).magnitude

                diff = target_value - reference_value

                obj_contrib = weight * (diff / denominator) ** 2
                obj_value += obj_contrib

                temperature = reference_property.thermodynamic_state.temperature
                pressure = reference_property.thermodynamic_state.pressure

                self._last_obj_details[property_type].append(
                    (
                        temperature.to(unit.kelvin),
                        pressure.to(unit.atmosphere),
                        target_property.substance.identifier,
                        reference_value,
                        target_value,
                        target_error,
                        diff,
                        weight,
                        denominator,
                        obj_contrib,
                    )
                )

                # compute objective gradient
                if AGrad is True:

                    # get gradients in physical unit
                    grad_array = estimated_gradients[reference_property.id]
                    # compute objective gradient
                    obj_grad += 2.0 * weight * diff * grad_array / denominator ** 2

                    if AHess is True:
                        obj_hess += (
                            2.0
                            * weight
                            * (np.outer(grad_array, grad_array))
                            / denominator ** 2
                        )

        return {"X": obj_value, "G": obj_grad, "H": obj_hess}

    def indicate(self):
        """
        print information into the output file about the last objective function evaluated
        This function should be called after get()
        """
        for property_name, details in self._last_obj_details.items():
            dict_for_print = {
                "  %s %s %s"
                % detail[:3]: "%9.3f %14.3f +- %-7.3f % 7.3f % 9.5f % 9.5f % 9.5f "
                % detail[3:]
                for detail in details
            }
            title = (
                "%s %s\nTemperature  Pressure Substance  Reference  Calculated +- "
                "Stdev     Delta    Weight    Denom     Term  " % (self.name, property_name)
            )
            printcool_dictionary(
                dict_for_print, title=title, bold=True, color=4, keywidth=15
            )
