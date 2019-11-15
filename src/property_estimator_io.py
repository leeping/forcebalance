""" @package forcebalance.property_estimator_io
For interfacing with property estimator to compute liquid properties.
Currently only support OpenForceField SMIRNOFF force field

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

logger = getLogger(__name__)

try:
    import propertyestimator
    from propertyestimator import unit
    from propertyestimator.client import PropertyEstimatorClient, ConnectionOptions, PropertyEstimatorOptions
    from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
    from propertyestimator.utils.exceptions import PropertyEstimatorException
    from propertyestimator.utils.openmm import openmm_quantity_to_pint
    from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
    from propertyestimator.workflow import WorkflowOptions
    from propertyestimator.properties import ParameterGradientKey
except ImportError:
    warn_once("Failed to import the propertyestimator package.")

try:
    from openforcefield.typing.engines import smirnoff
except ImportError:
    warn_once("Failed to import the openforcefield package.")


class PropertyEstimate_SMIRNOFF(Target):
    """A custom optimisation target which employs the propertyestimator
    package to rapidly estimate a collection of condensed phase physical
    properties at each optimisation epoch."""

    class OptionsFile:
        """Represents the set of options that a `PropertyEstimate_SMIRNOFF`
        target will be run with.

        Attributes
        ----------
        connection_options: propertyestimator.client.ConnectionOptions
            The options for how the `propertyestimator` client should
            connect to a running server instance.
        estimation_options: propertyestimator.client.PropertyEstimatorOptions
            The options for how properties should be estimated by the
            `propertyestimator` (e.g. the uncertainties to which properties
            should be estimated).
        data_set_path: str
            The path to a JSON serialized PhysicalPropertyDataSet which
            contains those physical properties which will be optimised
            against.
        weights: list of PropertyWeight
            The weighting of each property which will be optimised against.
        denominators: list of PropertyDenominator
            The denominators will be used to remove units from the properties
            and scale their values.
        """

        def __init__(self):

            self.connection_options = ConnectionOptions()
            self.estimation_options = PropertyEstimatorOptions()

            self.data_set_path = ''
            self.weights = {}
            self.denominators = {}

        def to_json(self):
            """Converts this class into a JSON string.

            Returns
            -------
            str
                The JSON representation of this class.
            """

            value = {
                'connection_options': self.connection_options.__getstate__(),
                'estimation_options': self.estimation_options.__getstate__(),

                'data_set_path': self.data_set_path,

                'weights': {
                    property_name: self.weights[property_name] for property_name in self.weights
                },
                'denominators': {
                    property_name: self.denominators[property_name] for property_name in self.denominators
                }
            }

            return json.dumps(value, sort_keys=True, indent=4, separators=(',', ': '), cls=TypedJSONEncoder)

        @classmethod
        def from_json(cls, json_source):
            """Creates this class from a JSON string.

            Parameters
            -------
            json_source: str or file-like object
                The JSON representation of this class.
            """

            if isinstance(json_source, str):
                with open(json_source, 'r') as file:
                    dictionary = json.load(file, cls=TypedJSONDecoder)
            else:
                dictionary = json.load(json_source, cls=TypedJSONDecoder)

            assert ('connection_options' in dictionary and
                    'estimation_options' in dictionary and
                    'data_set_path' in dictionary and
                    'weights' in dictionary and
                    'denominators' in dictionary)

            value = cls()

            value.connection_options = ConnectionOptions()
            value.connection_options.__setstate__(dictionary['connection_options'])

            value.estimation_options = PropertyEstimatorOptions()
            value.estimation_options.__setstate__(dictionary['estimation_options'])

            value.data_set_path = dictionary['data_set_path']

            value.weights = {
                property_name: dictionary['weights'][property_name] for property_name in dictionary['weights']
            }
            value.denominators = {
                property_name: dictionary['denominators'][property_name] for property_name in dictionary['denominators']
            }

            return value

    # A dictionary of the units that force balance expects each property in.
    @property
    def default_units(self):
        return {
            'Density': unit.kilogram / unit.meter ** 3,
            'DielectricConstant': unit.dimensionless,
            'EnthalpyOfVaporization': unit.kilojoules / unit.mole
        }

    def __init__(self, options, tgt_opts, forcefield):

        super(PropertyEstimate_SMIRNOFF, self).__init__(options, tgt_opts, forcefield)

        self._options = None  # The options for this target loaded from JSON.
        self._client = None  # The client object which will communicate with an already spun up server.

        self._data_set = None  # The data set of properties to estimate.
        self._reference_properties = {}  # The data set in a forcebalance compatible format.

        self._normalised_weights = None  # The normalised weights of the different properties.

        # Store a `Future` like object which can be queried for the results of
        # a property estimation.
        self._pending_estimate_request = None

        # Store a mapping between gradient keys and the force balance string representation.
        self._gradient_key_mappings = {}
        self._parameter_units = {}

        # Store a copy of the objective function details from the previous optimisation cycle.
        self._last_obj_details = {}

        # Get the filename for the property estimator input file.
        self.set_option(tgt_opts, 'prop_est_input', forceprint=True)

        # Initialize the target.
        self._initialize()

    def _refactor_properties_dictionary(self, properties):
        """Refactors a property dictionary of the form
        `property = dict[substance_id][property_index]` to one of the form
        `property = dict[property_type][substance_id][state_tuple]['value' or 'uncertainty']` where
        the state tuple is equal to (temperature in K .6f, pressure in atm .6f).

        Parameters
        ----------
        properties: dict of str and list of PhysicalProperty
            The original dictionary of properties.

        Returns
        -------
        refactored_properties: dict[str][str][tuple][str] : float
            The refactored properties dictionary.
        """

        refactored_properties = {}

        for substance_id in properties:

            for physical_property in properties[substance_id]:

                class_name = physical_property.__class__.__name__

                temperature = physical_property.thermodynamic_state.temperature.to(unit.kelvin).magnitude
                pressure = physical_property.thermodynamic_state.pressure.to(unit.atmosphere).magnitude

                state_tuple = ('%.6f' % temperature, '%.6f' % pressure)

                default_unit = self.default_units[class_name]

                value = physical_property.value.to(default_unit).magnitude
                uncertainty = physical_property.uncertainty.to(default_unit).magnitude

                if class_name not in refactored_properties:
                    refactored_properties[class_name] = {}

                if substance_id not in refactored_properties[class_name]:
                    refactored_properties[class_name][substance_id] = {}

                refactored_properties[class_name][substance_id][state_tuple] = {
                    'value': value,
                    'uncertainty': uncertainty
                }

        return refactored_properties

    def _initialize(self):
        """Initializes the property estimator target from an input json file.

        1. Reads the user specified input file.
        2. Creates a `propertyestimator` client object.
        3. Loads in a reference experimental data set.
        4. Assigns and normalises weights for each property.
        """

        # Load in the options from a user provided JSON file.
        print(os.path.join(self.tgtdir, self.prop_est_input))
        options_file_path = os.path.join(self.tgtdir, self.prop_est_input)
        self._options = self.OptionsFile.from_json(options_file_path)

        # Attempt to create a property estimator client object using the specified
        # connection options.
        self._client = PropertyEstimatorClient(connection_options=self._options.connection_options)

        # Load in the experimental data set.
        data_set_path = os.path.join(self.tgtdir, self._options.data_set_path)

        with open(data_set_path, 'r') as file:
            self._data_set = PhysicalPropertyDataSet.parse_json(file.read())

        if len(self._data_set.properties) == 0:
            raise ValueError('The physical property data set to optimise against is empty.')

        # Convert the reference data into a more easily comparable form.
        self._reference_properties = self._refactor_properties_dictionary(self._data_set.properties)

        # Print the reference data, and count the number of instances of
        # each property type.
        printcool("Loaded experimental data from property estimator")

        number_of_properties = {property_name: 0.0 for property_name in self._reference_properties}

        for property_name in self._reference_properties:

            for substance_id in self._reference_properties[property_name]:

                dict_for_print = {}

                for state_tuple in self._reference_properties[property_name][substance_id]:

                    value = self._reference_properties[property_name][substance_id][state_tuple]['value']
                    uncertainty = self._reference_properties[property_name][substance_id][state_tuple]['uncertainty']

                    dict_for_print["%sK-%satm" % state_tuple] = ("%f+/-%f" % (value, uncertainty))

                    number_of_properties[property_name] += 1.0

                printcool_dictionary(dict_for_print, title="Reference %s (%s) data" % (property_name, substance_id))

        # Assign and normalize weights for each phase point (average for now)
        self._normalised_weights = {}

        for property_name in self._reference_properties:

            self._normalised_weights[property_name] = (self._options.weights[property_name] /
                                                       number_of_properties[property_name])

    def _parameter_value_from_gradient_key(self, gradient_key):
        """Extracts the value of the parameter in the current
        open force field object pointed to by a given
        `ParameterGradientKey` object.

        Parameters
        ----------
        gradient_key: ParameterGradientKey
            The gradient key which points to the parameter of interest.

        Returns
        -------
        unit.Quantity
            The value of the parameter.
        bool
            Returns True if the parameter is a cosmetic one.
        """

        parameter_handler = self.FF.openff_forcefield.get_parameter_handler(gradient_key.tag)
        parameter = parameter_handler.parameters[gradient_key.smirks]

        attribute_split = re.split(r'(\d+)', gradient_key.attribute)
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

        if parameter_attribute is None or parameter_attribute in parameter._cosmetic_attribs:
            is_cosmetic = True

        return openmm_quantity_to_pint(parameter_value), is_cosmetic

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

            parameter_values[parameter_index] = parameter_value.to(expected_unit).magnitude

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

            gradients = (forward_physical_values - reverse_physical_values) / (2.0 * perturbation_amount)
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

        force_field = smirnoff.ForceField(self.FF.offxml, allow_cosmetic_attributes=True)

        # strip out cosmetic attributes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.offxml') as file:
            force_field.to_file(file.name, discard_cosmetic_attributes=True)
            force_field = smirnoff.ForceField(file.name)

        # Determine which gradients (if any) we should be estimating.
        parameter_gradient_keys = []

        self._gradient_key_mappings = {}
        self._parameter_units = {}

        if AGrad is True:

            index_counter = 0

            for field_list in self.FF.pfields:

                string_key = field_list[0]
                key_split = string_key.split('/')

                parameter_tag = key_split[0].strip()
                parameter_smirks = key_split[3].strip()
                parameter_attribute = key_split[2].strip()

                # Use the full attribute name (e.g. k1) for the gradient key.
                parameter_gradient_key = ParameterGradientKey(tag=parameter_tag,
                                                              smirks=parameter_smirks,
                                                              attribute=parameter_attribute)

                # Find the unit of the gradient parameter.
                parameter_value, is_cosmetic = self._parameter_value_from_gradient_key(parameter_gradient_key)

                if parameter_value is None or is_cosmetic:
                    # We don't wan't gradients w.r.t. cosmetic parameters.
                    continue

                parameter_unit = parameter_value.units
                parameter_gradient_keys.append(parameter_gradient_key)

                self._gradient_key_mappings[parameter_gradient_key] = index_counter
                self._parameter_units[parameter_gradient_key] = parameter_unit

                index_counter += 1

        # Submit the estimation request.
        self._pending_estimate_request = self._client.request_estimate(property_set=self._data_set,
                                                                       force_field_source=force_field,
                                                                       options=self._options.estimation_options,
                                                                       parameter_gradient_keys=parameter_gradient_keys)

        logger.info('Requesting the estimation of {} properties, and their '
                    'gradients with respect to {} parameters.\n'.format(self._data_set.number_of_properties,
                                                                        len(parameter_gradient_keys)))

        if self._pending_estimate_request.results(True) is None:

            raise RuntimeError('No `PropertyEstimatorServer` could be found to submit the '
                               'calculations to. Please double check that a server is running, '
                               'and that the connection settings specified in the input script '
                               'are correct.')

    @staticmethod
    def _check_estimation_request(estimation_request):
        """Checks whether a property estimation request has finished
        with any exceptions.

        Parameters
        ----------
        estimation_request: PropertyEstimatorClient.Request
            The request to check.
        """
        results = estimation_request.results()

        if results is None:
            raise ValueError('Trying to extract the results of an unfinished request.')

        # Check for any exceptions that were raised while estimating
        # the properties.
        if isinstance(results, PropertyEstimatorException):
            raise ValueError('An uncaught exception occured within the property '
                             'estimator (directory=%s: %s' % (results.directory, results.message))

        if len(results.unsuccessful_properties) > 0:

            exceptions = '\n'.join('%s: %s' % (result.directory, result.message) for result in results.exceptions)

            raise ValueError('Some properties could not be estimated:\n\n%s.' % exceptions)

        elif len(results.exceptions) > 0:

            exceptions = '\n'.join('%s: %s' % (result.directory, result.message) for result in results.exceptions)

            # In some cases, an exception will be raised when executing a property but
            # it will not stop the property from being estimated (e.g an error occured
            # while reweighting so a simulation was used to estimate the property instead).
            logger.warning('A number of non-fatal exceptions occured:\n\n%s.' % exceptions)

    def _extract_property_data(self, estimation_request, mvals, AGrad):
        """Extract the property estimates #and their gradients#
        from a relevant property estimator request object.

        Parameters
        ----------
        estimation_request: PropertyEstimatorClient.Request
            The request to extract the data from.

        Returns
        -------
        estimated_data: dict[str][str][tuple][str] : float
            The estimated properties in a dictionary of the form
            `estimated_data = dict[property_type][substance_id][state_tuple]['value' or 'uncertainty']`
        estimated_gradients: dict[str][str][tuple]: np.array of shape (n_params,)
            The estimated gradients in a dictionary.
            `estimated_gradients = dict[property_type][substance_id][state_tuple] = mval_gradients`
        """
        # Make sure the request actually finished and was error free.
        PropertyEstimate_SMIRNOFF._check_estimation_request(estimation_request)

        # Extract the results from the request.
        results = estimation_request.results()

        estimated_data = self._refactor_properties_dictionary(results.estimated_properties)
        estimated_gradients = {}

        if AGrad is False:
            return estimated_data, estimated_gradients

        jacobian = self._build_pvals_jacobian(mvals)

        # The below snippet will extract any property estimator calculated
        # gradients.
        for substance_id in results.estimated_properties:

            for physical_property in results.estimated_properties[substance_id]:

                # Convert the property estimator properties list into
                # a force balance dictionary.
                class_name = physical_property.__class__.__name__

                # Pull out any estimated gradients.
                if class_name not in estimated_gradients:
                    estimated_gradients[class_name] = {}

                if substance_id not in estimated_gradients[class_name]:
                    estimated_gradients[class_name][substance_id] = {}

                temperature = physical_property.thermodynamic_state.temperature.to(unit.kelvin).magnitude
                pressure = physical_property.thermodynamic_state.pressure.to(unit.atmosphere).magnitude

                state_tuple = ('%.6f' % temperature, '%.6f' % pressure)

                if state_tuple not in estimated_gradients[class_name][substance_id]:

                    estimated_gradients[class_name][substance_id][state_tuple] = \
                        np.zeros(len(self._gradient_key_mappings))

                logger.info('Gradients:\n\n')

                for gradient in physical_property.gradients:

                    parameter_index = self._gradient_key_mappings[gradient.key]
                    gradient_unit = self.default_units[class_name] / self._parameter_units[gradient.key]

                    logger.info('%s\n' % str(gradient))

                    if isinstance(gradient.value, unit.Quantity):
                        gradient_value = gradient.value.to(gradient_unit).magnitude
                    else:
                        gradient_value = gradient.value
                        assert isinstance(gradient_value, float)

                    estimated_gradients[class_name][substance_id][state_tuple][parameter_index] = gradient_value

        for property_type in estimated_gradients:

            for substance_id in estimated_gradients[property_type]:

                for state_tuple in estimated_gradients[property_type][substance_id]:

                    pval_gradients = estimated_gradients[property_type][substance_id][state_tuple]
                    mval_gradients = np.matmul(jacobian, pval_gradients)

                    estimated_gradients[property_type][substance_id][state_tuple] = mval_gradients

        return estimated_data, estimated_gradients

    def wq_complete(self):
        """
        Check if all jobs are finished
        This function should have a sleep in it if not finished.

        Returns
        -------
        finished: bool
            True if all jobs are finished, False if not
        """

        estimation_results = self._pending_estimate_request.results()

        return (isinstance(estimation_results, PropertyEstimatorException) or
                len(estimation_results.queued_properties) == 0)

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
        2. obj_hess is all zero when AHess == False or AGrad == False, because the hessian estimate
        depends on gradients
        """

        # Ensure the input flags are actual booleans.
        AGrad = bool(AGrad)
        AHess = bool(AHess)

        # Extract the properties estimated using the unperturbed parameters.
        estimated_property_data, estimated_gradients = self._extract_property_data(self._pending_estimate_request,
                                                                                   mvals, AGrad)

        # compute objective value
        obj_value = 0.0
        obj_grad = np.zeros(self.FF.np)
        obj_hess = np.zeros((self.FF.np, self.FF.np))

        # store details for printing
        self._last_obj_details = {}

        for property_name in self._reference_properties:

            self._last_obj_details[property_name] = []

            denominator = self._options.denominators[property_name]
            weight = self._normalised_weights[property_name]

            for substance_id in self._reference_properties[property_name]:

                for phase_point_key in self._reference_properties[property_name][substance_id]:

                    temperature, pressure = phase_point_key
                    reference_value = self._reference_properties[property_name][substance_id][phase_point_key]['value']

                    target_value = estimated_property_data[property_name][substance_id][phase_point_key]['value']
                    target_error = estimated_property_data[property_name][substance_id][phase_point_key]['uncertainty']

                    diff = target_value - reference_value

                    obj_contrib = weight * (diff / denominator) ** 2
                    obj_value += obj_contrib

                    self._last_obj_details[property_name].append((temperature,
                                                                  pressure,
                                                                  substance_id,
                                                                  reference_value,
                                                                  target_value,
                                                                  target_error,
                                                                  diff,
                                                                  weight,
                                                                  denominator,
                                                                  obj_contrib))

                    # compute objective gradient
                    if AGrad is True:

                        # get gradients in physical unit
                        grad_array = estimated_gradients[property_name][substance_id][phase_point_key]
                        # compute objective gradient
                        obj_grad += 2.0 * weight * diff * grad_array / denominator ** 2

                        if AHess is True:
                            obj_hess += 2.0 * weight * (np.outer(grad_array, grad_array)) / denominator ** 2

        return {'X': obj_value, 'G': obj_grad, 'H': obj_hess}

    def indicate(self):
        """
        print information into the output file about the last objective function evaluated
        This function should be called after get()
        """
        for property_name, details in self._last_obj_details.items():
            dict_for_print = {
                "  %sK %satm %s" % detail[:3]: "%9.3f %14.3f +- %-7.3f % 7.3f % 9.5f % 9.5f % 9.5f " % detail[3:] for
                detail in details}
            title = '%s %s\nTemperature  Pressure Substance  Reference  Calculated +- Stdev     Delta    Weight    ' \
                    'Denom     Term  ' % (self.name, property_name)
            printcool_dictionary(dict_for_print, title=title, bold=True, color=4, keywidth=15)
