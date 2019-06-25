""" @package forcebalance.property_estimator_io
For interfacing with property estimator to compute liquid properties.
Currently only support OpenForceField SMIRNOFF force field

author Yudong Qiu, Simon Boothroyd, Lee-Ping Wang
@date 06/2019
"""
from __future__ import division, print_function

import json
import os

import numpy as np
from forcebalance.nifty import warn_once, printcool, printcool_dictionary
from forcebalance.output import getLogger
from forcebalance.target import Target
from openforcefield.typing.engines import smirnoff
from propertyestimator.client import PropertyEstimatorClient, ConnectionOptions, PropertyEstimatorOptions
from propertyestimator.datasets import ThermoMLDataSet
from propertyestimator.utils.exceptions import PropertyEstimatorException
from simtk import unit

logger = getLogger(__name__)

try:
    import propertyestimator
except ImportError:
    # TODO: Shouldn't this be allowed to raise an exception,
    #       or at least signal in some way to force balance
    #       that it isn't available?
    warn_once("Failed to import the propertyestimator package.")


class PropertyEstimate_SMIRNOFF(Target):
    """A custom optimisation target which employs the propertyestimator
    package to rapidly estimate a collection of condensed phase physical
    properties at each optimisation epoch."""

    class OptionsFile:
        """Represents the set of options that a `PropertyEstimate_SMIRNOFF`
        target will be run with.

        Attributes
        ----------
        client_config: OptionsFile.ClientConfiguration
            Configuration options for a `propertyestimator` client.
        sources: list of OptionsFile.ThermoMLPropertySource
            A list of sources to the properties which this target aims
            to fit against.
        weights: list of PropertyWeight
            The weighting of each property which will be optimised against.
        denominators: list of PropertyDenominator
            ???
        """

        class ClientConfiguration:

            def __init__(self, server_address="localhost", server_port=8000):
                """Constructs a new ClientConfiguration object.

                Parameters
                ----------
                server_address: str
                    The address of a running `propertyestimator` server.
                server_port: int
                    The port of a running `propertyestimator` server.
                """

                assert server_address is not None and server_port is not None

                self.server_address = server_address
                self.server_port = server_port

            def to_dict(self):
                """Converts this object to a dictionary

                Returns
                -------
                dict of str and str
                    The dictionary representation of this class.
                """
                return {'server_address': self.server_address, 'server_port': self.server_port}

            @classmethod
            def from_dict(cls, dictionary):
                """Creates an instance of this class from a dictionary.

                Parameters
                ----------
                dict of str and str
                    The dictionary representation to build the class from.
                """
                return cls(dictionary.get('server_address'), dictionary.get('server_port'))

        class ThermoMLPropertySource:
            """Represents the source (either a path or a doi) of
            a ThermoML property.
            """

            @property
            def path(self):
                """str: A path to the ThermoML property .xml file."""
                return self._path

            @property
            def doi(self):
                """str: The doi of a ThermoML property .xml file."""
                return self._doi

            def __init__(self, path=None, doi=None):
                """Constructs a new ThermoMLPropertySource object.

                Parameters
                ----------
                path: str, optional
                    A path to the ThermoML property .xml file.
                doi: str, optional
                    The doi of a ThermoML property .xml file.
                """

                assert (path is None and doi is not None or
                        path is not None and doi is None)

                self._path = path
                self._doi = doi

            def to_dict(self):
                """Converts this object to a dictionary

                Returns
                -------
                dict of str and str
                    The dictionary representation of this class.
                """

                if self.path is not None:
                    return {'path': self.path}
                else:
                    return {'doi': self.doi}

            @classmethod
            def from_dict(cls, dictionary):
                """Creates an instance of this class from a dictionary.

                Parameters
                ----------
                dict of str and str
                    The dictionary representation to build the class from.
                """
                return cls(dictionary.get('path'), dictionary.get('doi'))

        def __init__(self):

            self.client_config = self.ClientConfiguration()
            self.sources = []
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
                'client_config': self.client_config.to_dict(),
                'sources': [source.to_dict() for source in self.sources],

                'weights': {
                    property_name: self.weights[property_name] for property_name in self.weights
                },
                'denominators': {
                    property_name: self.denominators[property_name] for property_name in self.denominators
                }
            }

            return json.dumps(value, sort_keys=True, indent=4, separators=(',', ': '))

        @classmethod
        def from_json(cls, json_string):
            """Creates this class from a JSON string.

            Parameters
            -------
            json_string
                The JSON representation of this class.
            """

            dictionary = json.loads(json_string)

            assert ('client_config' in dictionary and
                    'sources' in dictionary and
                    'weights' in dictionary and
                    'denominators' in dictionary)

            value = cls()

            value.client_config = cls.ClientConfiguration.from_dict(dictionary['client_config'])
            value.sources = [cls.ThermoMLPropertySource.from_dict(source) for source in dictionary['sources']]

            value.weights = {
                property_name: dictionary['weights'][property_name] for property_name in dictionary['weights']
            }
            value.denominators = {
                property_name: dictionary['denominators'][property_name] for property_name in dictionary['denominators']
            }

            return value

    # A mapping of property estimator property names to
    # force balance names.
    estimator_to_force_balance_map = {
        'Density': 'density',
        'DielectricConstant': 'dielectric'
    }

    # A mapping of force balance property names to
    # property estimator names.
    force_balance_to_estimator_map = {
        'density': 'Density',
        'dielectric': 'DielectricConstant'
    }

    # A dictionary of the units that force balance expects each property in.
    default_units = {
        'Density': unit.kilogram / unit.meter ** 3,
        'DielectricConstant': unit.dimensionless
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

        self._pending_gradient_requests = {}

        # Store a map between a force balance parameter index and a property estimator
        # `ParameterGradientKey`, as well as the inverse.
        # self._parameter_index_to_key = None
        # self._parameter_key_to_index = None

        # Store a copy of the objective function details from the previous optimisation cycle.
        self._last_obj_details = {}

        # Get the filename for the property estimator input file.
        self.set_option(tgt_opts, 'prop_est_input', forceprint=True)
        # Finite difference size for numerical gradients
        self.set_option(tgt_opts, 'liquid_fdiff_h', forceprint=True)

        # Initialize the target.
        self._initialize()

    @staticmethod
    def _property_to_key_value(physical_property):
        """Converts a property estimator `PhysicalProperty` object
        into a force balance key-value pair.

        TODO: Currently nothing actually stores the substance for
              which a property was calculated (e.g. water, methanol)

        Parameters
        ----------
        physical_property: propertyestimator.properties.PhysicalProperty
            The physical property to encode.

        Returns
        -------
        tuple
            The phase point key.
        float
            The property value.
        float
            The uncertainty in the value.
        """
        class_name = physical_property.__class__.__name__

        phase_point_key = (physical_property.thermodynamic_state.temperature.value_in_unit(unit.kelvin),
                           physical_property.thermodynamic_state.pressure.value_in_unit(unit.atmosphere))

        value = physical_property.value.value_in_unit(PropertyEstimate_SMIRNOFF.default_units[class_name])
        uncertainty = physical_property.uncertainty.value_in_unit(PropertyEstimate_SMIRNOFF.default_units[class_name])

        return phase_point_key, value, uncertainty

    def _initialize(self):
        """Initializes the property estimator target from an input json file.

        1. Reads the user specified input file.
        2. Creates a `propertyestimator` client object.
        3. Loads in a reference experimental data set.
        4. Assigns and normalises weights for each property.
        """

        # Load in the options from a user provided JSON file.
        with open(os.path.join(self.tgtdir, self.prop_est_input), 'r') as file:
            self._options = self.OptionsFile.from_json(file.read())

        # Attempt to create a property estimator client object using the specified
        # connection options.
        connection_options = ConnectionOptions(server_address=self._options.client_config.server_address,
                                               server_port=self._options.client_config.server_port)

        self._client = PropertyEstimatorClient(connection_options=connection_options)

        # Load in the experimental data set from either the specified doi or
        # local file paths.
        self._data_set = ThermoMLDataSet.from_file(*[source.path for source in
                                                     self._options.sources if source.path])

        self._data_set.merge(ThermoMLDataSet.from_doi(*[source.doi for source in
                                                        self._options.sources if source.doi]))

        if len(self._data_set.properties) == 0:
            raise ValueError('The physical property data set to optimise against is empty. '
                             'Either no physical properties were specified, or those that '
                             'were are unsupported by the estimator.')

        # Convert the reference data into a format that forcebalance can understand
        self._reference_properties = {}

        for substance_identifier in self._data_set.properties:

            for physical_property in self._data_set.properties[substance_identifier]:

                class_name = physical_property.__class__.__name__
                mapped_name = PropertyEstimate_SMIRNOFF.estimator_to_force_balance_map[class_name]

                if mapped_name not in self._reference_properties:
                    self._reference_properties[mapped_name] = {}

                key, value, _ = self._property_to_key_value(physical_property)
                self._reference_properties[mapped_name][key] = value

        # Print the reference data
        printcool("Loaded experimental data from property estimator")

        for property_name, property_data in self.ref_data.items():
            dict_for_print = {("%.2fK-%.1fatm" % phase_point_key): ("%f" % value) for phase_point_key, value in
                              property_data.items()}

            printcool_dictionary(dict_for_print, title="Reference %s data" % property_name)

        # Assign and normalize weights for each phase point (average for now)
        self._normalised_weights = {}

        for property_name in self._reference_properties:

            self._normalised_weights[property_name] = (self._options.weights[property_name] /
                                                       len(self._reference_properties[property_name]))

    def _build_parameter_gradient_keys(self):
        """Build the list of parameter gradient keys based
        on changes to the force field.

        Returns
        -------
        list of propertyestimator.properties.ParameterGradientKey
        """
        # TODO for Simon: Build the gradient keys. `self.FF.plist` may be of use?
        raise NotImplementedError()

    def _submit_request(self, mvals, reweight_only=False):
        """Submit a property estimator request to the property estimator,
        and request.

        Parameters
        ----------
        mvals: np.ndarray
            mvals array containing the math values of the parameters
        reweight_only: bool
            If true, the estimator will only attempt to estimate the
            properties using data reweighting.
        """

        self.FF.make(mvals)
        force_field = smirnoff.ForceField(self.FF.offxml)

        options = PropertyEstimatorOptions()

        if reweight_only:
            options.allowed_calculation_layers = ['ReweightingLayer']

        return self._client.request_estimate(property_set=self._data_set,
                                             force_field=force_field,
                                             options=options)

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
        # The below snipped will both estimate the properties, and their
        # gradients with respect to a set of chosen parameters.

        # # Make the force field based on the current values of the parameters.
        # self.FF.make(mvals)
        #
        # force_field = smirnoff.ForceField(self.FF.offxml)
        # parameter_gradient_keys = []
        #
        # if AGrad is True:
        #     parameter_gradient_keys = self._build_parameter_gradient_keys()
        #
        # self._pending_estimate_request = self._client.request_estimate(property_set=self._data_set,
        #                                                                force_field=force_field,
        #                                                                parameter_gradient_keys=
        #                                                                parameter_gradient_keys)

        # Submit the reference property estimation request.
        self._pending_estimate_request = self._submit_request(mvals)
        self._pending_gradient_requests = {}

        if AGrad is True:

            for parameter_index in range(len(mvals)):

                self._pending_gradient_requests[parameter_index] = {'reverse': None, 'forward': None}

                for direction, delta_h in zip(['reverse', 'forward'], [-self.liquid_fdiff_h, self.liquid_fdiff_h]):
                    # copy the original mvals and perturb
                    new_mvals = mvals.copy()
                    new_mvals[parameter_index] += delta_h

                    # Submit the request with the parameter perturbation
                    self._pending_gradient_requests[parameter_index][direction] = \
                        self._submit_request(new_mvals, reweight_only=True)

            logger.info(f'{len(mvals) * 2} jobs (each with {self._data_set.number_of_properties} properties) with +- '
                        f'perturbations and employing only reweighting were submitted to the property estimator\n')

    @staticmethod
    def _is_request_finished(estimation_request):
        """Checks whether a property estimation request has
        completed.

        Parameters
        ----------
        estimation_request: PropertyEstimatorClient.Request
            The request to check.

        Returns
        -------
        bool
            True if all properties have been attempted to be
            estimated.
        """
        results = estimation_request.results()
        return isinstance(results, PropertyEstimatorException) or len(results.queued_properties) == 0

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

        # Check for any exceptions that were raised while estimating
        # the properties.
        if isinstance(results, PropertyEstimatorException):
            raise ValueError(f'An uncaught exception occured within the property '
                             f'estimator (directory={results.directory}: {results.message}')

        if len(results.unsuccessful_properties) > 0:

            exceptions = '\n'.join(f'{result.directory}: {result.message}'
                                   for result in results.exceptions)

            # TODO: How should we handle this case.
            raise ValueError(f'Some properties could not be estimated:\n\n{exceptions}.')

        elif len(results.exceptions) > 0:

            exceptions = '\n'.join(f'{result.directory}: {result.message}'
                                   for result in results.exceptions)

            # In some cases, an exception will be raised when executing a property but
            # it will not stop the property from being estimated (e.g an error occured
            # while reweighting so a simulation was used to estimate the property instead).
            logger.warning(f'A number of non-fatal exceptions occured:\n\n{exceptions}')

    @staticmethod
    def _extract_property_data(estimation_request):
        """Extract the property estimates #and their gradients#
        from a relevant property estimator request object.

        Parameters
        ----------
        estimation_request: PropertyEstimatorClient.Request
            The request to extract the data from.

        Returns
        -------
        dict of str, Any
            The estimated properties in a force balance dictionary.
        # dict of str, Any, optional
        #     The estimated gradients in a force balance dictionary.
        """

        # TODO: A given property may have been estimated for a number of different substances.
        #  How should these be consumed by force balance?
        estimated_property_data = {'values': {}, 'value_errors': {}}
        # estimated_gradients = {}

        # Make sure the request actually finished and was error free.
        if not PropertyEstimate_SMIRNOFF._is_request_finished(estimation_request):
            raise ValueError('Trying to extract the results of an unfinished request.')

        PropertyEstimate_SMIRNOFF._check_estimation_request(estimation_request)

        # Extract the results from the request.
        results = estimation_request.results()

        for substance_id in results.estimated_properties:

            for physical_property in results.estimated_properties[substance_id]:

                # Convert the property estimator properties list into
                # a force balance dictionary.
                class_name = physical_property.__class__.__name__
                mapped_name = PropertyEstimate_SMIRNOFF.estimator_to_force_balance_map[class_name]

                if mapped_name not in estimated_property_data['values']:
                    estimated_property_data['values'][mapped_name] = {}
                    estimated_property_data['value_errors'][mapped_name] = {}

                key, value, uncertainty = PropertyEstimate_SMIRNOFF._property_to_key_value(physical_property)
                estimated_property_data['values'][mapped_name][key] = value
                estimated_property_data['value_errors'][mapped_name][key] = uncertainty

                # The below snippet will extract any property estimator calculated
                # gradients.
                #
                # if AGrad is False:
                #     continue
                #
                # # Pull out any estimated gradients.
                # if mapped_name not in estimated_gradients:
                #     estimated_gradients[mapped_name] = {}
                #
                # if key not in estimated_gradients[mapped_name]:
                #     estimated_gradients[mapped_name][key] = {}
                #
                # for gradient in physical_property.gradients:
                #     parameter_index = self._parameter_key_to_index(gradient.key)
                #     gradient_value = gradient.value
                #
                #     estimated_gradients[mapped_name][key][parameter_index] = gradient_value

        return estimated_property_data  # , estimated_gradients

    def wq_complete(self):
        """
        Check if all jobs are finished
        This function should have a sleep in it if not finished.

        Returns
        -------
        finished: bool
            True if all jobs are finished, False if not
        """

        all_finished = self._is_request_finished(self._pending_estimate_request)

        for parameter_index in self._pending_gradient_requests:

            for direction in self._pending_gradient_requests[parameter_index]:

                request = self._pending_gradient_requests[parameter_index][direction]
                all_finished = all_finished & self._is_request_finished(request)

        return all_finished

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

        # Extract the properties estimated using the unperturbed parameters.
        estimated_property_data = self._extract_property_data(self._pending_estimate_request)

        assert len(self._pending_gradient_requests) == self.FF.np, 'number of submitted jobs not consistent'

        zero_gradient = np.zeros(self.FF.np)
        estimated_gradients = {}

        for property_name in self._reference_properties:

            estimated_gradients[property_name] = {}

            for phase_point in self._reference_properties[property_name]:
                estimated_gradients[property_name][phase_point] = zero_gradient

        for parameter_index in range(len(mvals)):

            results_reverse = self._extract_property_data(self._pending_gradient_requests[parameter_index]['reverse'])
            results_forward = self._extract_property_data(self._pending_gradient_requests[parameter_index]['forward'])

            for property_name in estimated_gradients:

                for phase_point in estimated_gradients[property_name]:

                    value_plus = results_forward[property_name][phase_point]
                    value_minus = results_reverse[property_name][phase_point]

                    # three point formula
                    gradient = (value_plus - value_minus) / (self.liquid_fdiff_h * 2)
                    estimated_gradients[property_name][phase_point][parameter_index] = gradient

        # compute objective value
        obj_value = 0.0
        obj_grad = np.zeros(self.FF.np)
        obj_hess = np.zeros((self.FF.np, self.FF.np))

        # store details for printing
        self._last_obj_details = {}

        for property_name in self._reference_properties:

            self.last_obj_details[property_name] = []

            denom = self._options.denominators[property_name]
            weight = self._normalised_weights[property_name]

            for phase_point_key in self.ref_data[property_name]:

                temperature, pressure = phase_point_key
                ref_value = self.ref_data[property_name][phase_point_key]

                # TODO: load computed value and standard error from real data - is this done now?
                tar_value = estimated_property_data['values'][property_name][(temperature, pressure)]
                tar_error = estimated_property_data['value_errors'][property_name][(temperature, pressure)]

                diff = tar_value - ref_value

                obj_contrib = weight * (diff / denom) ** 2
                obj_value += obj_contrib

                self.last_obj_details[property_name].append(
                    (temperature, pressure, ref_value, tar_value, tar_error, diff, weight, denom, obj_contrib))

                # compute objective gradient
                if AGrad is True:
                    # TODO `param_names` not used?
                    # param_names = self.FF.plist
                    # get gradients in physical unit
                    grad_array = estimated_gradients[property_name][phase_point_key]
                    # compute objective gradient
                    # TODO: `this_obj_grad` is not used?
                    # this_obj_grad = 2.0 * weight * diff * grad_array / denom ** 2
                    if AHess is True:
                        obj_hess += 2.0 * weight * (np.outer(grad_array, grad_array)) / denom ** 2

        return {'X': obj_value, 'G': obj_grad, 'H': obj_hess}

    def indicate(self):
        """
        print information into the output file about the last objective function evaluated
        This function should be called after get()
        """
        for property_name, details in self._last_obj_details.items():
            dict_for_print = {
                "  %9.2fK %7.1fatm" % detail[:2]: "%9.3f %14.3f +- %-7.3f % 7.3f % 9.5f % 9.5f % 9.5f " % detail[2:] for
                detail in details}
            title = '%s %s\nTemperature  Pressure  Reference  Calculated +- Stdev     Delta    Weight    ' \
                    'Denom     Term  ' % (self.name, property_name)
            printcool_dictionary(dict_for_print, title=title, bold=True, color=4, keywidth=15)
