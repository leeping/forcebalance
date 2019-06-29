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

logger = getLogger(__name__)

try:
    import propertyestimator
    from propertyestimator.client import PropertyEstimatorClient, ConnectionOptions, PropertyEstimatorOptions
    from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
    from propertyestimator.utils.exceptions import PropertyEstimatorException
    from propertyestimator.utils.serialization import TypedJSONDecoder
    from propertyestimator.workflow import WorkflowOptions
except ImportError:
    warn_once("Failed to import the propertyestimator package.")

try:
    from openforcefield.typing.engines import smirnoff
except ImportError:
    warn_once("Failed to import the openforcefield package.")

try:
    from simtk import unit
except ImportError:
    warn_once("Failed to import the openmm package.")


class PropertyEstimate_SMIRNOFF(Target):
    """A custom optimisation target which employs the propertyestimator
    package to rapidly estimate a collection of condensed phase physical
    properties at each optimisation epoch."""

    class OptionsFile:
        """Represents the set of options that a `PropertyEstimate_SMIRNOFF`
        target will be run with.

        Attributes
        ----------
        client_config: propertyestimator.client.ConnectionOptions
            The options for how the `propertyestimator` client should
            connect to a running server instance.
        sources: list of OptionsFile.PropertySource
            A list of sources to the properties which this target aims
            to fit against.
        weights: list of PropertyWeight
            The weighting of each property which will be optimised against.
        denominators: list of PropertyDenominator
            ???
        """

        class PropertySource:
            """Represents the source (either a path or a doi) of
            a ThermoML property.
            """

            @property
            def thermoml_doi(self):
                """str: The doi of a ThermoML property .xml file."""
                return self._thermoml_doi

            @property
            def path(self):
                """str: A path to a either a JSON property file or a
                ThermoML .xml archive file."""
                return self._path

            def __init__(self, path=None, thermoml_doi=None):
                """Constructs a new PropertySource object.

                Parameters
                ----------
                path: str, optional
                    A path to the ThermoML property .xml file.
                thermoml_doi: str, optional
                    The doi of a ThermoML property .xml file.
                """

                assert (path is None and thermoml_doi is not None or
                        path is not None and thermoml_doi is None)

                self._path = path
                self._thermoml_doi = thermoml_doi

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
                    return {'thermoml_doi': self.thermoml_doi}

            @classmethod
            def from_dict(cls, dictionary):
                """Creates an instance of this class from a dictionary.

                Parameters
                ----------
                dict of str and str
                    The dictionary representation to build the class from.
                """
                return cls(dictionary.get('path'), dictionary.get('thermoml_doi'))

        def __init__(self):

            self.connection_options = ConnectionOptions()
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
                'connection_options': self.connection_options.__getstate__(),
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
        def from_json(cls, json_source):
            """Creates this class from a JSON string.

            Parameters
            -------
            json_source: str or file-like object
                The JSON representation of this class.
            """

            if isinstance(json_source, str):
                with open(json_source, 'r') as file:
                    dictionary = json.load(file)
            else:
                dictionary = json.load(json_source)

            assert ('connection_options' in dictionary and
                    'sources' in dictionary and
                    'weights' in dictionary and
                    'denominators' in dictionary)

            value = cls()

            value.connection_options = ConnectionOptions()
            value.connection_options.__setstate__(dictionary['connection_options'])

            value.sources = [cls.PropertySource.from_dict(source) for source in dictionary['sources']]

            value.weights = {
                property_name: dictionary['weights'][property_name] for property_name in dictionary['weights']
            }
            value.denominators = {
                property_name: dictionary['denominators'][property_name] for property_name in dictionary['denominators']
            }

            return value

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
    def _refactor_properties_dictionary(properties):
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
        dict of str and dict of str and dict of tuple and dict of str and float
            The refactored properties dictionary.
        """

        refactored_properties = {}

        for substance_id in properties:

            for physical_property in properties[substance_id]:

                class_name = physical_property.__class__.__name__

                temperature = physical_property.thermodynamic_state.temperature.value_in_unit(unit.kelvin)
                pressure = physical_property.thermodynamic_state.pressure.value_in_unit(unit.atmosphere)

                state_tuple = (f'{temperature:.6f}', f'{pressure:.6f}')

                default_unit = PropertyEstimate_SMIRNOFF.default_units[class_name]

                value = physical_property.value.value_in_unit(default_unit)
                uncertainty = physical_property.uncertainty.value_in_unit(default_unit)

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
        print(f'{os.path.join(self.tgtdir, self.prop_est_input)}')
        options_file_path = os.path.join(self.tgtdir, self.prop_est_input)
        self._options = self.OptionsFile.from_json(options_file_path)

        # Attempt to create a property estimator client object using the specified
        # connection options.
        self._client = PropertyEstimatorClient(connection_options=self._options.connection_options)

        # Load in the experimental data set from either the specified doi or
        # local file paths.
        self._data_set = PhysicalPropertyDataSet()

        self._data_set.merge(ThermoMLDataSet.from_doi(*[source.thermoml_doi for source in
                                                        self._options.sources if source.thermoml_doi]))

        self._data_set.merge(ThermoMLDataSet.from_file(*[
            os.path.join(self.tgtdir, source.path) for
            source in self._options.sources if source.path and os.path.splitext(source.path)[1] == '.xml'
        ]))

        for source in self._options.sources:

            print(source.path, os.path.splitext(source.path)[1].lower())

            if not source.path or os.path.splitext(source.path)[1].lower() != '.json':
                continue

            with open(os.path.join(self.tgtdir, source.path), 'r') as file:
                physical_property = json.load(file, cls=TypedJSONDecoder)

                if physical_property.substance.identifier not in self._data_set.properties:
                    self._data_set.properties[physical_property.substance.identifier] = []

                self._data_set.properties[physical_property.substance.identifier].append(physical_property)

        if len(self._data_set.properties) == 0:
            raise ValueError('The physical property data set to optimise against is empty. '
                             'Either no physical properties were specified, or those that '
                             'were are unsupported by the estimator.')

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

    def _build_parameter_gradient_keys(self):
        """Build the list of parameter gradient keys based
        on changes to the force field.

        Returns
        -------
        list of propertyestimator.properties.ParameterGradientKey
        """
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
        force_field = smirnoff.ForceField(self.FF.offxml, allow_cosmetic_attributes=True)

        options = PropertyEstimatorOptions()

        density_options = WorkflowOptions(convergence_mode=WorkflowOptions.ConvergenceMode.AbsoluteUncertainty,
                                          absolute_uncertainty=0.0005 * unit.grams / unit.milliliter)
        dielectric_options = WorkflowOptions(WorkflowOptions.ConvergenceMode.NoChecks)

        options.workflow_options = {
            'Density': density_options,
            'DielectricConstant': dielectric_options
        }

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
        self._pending_estimate_request.results(synchronous=True, polling_interval=5)

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
        dict of str and dict of str and dict of tuple and dict of str and float
            The estimated properties in a dictionary of the form
            `property = dict[property_type][substance_id][state_tuple]['value' or 'uncertainty']`
        # dict of str, Any, optional
        #     The estimated gradients in a force balance dictionary.
        """

        # Make sure the request actually finished and was error free.
        if not PropertyEstimate_SMIRNOFF._is_request_finished(estimation_request):
            raise ValueError('Trying to extract the results of an unfinished request.')

        PropertyEstimate_SMIRNOFF._check_estimation_request(estimation_request)

        # Extract the results from the request.
        results = estimation_request.results()

        estimated_data = PropertyEstimate_SMIRNOFF._refactor_properties_dictionary(results.estimated_properties)
        # estimated_gradients = {}

        # The below snippet will extract any property estimator calculated
        # gradients.
        # for substance_id in results.estimated_properties:
        #
        #     for physical_property in results.estimated_properties[substance_id]:
        #
        #         # Convert the property estimator properties list into
        #         # a force balance dictionary.
        #         class_name = physical_property.__class__.__name__
        #
        #         if AGrad is False:
        #             continue
        #
        #         # Pull out any estimated gradients.
        #         if class_name not in estimated_gradients:
        #             estimated_gradients[class_name] = {}
        #
        #         if substance_id not in estimated_gradients[class_name]:
        #             estimated_gradients[class_name][substance_id] = {}
        #
        #         if state_key not in estimated_gradients[class_name][substance_id]:
        #             estimated_gradients[class_name][substance_id][state_key] = {}
        #
        #         for gradient in physical_property.gradients:
        #             parameter_index = self._parameter_key_to_index(gradient.key)
        #             gradient_value = gradient.value
        #
        #             estimated_gradients[class_name][substance_id][state_key][parameter_index] = gradient_value

        return estimated_data  # , estimated_gradients

    def _extract_property_gradients(self, mvals):
        """Extract gradients of all of the observables with respect to
        the force field parameters.

        Returns
        -------
        dict of str and dict of str and dict of tuple and float
            The estimated gradients in a dictionary of the form
            `gradient = dict[property_type][substance_id][state_tuple]`
        """

        zero_gradient = np.zeros(self.FF.np)
        estimated_gradients = {}

        for property_name in self._reference_properties:

            estimated_gradients[property_name] = {}

            for substance_id in self._reference_properties[property_name]:

                estimated_gradients[substance_id] = {}

                for phase_point in self._reference_properties[property_name][substance_id]:
                    estimated_gradients[property_name][substance_id][phase_point] = zero_gradient

        for parameter_index in range(len(mvals)):

            results_reverse = self._extract_property_data(self._pending_gradient_requests[parameter_index]['reverse'])
            results_forward = self._extract_property_data(self._pending_gradient_requests[parameter_index]['forward'])

            for property_name in estimated_gradients:

                for substance_id in self._reference_properties[property_name]:

                    for phase_point in estimated_gradients[property_name][substance_id]:

                        value_plus = results_forward[property_name][substance_id][phase_point]
                        value_minus = results_reverse[property_name][substance_id][phase_point]

                        # three point formula
                        gradient = (value_plus - value_minus) / (self.liquid_fdiff_h * 2)
                        estimated_gradients[property_name][substance_id][phase_point][parameter_index] = gradient

        return estimated_gradients

    def wq_complete(self):
        """
        Check if all jobs are finished
        This function should have a sleep in it if not finished.

        Returns
        -------
        finished: bool
            True if all jobs are finished, False if not
        """

        if not self._is_request_finished(self._pending_estimate_request):
            return False

        for parameter_index in self._pending_gradient_requests:

            for direction in self._pending_gradient_requests[parameter_index]:

                request = self._pending_gradient_requests[parameter_index][direction]

                if self._is_request_finished(request):
                    continue

                return False

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
        2. obj_hess is all zero when AHess == False or AGrad == False, because the hessian estimate
        depends on gradients
        """

        # Extract the properties estimated using the unperturbed parameters.
        estimated_property_data = self._extract_property_data(self._pending_estimate_request)
        estimated_gradients = {}

        assert len(self._pending_gradient_requests) == self.FF.np, 'number of submitted jobs not consistent'

        if AGrad:
            # Optionally extract and gradients.
            estimated_gradients = self._extract_property_gradients(mvals)

        # compute objective value
        obj_value = 0.0
        obj_grad = np.zeros(self.FF.np)
        obj_hess = np.zeros((self.FF.np, self.FF.np))

        # store details for printing
        self._last_obj_details = {}

        for property_name in self._reference_properties:

            self._last_obj_details[property_name] = []

            denom = self._options.denominators[property_name]
            weight = self._normalised_weights[property_name]

            for substance_id in self._reference_properties[property_name]:

                for phase_point_key in self._reference_properties[property_name][substance_id]:

                    temperature, pressure = phase_point_key
                    ref_value = self._reference_properties[property_name][substance_id][phase_point_key]['value']

                    tar_value = estimated_property_data[property_name][substance_id][phase_point_key]['value']
                    tar_error = estimated_property_data[property_name][substance_id][phase_point_key]['uncertainty']

                    diff = tar_value - ref_value

                    obj_contrib = weight * (diff / denom) ** 2
                    obj_value += obj_contrib

                    self._last_obj_details[property_name].append(
                        (temperature, pressure, ref_value, tar_value, tar_error, diff, weight, denom, obj_contrib))

                    # compute objective gradient
                    if AGrad is True:
                        # get gradients in physical unit
                        grad_array = estimated_gradients[property_name][substance_id][phase_point_key]
                        # compute objective gradient
                        obj_grad += 2.0 * weight * diff * grad_array / denom ** 2
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
