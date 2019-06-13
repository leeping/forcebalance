""" @package forcebalance.property_estimator_io
For interfacing with property estimator to compute liquid properties.
Currently only support OpenForceField SMIRNOFF force field

author Yudong Qiu, Simon Boothroyd, Lee-Ping Wang
@date 06/2019
"""
from __future__ import division, print_function
import os
import time
import json
import copy
import numpy as np

from forcebalance.target import Target
from forcebalance.nifty import warn_once, printcool, printcool_dictionary
from forcebalance.output import getLogger
logger = getLogger(__name__)

try:
    import property_estimator
except ImportError:
    warn_once("Failed to import property_estimator package.")

class PropertyEstimate_SMIRNOFF(Target):
    """ Subclass of Target for property estimator interface """

    def __init__(self,options,tgt_opts,forcefield):
        # Initialize base class
        super(PropertyEstimate_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)
        # get filename for property estimator input file
        self.set_option(tgt_opts,'prop_est_input',forceprint=True)
        # Finite difference size for numerical gradients
        self.set_option(tgt_opts,'liquid_fdiff_h', forceprint=True)
        # initialize the properties to compute
        self.init_properties()

    def init_properties(self):
        """ read input configure, communicate to property estimator
        1. Read input file
        2. create client
        3. use client to get information about the experimental dataset
        """
        self.prop_est_configure = self.read_prop_est_input()
        # create a property estimator client
        ## TO-DO: use real client
        # self.client = property_estimator.Client.from_config(self.prop_est_configure.get('client_conf'))
        self.client = "replace this"
        # specify experimental data using doi
        ## TO-DO: get real reference data from property estimator, by specifing a list of doi's
        # prop_est_dataset = self.client.load_reference_dataset_from_doi(self.prop_est_configure['source_doi'])
        ## mock data
        prop_est_dataset = {
            'properties': {
                'density': {
                    (298.15, 1.0): 1.0,
                    (328.15, 1.0): 0.95,
                },
                'dielectric': {
                    (298.15, 1.0): 0.6,
                    (328.15, 1.0): 0.4,
                },
                'Hvap': {
                    (298.15, 1.0): 0.6,
                    (328.15, 1.0): 0.4,
                }
            }
        }
        # put reference data into FB format
        self.ref_data = {}
        for property_name, property_data in prop_est_dataset['properties'].items():
            self.ref_data[property_name] = {}
            for phase_point, value in property_data.items():
                temperature, pressure = phase_point
                # this self.ref_data[property_name][phase_point_key] structure should be kept
                phase_point_key = (temperature, pressure)
                self.ref_data[property_name][phase_point_key] = value
        # print the reference data
        printcool("Loaded experimental data from property estimator")
        for property_name, property_data in self.ref_data.items():
            dict_for_print = {("%.2fK-%.1fatm" % phase_point_key) : ("%f" % value) for phase_point_key, value in property_data.items()}
            printcool_dictionary(dict_for_print, title="Reference %s data" % property_name)
        # assign and normalize weights for each phase point (average for now)
        self.property_weights = {}
        for property_name in self.ref_data:
            self.property_weights[property_name] = self.prop_est_configure['weights'][property_name] / len(self.ref_data[property_name])

    def read_prop_est_input(self):
        """ Load property estimator input configurations
        Returns
        -------
        prop_est_configure: dict
        A dict contains the configuration of property estimation, includes client configure, and weights.

        Example
        -------
        >>> self.read_prop_est_input()
        {
            'client_conf': {
                'host': 'localhost',
                'port': '8301',
                'user': 'forcebalance'
            },
            'source_doi': ['10.123.23910/12891'],
            'weights': {
                'density': 1.0,
                'dielectric': 1.0,
                'Hvap': 1.0,
            },
            'denoms': {
                'density': 1.0, # kg/m3
                'dielectric': 2.0,
                'Hvap': 1.0,
            },
        }
        """
        ## TO-DO: update content of the configuration to meet the need for property estimater
        with open(os.path.join(self.tgtdir, self.prop_est_input)) as inputfile:
            prop_est_configure = json.load(inputfile)
        # validate the configure
        assert 'client_conf' in prop_est_configure, 'missing client configure'
        assert 'source_doi' in prop_est_configure, 'missing source_doi'
        assert 'weights' in prop_est_configure, 'missing weights'
        assert 'denoms' in prop_est_configure, 'missing denoms'
        return prop_est_configure

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        """
        submit jobs for evaluating the objective function

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
        # numerical gradients
        # create a structure for storing job ids for all perturbations
        self.prop_est_job_ids = {'0': None, '-h': [], '+h': []}
        # step 1: submit evalutation with zero perturbation
        self.prop_est_job_ids['0'] = self.submit_property_estimate(mvals)
        logger.info('One job submitted to property estimator\n')
        # step 2: submit evaluation with -h and +h perturbations
        if AGrad is True:
            for i_m in range(len(mvals)):
                for d_h in ['-h', '+h']:
                    delta_m = -self.liquid_fdiff_h if d_h == '-h' else +self.liquid_fdiff_h
                    # copy the original mvals and perturb
                    new_mvals = copy.deepcopy(mvals)
                    new_mvals[i_m] += delta_m
                    # submit the single job with perturbation
                    job_id = self.submit_property_estimate(new_mvals, reweight=True)
                    # store the job id
                    self.prop_est_job_ids[d_h].append(job_id)
            logger.info('%d jobs with +- perturbations and reweight submitted to property estimator\n' % (len(mvals)*2))
        # create a dictionary to store job data
        self.prop_est_job_state = {}
        for job_id in [self.prop_est_job_ids['0']] + self.prop_est_job_ids['-h'] + self.prop_est_job_ids['+h']:
            self.prop_est_job_state[job_id] = "submitted"

    def submit_property_estimate(self, mvals, reweight=False):
        """
        submit a single property estimate job

        Parameters
        ----------
        mvals: np.ndarray
            mvals array containing the math values of the parameters
        reweight: bool
            use reweight to or not

        Returns
        -------
        job_id: str
            The ID of the submitted job
        """
        # write a force field file with the current mvals
        # self.FF.offxml file will be overwritten using new parameters
        self.FF.make(mvals)
        ## TO-DO: implement real submit job
        job_spec = {
            'offxml': self.FF.offxml, # offxml filename
            'ref_doi': self.prop_est_configure['source_doi'], # doi string
            'reweight': reweight,
        }
        # job_id = self.client.submit_property_estimate(job_spec)
        ## mock
        job_id = '1293791'
        return job_id

    def wq_complete(self):
        """
        Check if all jobs are finished
        This function should have a sleep in it if not finished.

        Returns
        -------
        finished: bool
            True if all jobs are finished, False if not
        """
        # check status of the jobs that are not finished
        unfinished_job_ids = [job_id for job_id, status in self.prop_est_job_state.items() if status != 'COMPLETE']
        if len(unfinished_job_ids) == 0:
            return True
        else:
            for job_id in unfinished_job_ids:
                ## TO-DO: update function below for checking if job finished
                # status = self.client.check_job_status(job_id)
                # mock
                status = 'COMPLETE'
                self.prop_est_job_state[job_id] = status
            # check again if all jobs just finished
            if all(self.prop_est_job_state[job_id] == 'COMPLETE' for job_id in unfinished_job_ids):
                return True
            else:
                logger.info("%d/%d property estimate jobs finished\n" % (len(unfinished_job_ids), len(self.prop_est_job_state)))
                time.sleep(10)

    def download_property_estimate_data(self, job_id):
        """
        download property estimate data from server

        Parameters
        ----------
        job_id: string
            job_id received when submitting the job

        Returns
        -------
        property_data: dict
            dictionary of property data
        """
        ## TO-DO: implement real downlaod data from server
        # data = self.client.download_data(job_id)
        ## mock:
        prop_est_data = {
            'values': {
                'density': {
                    (298.15, 1.0): 1.03,
                    (328.15, 1.0): 0.95,
                },
                'dielectric': {
                    (298.15, 1.0): 0.64,
                    (328.15, 1.0): 0.44,
                },
                'Hvap': {
                    (298.15, 1.0): 0.61,
                    (328.15, 1.0): 0.43,
                }
            },
            'value_errors': {
                'density': {
                    (298.15, 1.0): 0.03,
                    (328.15, 1.0): 0.04,
                },
                'dielectric': {
                    (298.15, 1.0): 0.04,
                    (328.15, 1.0): 0.05,
                },
                'Hvap': {
                    (298.15, 1.0): 0.06,
                    (328.15, 1.0): 0.04,
                }
            }
        }
        return prop_est_data



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
        # downlaod property data for unperturbed simulation
        job_id_0 = self.prop_est_job_ids['0']
        prop_est_data = self.download_property_estimate_data(job_id_0)
        # download purturbed property values and do finite-difference calculations
        if AGrad is True:
            assert len(self.prop_est_job_ids['-h']) == self.FF.np, 'number of submitted jobs not consistent'
            zero_grad = np.zeros(self.FF.np)
            prop_est_gradients = {}
            for property_name in self.ref_data:
                prop_est_gradients[property_name] = {}
                for phase_point in self.ref_data[property_name]:
                    prop_est_gradients[property_name][phase_point] = zero_grad
            for i_m in range(len(mvals)):
                job_id_minus_h = self.prop_est_job_ids['-h'][i_m]
                prop_est_data_minus_h = self.download_property_estimate_data(job_id_minus_h)
                job_id_plus_h = self.prop_est_job_ids['+h'][i_m]
                prop_est_data_plus_h = self.download_property_estimate_data(job_id_plus_h)
                for property_name in prop_est_gradients:
                    for phase_point in prop_est_gradients[property_name]:
                        value_plus = prop_est_data_plus_h[property_name][phase_point]
                        value_minus = prop_est_data_minus_h[property_name][phase_point]
                        # three point formula
                        grad = (value_plus - value_minus) / (self.liquid_fdiff_h * 2)
                        prop_est_gradients[property_name][phase_point][i_m] = grad
        # compute objective value
        obj_value = 0.0
        obj_grad = np.zeros(self.FF.np)
        obj_hess = np.zeros((self.FF.np,self.FF.np))
        # store details for printing
        self.last_obj_details = {}
        for property_name in self.ref_data:
            self.last_obj_details[property_name] = []
            denom = self.prop_est_configure['denoms'][property_name]
            weight = self.property_weights[property_name]
            for phase_point_key in self.ref_data[property_name]:
                temperature, pressure = phase_point_key
                ref_value = self.ref_data[property_name][phase_point_key]
                ## TO-DO: load computed value and standard error from real data
                tar_value = prop_est_data['values'][property_name][(temperature, pressure)]
                tar_error = prop_est_data['value_errors'][property_name][(temperature, pressure)]
                diff = tar_value - ref_value
                obj_contrib = weight * (diff / denom)**2
                obj_value += obj_contrib
                self.last_obj_details[property_name].append((temperature, pressure, ref_value, tar_value, tar_error, diff, weight, denom, obj_contrib))
                # compute objective gradient
                if AGrad is True:
                    param_names = self.FF.plist
                    # get gradients in physical unit
                    grad_array = prop_est_gradients[property_name][phase_point_key]
                    # compute objective gradient
                    this_obj_grad = 2.0 * weight * diff * grad_array / denom**2
                    if AHess is True:
                        obj_hess += 2.0 * weight * (np.outer(grad_array, grad_array)) / denom**2
        return {'X':obj_value, 'G':obj_grad, 'H':obj_hess}

    def indicate(self):
        """
        print information into the output file about the last objective function evaluated
        This function should be called after get()
        """
        for property_name, details in self.last_obj_details.items():
            dict_for_print = {"  %9.2fK %7.1fatm" % detail[:2] : "%9.3f %14.3f +- %-7.3f % 7.3f % 9.5f % 9.5f % 9.5f " % detail[2:] for detail in details}
            title = '%s %s\nTemperature  Pressure  Reference  Calculated +- Stdev     Delta    Weight    Denom     Term  ' % (self.name, property_name)
            printcool_dictionary(dict_for_print, title=title, bold=True, color=4, keywidth=15)