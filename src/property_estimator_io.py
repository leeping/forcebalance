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
        # write a force field file with the current mvals
        self.FF.make(mvals)
        param_names = self.FF.plist
        ## TO-DO: Implement real submit jobs
        job_spec = {
            'offxml': self.FF.offxml, # offxml filename
            'ref_doi': self.prop_est_configure['source_doi'], # doi string
            'compute_gradients': param_names, # compute gradients for these params
            #'compute_hessian': param_names, # this can be skipped for now, FB will estimate hessian using gradient
        }
        #self.prop_est_job_id = self.client.submit_property_estimate(job_spec)
        ## mock job id
        self.prop_est_job_id = '82101023'

    def wq_complete(self):
        """
        Wait until all jobs complete.
        This function should block until all simulations are finished.
        This function is called before get()
        """
        while True:
            time.sleep(10)
            ## TO-DO: update function below for checking if job finished
            # status = self.client.check_job_status(self.prop_est_job_id)
            status = 'COMPLETE'
            if status == 'COMPLETE':
                return
            elif status == 'ERROR':
                logger.error("property estimator job failed")
                raise RuntimeError("property estimator job failed")

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
        ## TO-DO: get real property data
        # prop_est_data = self.client.download_job_data(self.prop_est_job_id)
        ## mock data
        zero_grad = np.zeros(self.FF.np)
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
            },
            'gradients': {
                'density': {
                    (298.15, 1.0): zero_grad,
                    (328.15, 1.0): zero_grad,
                },
                'dielectric': {
                    (298.15, 1.0): zero_grad,
                    (328.15, 1.0): zero_grad,
                },
                'Hvap': {
                    (298.15, 1.0): zero_grad,
                    (328.15, 1.0): zero_grad,
                }
            }
        }
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
                    ## TO-DO: load gradients from real data
                    grad_dict = prop_est_data['gradients'][property_name][(temperature, pressure)]
                    grad_array = np.array([grad_dict[param_name] for param_name in param_names])
                    # convert from d_P/d_pval to d_P/d_mval
                    grad_array = self.FF.create_pvals(grad)
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