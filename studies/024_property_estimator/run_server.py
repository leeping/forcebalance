#!/usr/bin/env python3
import argparse
import logging
import shutil
from os import path

from propertyestimator.backends import QueueWorkerResources, ComputeResources, DaskLocalCluster
from propertyestimator.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils import setup_timestamp_logging


def main(n_workers, cpus_per_worker, gpus_per_worker):

    if n_workers <= 0:
        raise ValueError('The number of workers must be greater than 0')
    if cpus_per_worker <= 0:
        raise ValueError('The number of CPU\'s per worker must be greater than 0')
    if gpus_per_worker < 0:
        raise ValueError('The number of GPU\'s per worker must be greater than or equal to 0')
    if 0 < gpus_per_worker != cpus_per_worker:
        raise ValueError('The number of GPU\'s per worker must match the number of '
                         'CPU\'s per worker.')

    # Set up logging for the propertyestimator.
    setup_timestamp_logging()
    logger = logging.getLogger()

    # Ask for an estimation server with access to 2 GPUs.
    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    # Set up a backend to run the calculations on. This assume running
    # on a HPC resources with the LSF queue system installed.
    if gpus_per_worker <= 0:
        worker_resources = ComputeResources(number_of_threads=cpus_per_worker)
    else:
        worker_resources = ComputeResources(number_of_threads=cpus_per_worker,
                                            number_of_gpus=gpus_per_worker,
                                            preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA)

    calculation_backend = DaskLocalCluster(number_of_workers=n_workers,
                                           resources_per_worker=worker_resources)

    # Create a backend to cache simulation files.
    storage_backend = LocalFileStorage(storage_directory)

    logger.info(f'Starting the server with {n_workers} workers, each with '
                f'{cpus_per_worker} CPUs and {gpus_per_worker} GPUs.')

    # Create an estimation server which will run the calculations.
    server = PropertyEstimatorServer(calculation_backend=calculation_backend,
                                     storage_backend=storage_backend,
                                     working_directory=working_directory,
                                     port=8000)

    # Tell the server to start listening for estimation requests.
    server.start_listening_loop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start a PropertyEstimatorServer with a '
                                                 'specified number of workers, each with '
                                                 'access to the specified compute resources.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--workers', '-nwork',
                        type=int,
                        help='The number of compute workers to spawn.',
                        required=False,
                        default=1)

    parser.add_argument('--cpus_per_worker', '-ncpus',
                        type=int,
                        help='The number CPUs each worker should have acces to. '
                             'The server will consume a total of `nwork * ncpus` CPU\'s.',
                        required=False,
                        default=1)

    parser.add_argument('--gpus_per_worker', '-ngpus',
                        type=int,
                        help='The number CPUs each worker should have acces to. '
                             'The server will consume a total of `nwork * ngpus` GPU\'s.',
                        required=False,
                        default=1)

    args = parser.parse_args()

    main(args.workers, args.cpus_per_worker, args.gpus_per_worker)
