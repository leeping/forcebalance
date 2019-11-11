#!/usr/bin/env python3
import shutil
from os import path

from propertyestimator import unit
from propertyestimator.backends import QueueWorkerResources, DaskLSFBackend
from propertyestimator.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils import setup_timestamp_logging


def setup_server(max_number_of_workers=1):

    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    # Set up a backend to run the calculations on. This assume running
    # on a HPC resources with the LSF queue system installed.
    queue_resources = QueueWorkerResources(number_of_threads=1,
                                           number_of_gpus=1,
                                           preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                           per_thread_memory_limit=5 * unit.gigabyte,
                                           wallclock_time_limit="05:59")

    worker_script_commands = [
        'conda activate forcebalance',
        'module load cuda/10.1'
    ]

    calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                         maximum_number_of_workers=max_number_of_workers,
                                         resources_per_worker=queue_resources,
                                         queue_name='gpuqueue',
                                         setup_script_commands=worker_script_commands,
                                         adaptive_interval='1000ms')

    # Create a backend to cache simulation files.
    storage_backend = LocalFileStorage(storage_directory)

    # Create an estimation server which will run the calculations.
    server = PropertyEstimatorServer(calculation_backend=calculation_backend,
                                     storage_backend=storage_backend,
                                     working_directory=working_directory,
                                     port=8000)

    return server


def main():

    # Set up logging for the propertyestimator.
    setup_timestamp_logging('server_logger_output.log')

    # Ask for an estimation server with access to 2 GPUs.
    server = setup_server(max_number_of_workers=2)

    # Tell the server to start listening for estimation requests.
    server.start_listening_loop()


if __name__ == '__main__':
    main()
