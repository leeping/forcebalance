#!/usr/bin/env python3
import json
import shutil
from enum import Enum
from os import path

from propertyestimator.backends import DaskLocalCluster, QueueWorkerResources, DaskLSFBackend, ComputeResources
from propertyestimator.properties import Density, PropertyPhase, MeasurementSource
from propertyestimator.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator import unit


class BackendType(Enum):

    LocalCPU = 'LocalCPU'
    LocalGPU = 'LocalGPU'
    GPU = 'GPU'
    CPU = 'CPU'


def setup_server(backend_type=BackendType.GPU, max_number_of_workers=1):

    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    calculation_backend = None

    if backend_type == BackendType.LocalCPU:

        calculation_backend = DaskLocalCluster(number_of_workers=1)

    if backend_type == BackendType.LocalGPU:

        compute_resources = ComputeResources(number_of_threads=1, number_of_gpus=1,
                                             preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA)

        calculation_backend = DaskLocalCluster(number_of_workers=1,
                                               resources_per_worker=compute_resources)

    elif backend_type == BackendType.GPU:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               number_of_gpus=1,
                                               preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                               per_thread_memory_limit=5 * unit.gigabyte,
                                               wallclock_time_limit="05:59")

        extra_script_options = [
            '-m "ls-gpu lt-gpu"'
        ]

        worker_script_commands = [
            'export OE_LICENSE="/home/boothros/oe_license.txt"',
            '. /home/boothros/miniconda3/etc/profile.d/conda.sh',
            'conda activate forcebalance',
            'module load cuda/9.2'
        ]

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=max_number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name='gpuqueue',
                                             setup_script_commands=worker_script_commands,
                                             extra_script_options=extra_script_options,
                                             adaptive_interval='1000ms')

    elif backend_type == BackendType.CPU:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               per_thread_memory_limit=5 * unit.gigabyte,
                                               wallclock_time_limit="01:30")

        worker_script_commands = [
            'export OE_LICENSE="/home/boothros/oe_license.txt"',
            '. /home/boothros/miniconda3/etc/profile.d/conda.sh',
            'conda activate forcebalance',
        ]

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=max_number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name='cpuqueue',
                                             setup_script_commands=worker_script_commands,
                                             adaptive_interval='1000ms')

    storage_backend = LocalFileStorage(storage_directory)

    server = PropertyEstimatorServer(calculation_backend=calculation_backend,
                                     storage_backend=storage_backend,
                                     working_directory=working_directory,
                                     port=8000)

    return server


def main():

    setup_timestamp_logging('server_logger_output.log')

    server = setup_server(backend_type=BackendType.GPU,
                          max_number_of_workers=2)

    server.start_listening_loop()


if __name__ == '__main__':
    main()
