#!/usr/bin/env python3
import shutil
from os import path

from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.utils import setup_timestamp_logging


def main():

    # Set up logging for the evaluator.
    setup_timestamp_logging()

    # Set up the directory structure.
    working_directory = "working_directory"

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    # Set up a backend to run the calculations on with the requested resources.
    worker_resources = ComputeResources(number_of_threads=1)

    calculation_backend = DaskLocalCluster(
        number_of_workers=1, resources_per_worker=worker_resources
    )

    with calculation_backend:

        server = EvaluatorServer(
            calculation_backend=calculation_backend,
            working_directory=working_directory,
            port=8000,
        )

        # Tell the server to start listening for estimation requests.
        server.start()


if __name__ == "__main__":
    main()
