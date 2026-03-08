#!/usr/bin/env python3
import multiprocessing

# Use "spawn" instead of the Linux default "fork". When forking from the
# multi-threaded Dask/Tornado process, child processes inherit locked mutexes
# from sibling threads and deadlock immediately. "spawn" starts a fresh
# interpreter with no inherited thread state, avoiding the deadlock entirely.
# This is the primary fix for Python 3.9 / older openff-evaluator versions.
# LW 2026-03-07: this fix was suggested by Claude.
multiprocessing.set_start_method("spawn", force=True)

import shutil
from os import path

# Setting this attribute bypasses multiprocessing.Manager() +
# multiprocessing.Process() in _Multiprocessor.run() (backends/dask.py).
# Without it, forking from the multi-threaded Dask/Tornado process
# deadlocks on Python 3.9/3.10 due to fork-safety issues (locked mutexes
# inherited by the child process). With it, tasks run directly in the
# Dask worker thread instead of being dispatched to a subprocess.
# LW 2026-03-07: backend/dask.py is from OpenFF Evaluator
import openff.evaluator
openff.evaluator._called_from_test = True

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
