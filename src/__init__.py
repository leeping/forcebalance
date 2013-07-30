try:
    __import__("numpy")
except ImportError:
    print "Could not load numpy module, exiting..."
    exit()

try:
    __import__("scipy")
except ImportError:
    print "Could not load scipy module, exiting..."
    exit()

import parser, forcefield, optimizer, objective, output
from collections import defaultdict

# Global variable corresponding to the Work Queue object
WORK_QUEUE = None

# Global variable containing a mapping from target names to Work Queue task IDs
WQIDS = defaultdict(list)

