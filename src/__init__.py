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

import logging
import parser, forcefield, optimizer, objective, nifty

# set up package level logger that by default writes to stdout
logging.logger=logging.getLogger('forcebalance')
logging.logger.addHandler(nifty.RawFileHandler('test.log','w'))
logging.logger.setLevel(logging.INFO)

