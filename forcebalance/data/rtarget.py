from __future__ import print_function
import forcebalance
import forcebalance.objective
import forcebalance.nifty
from forcebalance.nifty import wopen
import tarfile
import os
import forcebalance.output
logger = forcebalance.output.getLogger("forcebalance")
logger.setLevel(forcebalance.output.DEBUG)

# load pickled variables from forcebalance.p
if os.path.exists('forcebalance.p'):
    mvals, AGrad, AHess, id_string, options, tgt_opts, forcefield, pgrad = forcebalance.nifty.lp_load('forcebalance.p')
else:
    forcefield, mvals = forcebalance.nifty.lp_load('forcefield.p')
    AGrad, AHess, id_string, options, tgt_opts, pgrad = forcebalance.nifty.lp_load('options.p')

print("Evaluating remote target ID: %s" % id_string)

options['root'] = os.getcwd()
forcefield.root = os.getcwd()

# set up forcefield
forcefield.make(mvals, printdir="forcefield")

# set up and evaluate target
tar = tarfile.open("target.tar.bz2", "r")
tar.extractall()
tar.close()

Tgt = forcebalance.objective.Implemented_Targets[tgt_opts['type']](options,tgt_opts,forcefield)
Tgt.read_objective = False
Tgt.read_indicate = False
Tgt.write_objective = False
Tgt.write_indicate = False

# The "active" parameters are determined by the master, written to disk and sent over.
Tgt.pgrad = pgrad

# Should the remote target be submitting jobs of its own??
Tgt.submit_jobs(mvals, AGrad = AGrad, AHess = AHess)

Ans = Tgt.meta_get(mvals, AGrad = AGrad, AHess = AHess)

# go to the tmp folder
os.chdir(os.path.join(Tgt.root,Tgt.tempdir))
# go to the 'iter_0000' folder
for f in os.listdir('.'):
    if os.path.isdir(f) and f.startswith('iter'):
        os.chdir(f)
        break

forcebalance.nifty.lp_dump(Ans, 'objective.p')        # or some other method of storing resulting objective

# also run target.indicate()
logger = forcebalance.output.getLogger("forcebalance")
logger.addHandler(forcebalance.output.RawFileHandler('indicate.log'))
Tgt.indicate()
print("\n")

# compress all files into target_result.tar.bz2

with tarfile.open(name=os.path.join(options['root'],"target_result.tar.bz2"), mode='w:bz2', dereference=True) as tar:
    for f in os.listdir('.'):
        if os.path.isfile(f):
            tar.add(f)


