import forcebalance
import forcebalance.objective
import forcebalance.nifty
import tarfile
import os

# load pickled variables from forcebalance.p
f=open('forcebalance.p', 'r')
mvals, AGrad, AHess, options, tgt_opts, forcefield = forcebalance.nifty.lp_load(f)
f.close()

options['root'] = os.getcwd()

# set up forcefield
forcefield.make(mvals)

# set up and evaluate target
tar = tarfile.open("target.tar.bz2", "r")
os.mkdir("targets")
tar.extractall(path="targets")
tar.close()

Tgt = forcebalance.objective.Implemented_Targets[tgt_opts['type']](options,tgt_opts,forcefield)
Tgt.stage(mvals, AGrad = True, AHess = True)
Ans = Tgt.get(mvals, AGrad=True, AHess=True)

with open('objective.p', 'w') as f:
    forcebalance.nifty.lp_dump(f,Ans)        # or some other method of storing resulting objective

# also run target.indicate()
logger = forcebalance.output.getLogger("forcebalance")
logger.addHandler(forcebalance.nifty.RawFileHandler("indicate.log"))
Tgt.indicate()
