import forcebalance
import forcebalance.objective
import forcebalance.nifty
import tarfile
import os
import forcebalance.output
logger = forcebalance.output.getLogger("forcebalance")
logger.setLevel(forcebalance.output.DEBUG)

#os.system('rm -rf ../cache')

# load pickled variables from forcebalance.p
f=open('forcebalance.p', 'r')
mvals, AGrad, AHess, n, options, tgt_opts, forcefield = forcebalance.nifty.lp_load(f)
f.close()

options['root'] = os.getcwd()

# set up forcefield
forcefield.make(mvals)

# set up and evaluate target
tar = tarfile.open("target.tar.bz2", "r")
tar.extractall()
tar.close()

Tgt = forcebalance.objective.Implemented_Targets[tgt_opts['type']](options,tgt_opts,forcefield)
Tgt.submit_jobs(mvals, AGrad = True, AHess = True)

Ans = Tgt.sget(mvals, AGrad=True, AHess=True)

with open('%s_%i_objective.p' % (Tgt.name, n), 'w') as f:
    forcebalance.nifty.lp_dump(Ans, f)        # or some other method of storing resulting objective

# also run target.indicate()
logger = forcebalance.output.getLogger("forcebalance")
logger.addHandler(forcebalance.output.RawFileHandler('%s_%i_indicate.log' % (Tgt.name, n)))
Tgt.indicate()

print "\n"
    
