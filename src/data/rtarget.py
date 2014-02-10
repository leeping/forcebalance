import forcebalance
import forcebalance.objective
import forcebalance.nifty
from forcebalance.nifty import wopen
import tarfile
import os
import forcebalance.output
logger = forcebalance.output.getLogger("forcebalance")
logger.setLevel(forcebalance.output.DEBUG)

#os.system('rm -rf ../cache')

# load pickled variables from forcebalance.p
f=open('forcebalance.p', 'r')
mvals, AGrad, AHess, id_string, options, tgt_opts, forcefield = forcebalance.nifty.lp_load(f)
f.close()

options['root'] = os.getcwd()
forcefield.root = os.getcwd()

# set up forcefield
forcefield.make(mvals, printdir="forcefield")

# set up and evaluate target
tar = tarfile.open("%s.tar.bz2" % tgt_opts["name"], "r")
tar.extractall()
tar.close()

Tgt = forcebalance.objective.Implemented_Targets[tgt_opts['type']](options,tgt_opts,forcefield)
Tgt.submit_jobs(mvals, AGrad = True, AHess = True)

Ans = Tgt.sget(mvals, AGrad=True, AHess=True)

with wopen('objective_%s.p' % id_string) as f:
    forcebalance.nifty.lp_dump(Ans, f)        # or some other method of storing resulting objective

# also run target.indicate()
logger = forcebalance.output.getLogger("forcebalance")
logger.addHandler(forcebalance.output.RawFileHandler('indicate_%s.log' % id_string))
Tgt.indicate()

print "\n"
    
