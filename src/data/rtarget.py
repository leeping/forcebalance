import forcebalance
import forcebalance.objective
import forcebalance.nifty
import tarfile
import os
import forcebalance.output
logger = forcebalance.output.getLogger("forcebalance")
logger.setLevel(forcebalance.output.DEBUG)

os.system('rm -rf ../cache')

# load pickled variables from forcebalance.p
f=open('forcebalance.p', 'r')
mvals, AGrad, AHess, options, tgt_opts, forcefield = forcebalance.nifty.lp_load(f)
f.close()

options['root'] = os.getcwd()

# set up forcefield
tar = tarfile.open("forcefield.tar.bz2", "r")
tar.extractall()
tar.close()
forcefield.make(mvals)

# set up and evaluate target
tar = tarfile.open("target.tar.bz2", "r")
tar.extractall()
tar.close()

Tgt = forcebalance.objective.Implemented_Targets[tgt_opts['type']](options,tgt_opts,forcefield)
Tgt.submit_jobs(mvals, AGrad = True, AHess = True)

os.chdir("temp/cluster-12/")

Ans = Tgt.get(mvals, AGrad=True, AHess=True)

os.chdir("../..")

with open('objective.p', 'w') as f:
    forcebalance.nifty.lp_dump(Ans, f)        # or some other method of storing resulting objective

# also run target.indicate()
logger = forcebalance.output.getLogger("forcebalance")
logger.addHandler(forcebalance.output.RawFileHandler("indicate.log"))
Tgt.indicate()

os.chdir(Tgt.root)
os.system('rm -rf temp forcefield targets backups')

print "\n"
    
