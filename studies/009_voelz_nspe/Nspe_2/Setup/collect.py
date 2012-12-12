import os, sys, string, glob

xyzfiles = glob.glob('structures/xyz/*.xyz')
os.system('rm all.xyz')

for omega in [0,180]:
 for chi1 in range(0,361,15):
  for phi1 in range(0,361,30):

    xyzfile = 'structures/xyz/min.omega.%d.chi1.%d.phi1.%d.xyz'%(omega,chi1,phi1)
    if os.path.exists(xyzfile):
       os.system( 'cat %s >> all.xyz'%xyzfile )
    
