import os, sys, glob, commands
import numpy as np
from scipy import loadtxt 
from matplotlib import pyplot


def get_mm_energies(topfile, name, Testing=False):
    # Calculate energies before:
    run_cmd('grompp -f singlepoint.mdp -c chi0.omega0.gro -p %s -o %s.tpr'%(topfile, name), Testing)
    run_cmd('mdrun -s %s.tpr -rerun all.gro'%name, Testing)
    run_cmd('echo "9\n\n" | g_energy -f ener.edr', Testing)
    run_cmd('rm ./#*')
    return get_energies_from_xvg('energy.xvg')/4.184 # convert from kJ to kcal
 
def run_cmd(cmd, Testing=False):
    print '>>>', cmd
    if not Testing:
        os.system(cmd)

def get_energies_from_xvg(xvgfile, normalize=True):
    """Returns a numpy array of the columns of data in an xvgfile."""
    fin = open(xvgfile,'r')
    datalist = [ [float(x) for x in s.strip().split()] for s in fin.readlines() if ((s[0] != '#') and (s[0] != '@')) ]
    data =  np.array(datalist)[:,1]
    if normalize:
        data = data - data.min()
    return data

def gridplot(X,Y,grid, title):
    """Make a matplotlib 2D plot."""
    X,Y = np.meshgrid(X,Y)
    pyplot.pcolor(X, Y, grid.transpose())
    #pyplot.imshow(X,Y, grid)
    pyplot.colorbar()
    pyplot.xlabel('chi (degrees)')
    pyplot.ylabel('omega (degrees)')
    pyplot.title(title)


def gridplot2(X, Y, Z):
    CS = pyplot.contour(X, Y, Z)
    pyplot.clabel(CS, inline=1, fontsize=10)
    pyplot.title('Simplest default with labels')


# Main

# get the chi, omega angles and QM energies for each all.gro frame
chis, omegas, qm_energies = [], [], []
lines = commands.getoutput('cat all.gro | grep Energy').split('\n')
for line in lines:
    qm_energies.append(float(line.split()[-1]))
    chis.append( int( [s for s in line.split() if s.count('min.')][0].split('.')[4] ))
    omegas.append(  int([s for s in line.split() if s.count('min.')][0].split('.')[6] ))

# Process qm_energies 
qm_energies = np.array(qm_energies)
qm_energies = qm_energies - qm_energies.min()  # normalize 
print 'len(qm_energies)', len(qm_energies)

# Process chi and omega
chis = (np.array(chis)+185)%360-185
omegas = (np.array(omegas)+185)%360-185

print 'chis', chis
print 'omegas', omegas

# calculate MM energies before the optimization
before_top = '../Setup/LIG_GMX.top'
before_mm_energies = get_mm_energies(before_top, 'before')
print 'before_mm_energies', before_mm_energies

# Calculate MM energies after the optimization
after_top = 'after.top'
after_mm_energies = get_mm_energies(after_top, 'after')
print 'after_mm_energies', after_mm_energies

# arrange all the energies in 2D grids
#bg = -1.
qm_grid = np.zeros( (24,24), np.float)
mm_before_grid = np.zeros( (24,24), np.float)
mm_after_grid = np.zeros( (24,24), np.float)

nbins = len(chis) 
X, Y = range(-180, 180, 15), range(-180, 180, 15)
print 'X, Y', X, Y, len(X)
print 'nbins', nbins
for k in range(nbins):
    print 'chis[k]', chis[k], 'omegas[k]', omegas[k]
    i = X.index( ((chis[k]+185)%360-185))
    j = Y.index( ((omegas[k]+185)%360-185))
    print 'i,j', i,j
    qm_grid[i,j] = qm_energies[k]
    mm_before_grid[i,j] = before_mm_energies[k]
    mm_after_grid[i,j] = after_mm_energies[k]

# plot the results
offset = 1.
pyplot.figure()
pyplot.subplot(2,2,1)
gridplot( X, Y, qm_grid, 'QM energies (kcal/mol)' )

pyplot.subplot(2,2,3)
pyplot.pcolor(mm_before_grid)
gridplot(X, Y, mm_before_grid, 'MM energies (kcal/mol) before parameterization' )

pyplot.subplot(2,2,4)
pyplot.pcolor(mm_after_grid)
gridplot(X,Y, mm_after_grid, 'MM energies (kcal/mol) after parameterization')

pyplot.show()
