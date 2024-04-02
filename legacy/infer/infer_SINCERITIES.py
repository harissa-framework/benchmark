# Script pour lancer tous les benchmarks de GENIE3
import sys; sys.path += ['./_scripts']
import time as timer
import numpy as np
from sincerities import sincerities

# Number of runs
N = 10

# Inference for Network4
for r in range(N):
    print('Run {} inference...'.format(r+1))
    fname = 'Network4/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    score = sincerities(x)
    np.save('Network4/SINCERITIES/score_{}'.format(r+1), score)


# Inference for Bifurcation
for r in range(N):
    print('Bifurcation - Run {} inference...'.format(r+1))
    fname = 'Bifurcation/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    score = sincerities(x)
    np.save('Bifurcation/SINCERITIES/score_{}'.format(r+1), score)
    
    
# Inference for Trifurcation
for r in range(N):
    print('Trifurcation - Run {} inference...'.format(r+1))
    fname = 'Trifurcation/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    score = sincerities(x)
    np.save('Trifurcation/SINCERITIES/score_{}'.format(r+1), score)
    
# Inference for Cycle
for r in range(N):
    print('Cycle - Run {} inference...'.format(r+1))
    fname = 'Cycle/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    score = sincerities(x)
    np.save('Cycle/SINCERITIES/score_{}'.format(r+1), score)

# Inference for tree-like networks
for n in [5,10,20,50,100]:
    runtime = np.zeros(N)
    for r in range(N):
        print('Trees{} - Run {} inference...'.format(n,r+1))
        fname = 'Trees{}/Data/data_{}.txt'.format(n,r+1)
        data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
        time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
        x = data.T
        x[:,0] = time
        t0 = timer.time()
        score = sincerities(x)
        t1 = timer.time()
        runtime[r] = t1 - t0
        np.save('Trees{}/SINCERITIES/score_{}'.format(n,r+1), score)
    # Save running times
    np.savetxt('Trees{}/SINCERITIES/runtime.txt'.format(n), runtime.T)
