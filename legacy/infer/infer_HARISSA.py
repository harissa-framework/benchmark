# Script pour lancer tous les benchmarks de HARISSA
import sys; sys.path += ['../']
import time as timer
import numpy as np
from Packages.harissa import NetworkModel

# Number of runs
N = 10

# Inference for Cycle
for r in range(N):
    print('Cycle - Run {} inference...'.format(r+1))
    fname = 'Cycle/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    model = NetworkModel()
    model.fit(x)
    score = model.inter
    np.save('Cycle/HARISSA/score_{}'.format(r+1), score)

# Inference for Network4
for r in range(N):
    print('Run {} inference...'.format(r+1))
    fname = 'Network4/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    model = NetworkModel()
    model.fit(x)
    score = model.inter
    np.save('Network4/HARISSA/score_{}'.format(r+1), score)


# Inference for Trifurcation
for r in range(N):
    print('Trifurcation - Run {} inference...'.format(r+1))
    fname = 'Trifurcation/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    model = NetworkModel()
    model.fit(x)
    score = model.inter
    np.save('Trifurcation/HARISSA/score_{}'.format(r+1), score)

# Inference for Bifurcation
for r in range(N):
    print('Bifurcation - Run {} inference...'.format(r+1))
    fname = 'Bifurcation/Data/data_{}.txt'.format(r+1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:,0] = time
    model = NetworkModel()
    model.fit(x)
    score = model.inter
    np.save('Bifurcation/HARISSA/score_{}'.format(r+1), score)

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
        model = NetworkModel()
        t0 = timer.time()
        model.fit(x)
        t1 = timer.time()
        runtime[r] = t1 - t0
        score = model.inter
        np.save('Trees{}/HARISSA/score_{}'.format(n,r+1), score)
    # Save running times
    np.savetxt('Trees{}/HARISSA/runtime.txt'.format(n), runtime.T)


