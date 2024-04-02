# Script pour lancer tous les benchmarks de CARDAMOM
import sys; sys.path += ['../']
import time as timer
import numpy as np
from Packages.cardamom import NetworkModel

# Number of runs
N = 10
# Print information
verb = 0

# Inference for Network4
for r in range(0, N):
    print('Run {} inference...'.format(r+1))
    fname = 'Network4/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.fit(x, verb=verb)
    np.save('Network4/CARDAMOM/score_{}'.format(r+1), model.inter)
    np.save('Network4/CARDAMOM/basal_{}'.format(r+1), model.basal)
    np.save('Network4/CARDAMOM/kmin_{}'.format(r+1), np.min(model.data_bool, 0))
    np.save('Network4/CARDAMOM/kmax_{}'.format(r+1), np.max(model.data_bool, 0))
    np.save('Network4/CARDAMOM/bet_{}'.format(r+1), model.a[-1, :])

# Inference for Cycle
for r in range(0, N):
    print('Cycle - Run {} inference...'.format(r + 1))
    fname = 'Cycle/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.fit(x, verb=verb)
    np.save('Cycle/CARDAMOM/score_{}'.format(r + 1), model.inter)

# Inference for Bifurcation
for r in range(0, N):
    print('Bifurcation - Run {} inference...'.format(r + 1))
    fname = 'Bifurcation/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.fit(x, verb=verb)
    np.save('Bifurcation/CARDAMOM/score_{}'.format(r + 1), model.inter)


# Inference for Trifurcation
for r in range(0, N):
    print('Trifurcation - Run {} inference...'.format(r + 1))
    fname = 'Trifurcation/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.fit(x, verb=verb)
    np.save('Trifurcation/CARDAMOM/score_{}'.format(r + 1), model.inter)

# Inference for tree-like networks
for n in [5, 10, 20, 50, 100]:
    runtime = np.zeros(N)
    for r in range(N):
        print('Trees{} - Run {} inference...'.format(n,r+1))
        fname = 'Trees{}/Data/data_{}.txt'.format(n,r+1)
        data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
        time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
        x = data.T
        x[:,0] = time
        G = np.size(x, 1)
        model = NetworkModel(G - 1)
        t0 = timer.time()
        model.fit(x, verb=verb)
        t1 = timer.time()
        runtime[r] = t1 - t0
        score = model.inter
        np.save('Trees{}/CARDAMOM/score_{}'.format(n,r+1), score)
    # Save running times
    np.savetxt('Trees{}/CARDAMOM/runtime.txt'.format(n), runtime.T)
