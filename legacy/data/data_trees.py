# Generate data for tree-like networks
import sys; sys.path += ['../']
import numpy as np
from Packages.harissa import Tree

np.random.seed(0)

# Number of cells
C = 1000

# Number of runs
N = 10

# Time points
k = np.linspace(0, C, 11, dtype='int')
t = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96] # np.linspace(0, 25, 10, dtype='int')
print(f't = {t}')
time = np.zeros(C, dtype='int')
for i in range(10): time[k[i]:k[i+1]] = t[i]

# Number of genes
for G in [5, 10, 20, 50, 100]:
    print(f'Tree-like networks with {G} genes:')
    # Prepare data
    data = np.zeros((C+1,G+2), dtype='int')
    data[0][1:] = np.arange(G+1)
    data[1:,0] = time # Time points
    data[1:,1] = 1 * (time > 0) # Stimulus

    for r in range(N):
        print(f'Run {r+1}...')
        
        # Initialize the model
        model = Tree(G)
        model.d[0] = 1
        model.d[1] = 0.2
        model.d /= 4

        # Save true network topology
        inter = 1 * (abs(model.inter) > 0)
        np.save(f'Trees{G}/True/inter_{r+1}', inter)

        # Generate data
        for k in range(C):
            # print(f'* Cell {k+1}')
            sim = model.simulate(time[k], burnin=5)
            data[k+1,2:] = np.random.poisson(sim.m[0])

        # Save data for use with PIDC
        fname = f'Trees{G}/Data/data_{r+1}.txt'
        np.savetxt(fname, data.T, fmt='%d', delimiter='\t')
