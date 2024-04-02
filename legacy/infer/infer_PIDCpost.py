# Post-traitement de la sortie de PIDC
import os
import numpy as np

# Number of runs
N = 10

# Network4
for r in range(N):
    print('Run {} results...'.format(r+1))
    fname = 'Network4/PIDC/score_{}.txt'.format(r+1)
    plist = np.loadtxt(fname)
    G = 5
    score = np.zeros((G,G))
    for i in range(plist[:,0].size):
        g1, g2 = int(plist[i,0]), int(plist[i,1])
        score[g1,g2] = plist[i,2]
    np.save('Network4/PIDC/score_{}'.format(r+1), score)
    os.remove(fname)

# Cycle
for r in range(N):
    print('Cycle - Run {} results...'.format(r+1))
    fname = 'Cycle/PIDC/score_{}.txt'.format(r+1)
    plist = np.loadtxt(fname)
    G = 6
    score = np.zeros((G,G))
    for i in range(plist[:,0].size):
        g1, g2 = int(plist[i,0]), int(plist[i,1])
        score[g1,g2] = plist[i,2]
    np.save('Cycle/PIDC/score_{}'.format(r+1), score)
    os.remove(fname)

# Trifurcation
for r in range(N):
    print('Trifurcation - Run {} results...'.format(r+1))
    fname = 'Trifurcation/PIDC/score_{}.txt'.format(r+1)
    plist = np.loadtxt(fname)
    G = 9
    score = np.zeros((G,G))
    for i in range(plist[:,0].size):
        g1, g2 = int(plist[i,0]), int(plist[i,1])
        score[g1,g2] = plist[i,2]
    np.save('Trifurcation/PIDC/score_{}'.format(r+1), score)
    os.remove(fname)

# Tree-like networks
for n in [5,10,20,50,100]:
    for r in range(N):
        print('Trees{} - Run {} results...'.format(n,r+1))
        fname = 'Trees{}/PIDC/score_{}.txt'.format(n,r+1)
        plist = np.loadtxt(fname)
        G = n + 1
        score = np.zeros((G,G))
        for i in range(plist[:,0].size):
            g1, g2 = int(plist[i,0]), int(plist[i,1])
            score[g1,g2] = plist[i,2]
        np.save('Trees{}/PIDC/score_{}'.format(n,r+1), score)
        os.remove(fname)