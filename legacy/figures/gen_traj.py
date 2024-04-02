# Generation of trajectory data for figure 1
from harissa import NetworkModel
import numpy as np

# Number of cells
C = 1000

# Number of timepoints
T = 500

# Definition of test networks
network = {}

# Network FN4
network['FN4'] = NetworkModel(4)
network['FN4'].d[0] = 0.2
network['FN4'].d[1] = 0.04
network['FN4'].basal[1:] = -5
network['FN4'].inter[0,1] = 10
network['FN4'].inter[1,2] = 10
network['FN4'].inter[1,3] = 10
network['FN4'].inter[3,4] = 10
network['FN4'].inter[4,1] = -10
network['FN4'].inter[2,2] = 10
network['FN4'].inter[3,3] = 10

# Network CN5
network['CN5'] = NetworkModel(5)
network['CN5'].d[0] = 0.5
network['CN5'].d[1] = 0.1
network['CN5'].basal[1:] = [-5, 4, 4, -5, -5]
network['CN5'].inter[0,1] = 10
network['CN5'].inter[1,2] = -10
network['CN5'].inter[2,3] = -10
network['CN5'].inter[3,4] = 10
network['CN5'].inter[4,5] = 10
network['CN5'].inter[5,1] = -10

# Network FN8
network['FN8'] = NetworkModel(8)
network['FN8'].d[0] = 0.4
network['FN8'].d[1] = 0.08
network['FN8'].basal[1:] = -5
network['FN8'].inter[0,1] = 10
network['FN8'].inter[1,2] = 10
network['FN8'].inter[2,3] = 10
network['FN8'].inter[3,4] = 10
network['FN8'].inter[3,5] = 10
network['FN8'].inter[3,6] = 10
network['FN8'].inter[4,1] = -10
network['FN8'].inter[5,1] = -10
network['FN8'].inter[6,1] = -10
network['FN8'].inter[4,4] = 10
network['FN8'].inter[5,5] = 10
network['FN8'].inter[6,6] = 10
network['FN8'].inter[4,8] = -10
network['FN8'].inter[4,7] = -10
network['FN8'].inter[6,7] = 10
network['FN8'].inter[7,6] = 10
network['FN8'].inter[8,8] = 10

# Network BN8
network['BN8'] = NetworkModel(8)
network['BN8'].d[0] = 0.25
network['BN8'].d[1] = 0.05
network['BN8'].basal[1:] = -4
network['BN8'].inter[0,1] = 10
network['BN8'].inter[1,2] = 10
network['BN8'].inter[1,3] = 10
network['BN8'].inter[3,2] = -10
network['BN8'].inter[2,3] = -10
network['BN8'].inter[2,2] = 5
network['BN8'].inter[3,3] = 5
network['BN8'].inter[2,4] = 10
network['BN8'].inter[3,5] = 10
network['BN8'].inter[2,5] = -10
network['BN8'].inter[3,4] = -10
network['BN8'].inter[4,7] = -10
network['BN8'].inter[5,6] = -10
network['BN8'].inter[4,6] = 10
network['BN8'].inter[5,7] = 10
network['BN8'].inter[7,8] = 10
network['BN8'].inter[6,8] = -10

# Network Tree
network['Tree'] = NetworkModel(5)
network['Tree'].d[0] = 0.25
network['Tree'].d[1] = 0.05
network['Tree'].basal[1:] = -5
network['Tree'].inter[0,2] = 10
network['Tree'].inter[2,4] = 10
network['Tree'].inter[2,5] = 10
network['Tree'].inter[4,1] = 10
network['Tree'].inter[1,3] = 10


if __name__ == '__main__':
    # Number of cells to save
    N = 1
    # Simulation routine
    t = np.linspace(0, 96, T)
    for name, model in network.items():
        print(f'Generating data for network {name}...')
        G = model.basal.size - 1
        data = np.zeros((C,T,G))
        traj = np.zeros((N+1,T,G+1))
        # Simulate all trajectories
        for k in range(C):
            # print(f'Cell {k+1}')
            sim = model.simulate(t, burnin=20)
            data[k] = sim.m
        # Store the timepoints
        traj[:,:,0] = t
        # Store the average trajectory
        traj[0,:,1:] = data.mean(axis=0)
        # Store some cell trajectories
        traj[1:,:,1:] = data[:N]
        # Export the data
        np.save(f'traj_{name}', traj)
