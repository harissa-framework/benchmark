# Test networks and simulated data
import sys; sys.path += ['../..']
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from gen_traj import network as model
from Packages.harissa.utils import plot_network as nt
nt.activ = '#28AD27'
nt.inhib = '#ED3A28'

# Data location
path = '../../Benchmark/'

# List of benchmarks
networks = ['FN4', 'CN5', 'FN8', 'BN8', 'Tree']
name = {'FN4': 'Network4', 'CN5': 'Cycle', 'FN8': 'Trifurcation',
    'BN8': 'Bifurcation', 'Tree': 'Trees5'}
size = [5, 6, 9, 9, 6]

# Number of datasets for each benchmark
N = 10

# Figure
fig = plt.figure(figsize=(8.53,8.185))
grid = gs.GridSpec(6, 4, height_ratios=[0.07,1,1,1,1,1], hspace=0.3,
    width_ratios=[0.85,0,1.55,0.8], wspace=0.3)

for i in range(6):
    for j in range(4):
        ax = plt.subplot(grid[i,j])
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
        if j == 1: ax.axis('off')

# Axis settings
def configure(ax):
    # ax.set_aspect('equal','box')
    ax.set_xlabel('UMAP1', fontsize=6, labelpad=2)
    ax.set_ylabel('UMAP2', fontsize=6, labelpad=2)
    ax.tick_params(axis='both', labelsize=6)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines[['top','right']].set_visible(False)

# Panel settings
opt = {'xy': (0,1), 'xycoords': 'axes fraction', 'fontsize': 10,
    'textcoords': 'offset points', 'annotation_clip': False}

# Color settings
c_gene = [plt.get_cmap('Set2')(i) for i in range(8)]
c_umap = [plt.get_cmap('viridis',10)(i) for i in range(10)]

# A. Networks
axA = plt.subplot(grid[0,0])
x0, y0 = 0, 6
axA.annotate('A', xytext=(x0,y0), fontweight='bold', **opt)
axA.annotate('Networks', xytext=(x0+14,y0), **opt)
# Legend panel
c = [nt.activ, nt.inhib]
labels = ['Activation','Inhibition']
lines = [Line2D([0], [0], color=col) for col in c]
axA.legend(lines, labels, ncol=2, frameon=False, borderaxespad=0, borderpad=0,
    loc='upper left', handlelength=1.5, columnspacing=1.7, fontsize=6,
    bbox_to_anchor=[0.0055,0.75])
axA.axis('off')
# axA.get_xaxis().set_visible(False)
# axA.get_yaxis().set_visible(False)
# All networks
for n, network in enumerate(networks):
    ax = plt.subplot(grid[n+1,0])
    for direction in ['left', 'right', 'bottom', 'top']:
        ax.spines[direction].set_color('lightgray')
        ax.tick_params(axis='x', bottom=True, labelbottom=True)
        ax.tick_params(axis='y', left=True, labelleft=True)
    # Network name
    # xn, yn = 0.015, 0.85
    xn, yn = 0.015, 0.83
    optn = {'fontsize': 9, 'transform': ax.transAxes, 'ha': 'left'}
    ax.text(xn, yn, network, **optn)
    ax.text(xn, yn+0.01, network, color='none', zorder=0, bbox=dict(
        boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)
    # Plot the network
    p = np.load(f'pos_{network}.npy')
    p[:,1] -= 0.1
    nt.plot_network(model[network].inter, axes=ax, scale=7, layout=p,
        fontsize=7, vcolor=p.shape[0]*['#5C5C5C'], bend=0.14)
    ax.axis('off')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

# B. Trajectories
axB = plt.subplot(grid[0,2])
x0, y0 = -22, 6
axB.annotate('B', xytext=(x0,y0), fontweight='bold', **opt)
axB.annotate('Trajectories (mRNA levels)', xytext=(x0+14,y0), **opt)
# Legend panel
labels = [f'Gene {i+1}' for i in range(8)]
lines = [Line2D([0], [0], color=c) for c in c_gene]
axB.legend(lines, labels, ncol=4, frameon=False, borderaxespad=0, borderpad=0,
    loc='upper right', handlelength=1.5, columnspacing=2, fontsize=6,
    bbox_to_anchor=[1,0.75])
axB.axis('off')
# axB.get_xaxis().set_visible(False)
# axB.get_yaxis().set_visible(False)
# All trajectories
for n, network in enumerate(networks):
    panelB = grid[n+1,2].subgridspec(2, 1, hspace=0.1)
    # Load trajectory data
    x = np.load(f'traj_{network}.npy')
    G, time = x.shape[2] - 1, x[0,:,0]
    xtimes = np.linspace(np.min(time),np.max(time),5, dtype=int)
    # Single-cell trajectory
    ax = plt.subplot(panelB[0])
    traj = x[1,:,1:].T
    for i in range(G):
        ax.plot(time, traj[i], c=c_gene[i], lw=1.4)
    ax.set_ylabel(r'$M$    ', fontsize=6, labelpad=0)
    ax.tick_params(axis='x', length=2, labelbottom=False)
    ax.tick_params(axis='y', labelsize=6, length=3, pad=1)
    ax.set_xlim(np.min(time),np.max(time))
    ax.set_ylim(0,500)
    ax.set_xticks(xtimes)
    ax.set_yticks([0,400])
    ax.spines[['top','right']].set_visible(False)
    ax.spines.left.set_bounds((0,400))
    # Population trajectory
    ax = plt.subplot(panelB[1])
    traj = x[0,:,1:].T
    for i in range(G):
        ax.plot(time, traj[i], c=c_gene[i], lw=1.4)
    ax.set_ylabel(r'$\langle M \rangle$    ', fontsize=6, labelpad=0)
    ax.tick_params(axis='x', labelsize=6, length=2, pad=1.5)
    ax.tick_params(axis='y', labelsize=6, length=3, pad=1)
    ax.set_xlim(np.min(time),np.max(time))
    ax.set_ylim(0,125)
    ax.set_xticks(xtimes)
    ax.set_yticks([0,100])
    ax.set_xticklabels(xtimes)
    ax.spines[['top','right']].set_visible(False)
    ax.spines.left.set_bounds((0,100))
    # Timepoints
    opttime = {'xy': (0.81,0), 'xycoords': 'axes fraction', 'fontsize': 6,
        'textcoords': 'offset points', 'annotation_clip': False}
    ax.annotate('Time (h)', xytext=(0,-7.8), **opttime)

# C. Snapshots
axC = plt.subplot(grid[0,3])
x0, y0 = -9, 6
axC.annotate('C', xytext=(x0,y0), fontweight='bold', **opt)
axC.annotate('Snapshots', xytext=(x0+14,y0), **opt)
# Legend panel
time = (0,6,12,24,36,48,60,72,84,96)
T = len(time)
labels = [f'{int(t)}h' for t in time]
lines = [Line2D([0], [0], color=c, ls='', marker='.', ms=5) for c in c_umap]
axC.legend(lines, labels, ncol=5, frameon=False, borderaxespad=0, borderpad=0,
    loc='upper right', handlelength=0.01, columnspacing=1.25, fontsize=6,
    bbox_to_anchor=[1,0.75], handletextpad=0.55)
axC.axis('off')
# axC.get_xaxis().set_visible(False)
# axC.get_yaxis().set_visible(False)

# Routine for each benchmark
for n, network in enumerate(networks):
    dataset = []
    for r in range(N):
        # Load the data
        file = path+f'{name[network]}/Data/data_{r+1}.txt'
        data = np.loadtxt(file, delimiter='\t')[:,1:].T
        # Remove stimulus
        data = np.delete(data, 1, axis=1)
        # Dither the first timepoints
        # data[:300,1:] += np.random.poisson(10, size=data[:300,1:].shape)
        data[:300,1:] += np.random.gamma(2, scale=5, size=data[:300,1:].shape)
        # Gather data
        dataset.append(data)

    # UMAP representation
    # data = np.concatenate(dataset, axis=0)
    data = dataset[0] # First dataset of the benchmark

    # # ***************************
    # # Compute the UMAP projection
    # from umap import UMAP
    # reducer = UMAP(random_state=0, min_dist=0.15)
    # proj = reducer.fit(data[:,1:])
    # x = proj.transform(data[:,1:]); np.save(f'x_{network}', x)
    # # ***************************

    # Load pre-transformed data
    x = np.load(f'x_{network}.npy')

    time = np.sort(list(set(data[:,0]))); T = time.size
    c_x = [c_umap[np.argwhere(time==t)[0,0]] for t in data[:,0]]

    ax = plt.subplot(grid[n+1,3])
    configure(ax)
    ax.scatter(x[:,0], x[:,1], c=c_x, s=2)
    ax.spines.left.set_bounds((ax.get_ylim()[0],0.92*ax.get_ylim()[1]))
    ax.spines.bottom.set_bounds((ax.get_xlim()[0],0.98*ax.get_xlim()[1]))

# Export the figure
fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight', pad_inches=0.04)
