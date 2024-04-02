# Marginal distributions (original data | inferred network | null network)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  matplotlib.colors import LinearSegmentedColormap, Normalize

# Load the data
path = '../../Semrau/Data/'
data_real = np.loadtxt(path+'data_real.txt', dtype=float, delimiter='\t')

# Load genes and sort them
genes = np.loadtxt(path+'Semrau2017panel_genes.txt', dtype='str')[1:,1]
order = np.argsort(genes)
genes = genes[order]
G = genes.size

# Order the data
order = [0] + list(order+1)
data_real = data_real[:,order]

# Load simulated data
data_netw, data_null = [], []
N = 10 # Number of simulated datasets
for n in range(N):
    # Inferred network
    path_netw = path + f'data_model_0{n}.txt'
    data = np.loadtxt(path_netw, dtype=float, delimiter='\t')[1:]
    data_netw.append(np.delete(data, 1, axis=1)[:,order])
    # Without interactions
    path_null = path + f'data_model_1{n}.txt'
    data = np.loadtxt(path_null, dtype=float, delimiter='\t')[1:]
    data_null.append(np.delete(data, 1, axis=1)[:,order])

# Timepoints
time = np.sort(list(set(data_real[:,0])))
T = time.size

# # *********************************
# # Compute EM distances and p-values
# import ot; from scipy.stats import ks_2samp
# def emdistances1D(x1, x2):
#     """List of 1D marginal earth mover's distances"""
#     return [ot.emd2_1d(x1[:,i],x2[:,i]) for i in range(x1.shape[1])]
# def normalize(x1, x2):
#     """Normalize genes: particular choice to get values between 0 and 1"""
#     z1, z2 = np.zeros(x1.shape), np.zeros(x2.shape)
#     for i in range(x1.shape[1]):
#         s = (1+max(np.max(x1[:,i]),np.max(x2[:,i]))**2)**0.5
#         z1[:,i], z2[:,i] = x1[:,i]/s, x2[:,i]/s
#     return z1, z2
# dist_netw, pval_netw = np.zeros((N,T)), np.zeros((N,T,G))
# dist_null, pval_null = np.zeros((N,T)), np.zeros((N,T,G))
# for n in range(N):
#     for t in range(T):
#         # Select data n at time t
#         x_real = data_real[data_real[:,0]==time[t],1:]
#         x_netw = data_netw[n][data_netw[n][:,0]==time[t],1:]
#         x_null = data_null[n][data_null[n][:,0]==time[t],1:]
#         # Sum of 1D marginal EM distances
#         dist_netw[n,t] = np.sum(emdistances1D(*normalize(x_real,x_netw)))
#         dist_null[n,t] = np.sum(emdistances1D(*normalize(x_real,x_null)))
#         # Individual p-values
#         for i in range(G):
#             pval_netw[n,t,i] = ks_2samp(x_real[:,i],x_netw[:,i])[1]
#             pval_null[n,t,i] = ks_2samp(x_real[:,i],x_null[:,i])[1]
# np.save('mdist_netw', dist_netw); np.save('pval_netw', pval_netw)
# np.save('mdist_null', dist_null); np.save('pval_null', pval_null)
# # *********************************

# Load pre-computed values
dist_netw, pval_netw = np.load('mdist_netw.npy'), np.load('pval_netw.npy')
dist_null, pval_null = np.load('mdist_null.npy'), np.load('pval_null.npy')

# Figure
fig = plt.figure(figsize=(8,8.1))
grid = gs.GridSpec(6, 4, wspace=0, hspace=0,
    width_ratios=[0.09,1.48,0.32,1],
    height_ratios=[0.49,0.2,0.031,0.85,0.22,0.516])
panelA = grid[0,:]
panelB = grid[3,1].subgridspec(3, 2, wspace=0.25)
panelr = grid[2:4,3].subgridspec(2, 1, hspace=0.4)
panelC = panelr[0]
panelD = panelr[1]
panelu = grid[5,:].subgridspec(1, 4, width_ratios=[1,1,1,0.1], wspace=0.25)

# ax = plt.subplot(grid[2,:-1])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# ax = plt.subplot(grid[3,0])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# ax = plt.subplot(grid[3,2])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# ax = plt.subplot(grid[1,:])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# ax = plt.subplot(grid[4,:])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# Panel settings
opt = {'xy': (0,1), 'xycoords': 'axes fraction', 'fontsize': 10,
    'textcoords': 'offset points', 'annotation_clip': False}

# Color settings
tab1 = plt.get_cmap('tab10')
tab2 = plt.get_cmap('tab20c')
tab3 = plt.get_cmap('Pastel1')
colors = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
    '#d9ef8b','#a6d96a','#66bd63','#1a9850']

# A. KS test p-values
axA = plt.subplot(panelA)
axA.annotate('A', xytext=(-14,6), fontweight='bold', **opt)
axA.annotate('KS test p-values', xytext=(0,6), **opt)
# axA.set_title('KS test p-values', fontsize=10)
pval = np.mean(pval_netw, axis=0)
cmap = LinearSegmentedColormap.from_list('pvalue', colors)
norm = Normalize(vmin=0, vmax=0.1)
# Plot the heatmap
im = axA.imshow(pval, cmap=cmap, norm=norm)
axA.set_aspect('equal','box')
axA.set_xlim(-0.5,G-0.5)
axA.set_ylim(T-0.5,-0.5)
# Create colorbar
divider = make_axes_locatable(axA)
cax = divider.append_axes('right', '1.5%', pad='2%')
cbar = axA.figure.colorbar(im, cax=cax, extend='max')
pticks = np.array([0,1,3,5,7,9])
cbar.set_ticks(pticks/100 + 0.0007)
cbar.ax.set_yticklabels([0]+[f'{p}%' for p in pticks[1:]], fontsize=6)
cbar.ax.spines[:].set_visible(False)
cbar.ax.tick_params(axis='y',direction='out', length=1.5, pad=1.5)
axA.set_xticks(np.arange(G))
axA.set_yticks(np.arange(T))
axA.set_xticklabels(genes, rotation=45, ha='right', rotation_mode='anchor',
    fontsize=6.5)
axA.set_yticklabels([f'{int(t)}h' for t in time], fontsize=6.5)
axA.spines[:].set_visible(False)
axA.set_xticks(np.arange(G+1)-0.5, minor=True)
axA.set_yticks(np.arange(T+1)-0.5, minor=True)
axA.grid(which='minor', color='w', linestyle='-', linewidth=1)
axA.tick_params(which='minor', bottom=False, left=False)
axA.tick_params(which='major', bottom=False, left=False)
axA.tick_params(axis='x',direction='out', pad=-0.1)
axA.tick_params(axis='y',direction='out', pad=-0.1)

# B. Marginal distributions
n = 5
vgenes = [12,37]
vtimes = [0,5,8]
for j, g in enumerate(vgenes):
    nmax = max(np.max(data_real[:,g]),np.max(data_netw[n][:,g])) + 1
    nbins = int(nmax/2)
    bins = np.linspace(0, nmax, nbins)
    for i, t in enumerate(vtimes):
        axB = plt.subplot(panelB[i,j])
        axB.tick_params(axis='both', labelsize=7)
        if i == 0 and j == 0:
            axB.annotate('B', xytext=(-28,6), fontweight='bold', **opt)
            axB.set_title(genes[g-1], fontsize=10)
        if i == 0 and j == len(vgenes)-1:
            axB.set_title(genes[g-1], fontsize=10)
        if i < len(vtimes)-1:
            axB.tick_params(labelbottom=False)
        if i == len(vtimes)-1:
            axB.set_xlabel('mRNA (copies per cell)', fontsize=7, labelpad=0)
        if i == 1 and j == 0:
            axB.set_ylabel('Probability density', fontsize=7)
        # Get the data for gene g t time t
        x_real = data_real[data_real[:,0]==time[t],g]
        x_netw = data_netw[n][data_netw[n][:,0]==time[t],g]
        x_null = data_null[n][data_null[n][:,0]==time[t],g]
        # Plot histograms
        c = tab3(6)
        axB.hist(x_real, density=True, bins=bins, color=c, histtype='bar',
            label='Original data')
        axB.hist(x_netw, density=True, bins=bins, histtype='step', ec=tab1(0),
            label='Inferred network', zorder=2)
        axB.hist(x_null, density=True, bins=bins, histtype='step', ec=tab1(1),
            label='Without interactions', zorder=1)
        axB.set_xlim(0,30)
        axB.set_ylim(0,0.5)
        if i == 2 and j == 1:
            axB.legend(fontsize=7, frameon=False, loc='upper left', ncol=1,
                handlelength=1, borderaxespad=0.2, borderpad=0.2,
                labelspacing=0.3, handletextpad=0.5)
        optB = {'xy': (1,1), 'xycoords': 'axes fraction', 'fontsize': 10,
            'textcoords': 'offset points', 'ha': 'right'}
        axB.annotate(f'{int(time[t])}h', xytext=(-5,-11), **optB)

# C. Average EM distance
axC = plt.subplot(panelC)
axC.annotate('C', xytext=(-28,0), fontweight='bold', **opt)
# Inferred network
m = np.mean(dist_netw, axis=0)
q0 = np.quantile(dist_netw, 0.1, axis=0)
q1 = np.quantile(dist_netw, 0.9, axis=0)
line1 = axC.plot(time, m, color=tab1(0), zorder=2)[0]
axC.fill_between(time, q0, q1, facecolor=tab2(3), zorder=2)
# Without interactions
m = np.mean(dist_null, axis=0)
q0 = np.quantile(dist_null, 0.1, axis=0)
q1 = np.quantile(dist_null, 0.9, axis=0)
line2 = axC.plot(time, m, color=tab1(1), zorder=1)[0]
axC.fill_between(time, q0, q1, facecolor=tab2(7), zorder=1)
# Axis settings
labels = ['Inferred network','Without interactions']
axC.legend([line1,line2], labels, fontsize=7,
    frameon=False, loc='upper left', handlelength=1, borderaxespad=0.25)
axC.set_xlabel('Time (h)', fontsize=7, labelpad=0)
axC.set_ylabel('EM distance', fontsize=7)
axC.set_xlim(np.min(time),np.max(time))
axC.set_ylim(0)
axC.set_xticks(time)
axC.set_xticklabels([f'{int(t)}' for t in time])
axC.tick_params(axis='both', labelsize=7)

# D. Average KS p-value
axD = plt.subplot(panelD)
axD.annotate('D', xytext=(-28,0), fontweight='bold', **opt)
# Inferred network
mpval_netw = np.mean(pval_netw, axis=2)
m = np.mean(mpval_netw, axis=0)
q0 = np.quantile(mpval_netw, 0.1, axis=0)
q1 = np.quantile(mpval_netw, 0.9, axis=0)
line1 = axD.plot(time, m, color=tab1(0), zorder=2)[0]
axD.fill_between(time, q0, q1, facecolor=tab2(3), zorder=2)
# Without interactions
mpval_null = np.mean(pval_null, axis=2)
m = np.mean(mpval_null, axis=0)
q0 = np.quantile(mpval_null, 0.1, axis=0)
q1 = np.quantile(mpval_null, 0.9, axis=0)
line2 = axD.plot(time, m, color=tab1(1), zorder=1)[0]
axD.fill_between(time, q0, q1, facecolor=tab2(7), zorder=1)
# Axis settings
labels = ['Inferred network','Without interactions']
axD.legend([line1,line2], labels, fontsize=7,
    frameon=False, loc='lower left', handlelength=1, borderaxespad=0.25)
axD.set_xlabel('Time (h)', fontsize=7, labelpad=0)
axD.set_ylabel('KS p-value', fontsize=7)
axD.set_xlim(np.min(time),np.max(time))
axD.set_ylim(0,1)
axD.set_xticks(time)
axD.set_xticklabels([f'{int(t)}' for t in time])
axD.tick_params(axis='both', labelsize=7)

# # Translate panel C
# pos1 = axC.get_position() # get the original position 
# pos2 = [pos1.x0+0.011, pos1.y0, pos1.width, pos1.height] 
# axC.set_position(pos2) # set a new position
# # Translate panel D
# pos1 = axD.get_position() # get the original position 
# pos2 = [pos1.x0+0.011, pos1.y0, pos1.width, pos1.height] 
# axD.set_position(pos2) # set a new position

# UMAP representations (original data | inferred network | null network)
from matplotlib.lines import Line2D

# Data location
path_real = '../../Semrau/Data/data_real.txt'
path_netw = '../../Semrau/Data/data_model_00.txt'
path_null = '../../Semrau/Data/data_model_10.txt'

# Load the data
data_real = np.loadtxt(path_real, dtype=float, delimiter='\t')
data_netw = np.loadtxt(path_netw, dtype=float, delimiter='\t')[1:]
data_null = np.loadtxt(path_null, dtype=float, delimiter='\t')[1:]

# Remove stimulus
data_netw = np.delete(data_netw, 1, axis=1)
data_null = np.delete(data_null, 1, axis=1)

# Remove Sparc gene (index = 34)
data_real = np.delete(data_real, 34, axis=1)
data_netw = np.delete(data_netw, 34, axis=1)
data_null = np.delete(data_null, 34, axis=1)

# # ***************************
# # Compute the UMAP projection
# from umap import UMAP
# reducer = UMAP(random_state=42, min_dist=0.15)
# proj = reducer.fit(data_real[:,1:])
# x_real = proj.transform(data_real[:,1:]); np.save('x_real', x_real)
# x_netw = proj.transform(data_netw[:,1:]); np.save('x_netw', x_netw)
# x_null = proj.transform(data_null[:,1:]); np.save('x_null', x_null)
# # ***************************

# Load pre-transformed data
x_real = np.load('x_real.npy')
x_netw = np.load('x_netw.npy')
x_null = np.load('x_null.npy')

# Figure parts
ax0 = plt.subplot(panelu[0,0])
ax1 = plt.subplot(panelu[0,1])
ax2 = plt.subplot(panelu[0,2])
ax3 = plt.subplot(panelu[0,3])

# Axis settings
def configure(ax):
    # ax.set_aspect('equal','box')
    ax.set_xlabel('UMAP1', fontsize=7, weight='bold')
    ax.set_ylabel('UMAP2', fontsize=7, weight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines[['top','right']].set_visible(False)

# Panel settings
opt = {'xy': (0,1), 'xycoords': 'axes fraction', 'fontsize': 10,
    'textcoords': 'offset points', 'annotation_clip': False}
x0, y0, d0 = -11.5, 0, 14
size = 2

# Timepoint colors
time = np.sort(list(set(data_real[:,0]))); T = time.size
cmap = [plt.get_cmap('viridis',T)(i) for i in range(T)]
# cmap = [plt.get_cmap('YlGnBu',T+1)(i+1) for i in range(T)]
c_real = [cmap[np.argwhere(time==t)[0,0]] for t in data_real[:,0]]
c_netw = [cmap[np.argwhere(time==t)[0,0]] for t in data_netw[:,0]]
c_null = [cmap[np.argwhere(time==t)[0,0]] for t in data_null[:,0]]

# E. Original data
configure(ax0)
title = 'Original data'
ax0.annotate('E', xytext=(x0,y0), fontweight='bold', **opt)
ax0.annotate(title, xytext=(x0+d0,y0), **opt)
# ax0.set_title(title, fontsize=10)
ax0.scatter(x_real[:,0], x_real[:,1], c=c_real, s=size)
ax0.spines.left.set_bounds((ax0.get_ylim()[0],0.96*ax0.get_ylim()[1]))
ax0.spines.bottom.set_bounds((ax0.get_xlim()[0],0.98*ax0.get_xlim()[1]))

# F. Inferred network
configure(ax1)
title = 'Inferred network'
ax1.annotate('F', xytext=(x0,y0), fontweight='bold', **opt)
ax1.annotate(title, xytext=(x0+d0,y0), **opt)
# ax1.set_title(title, fontsize=10)
ax1.scatter(x_netw[:,0], x_netw[:,1], c=c_netw, s=size)
ax1.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())
ax1.spines.left.set_bounds((ax0.get_ylim()[0],0.96*ax0.get_ylim()[1]))
ax1.spines.bottom.set_bounds((ax0.get_xlim()[0],0.98*ax0.get_xlim()[1]))

# G. Without interactions
configure(ax2)
title = 'Without interactions'
ax2.annotate('G', xytext=(x0,y0), fontweight='bold', **opt)
ax2.annotate(title, xytext=(x0+d0,y0), **opt)
# ax2.set_title(title, fontsize=10)
ax2.scatter(x_null[:,0], x_null[:,1], c=c_null, s=size)
ax2.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())
ax2.spines.left.set_bounds((ax0.get_ylim()[0],0.96*ax0.get_ylim()[1]))
ax2.spines.bottom.set_bounds((ax0.get_xlim()[0],0.98*ax0.get_xlim()[1]))

# Time legend panel
labels = [f'{int(time[k])}h' for k in range(T)]
lines = [Line2D([0], [0], color=cmap[k], lw=5) for k in range(T)]
ax3.legend(lines, labels, ncol=1, frameon=False, borderaxespad=0,
    loc='lower left', handlelength=0.25, fontsize=7,
    borderpad=0, labelspacing=0.58, bbox_to_anchor=(-0.2,-0.01))

ax3.text(-0.5, 0.93, 'Time:', transform=ax3.transAxes, fontsize=7)
ax3.axis('off')

# ax3.get_xaxis().set_visible(False)
# ax3.get_yaxis().set_visible(False)

# Export the figure
fig.savefig('figure7.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
