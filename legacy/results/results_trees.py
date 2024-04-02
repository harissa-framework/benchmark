# Affiche les resultats pour la figure 6
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# Colors
cT = 'lightgray'
cG = 'dodgerblue'
cH = 'orangered'
cC = 'purple'
cS = 'orange'
cP = 'tab:green'
cPr = 'black'

size = [5, 10, 20, 50, 100]

N = 10
K = len(size)

# Compute AUPR mean curves
d = ''

mauprTd, mauprGd, mauprHd, mauprCd, mauprSd, mauprPrd, mauprPd = [], [], [], [], [], [], []
mauprTu, mauprGu, mauprHu, mauprCu, mauprSu, mauprPru, mauprPu = [], [], [], [], [], [], []

for n in size:
    print('Importing data for Trees{}{}...'.format(n, d))
    G = n + 1
    NETWORK = np.zeros((N,G,G))
    HARISSA = np.zeros((N,G,G))
    CARDAMOM = np.zeros((N, G, G))
    SINCERITIES = np.zeros((N,G,G))
    PIDC = np.zeros((N,G,G))
    GENIE3 = np.zeros((N,G,G))
    PEARSON = np.zeros((N, G, G))

    # AUPR values
    auprTd, auprGd, auprHd, auprCd, auprSd, auprPrd, auprPd = [], [], [], [], [], [], []
    auprTu, auprGu, auprHu, auprCu, auprSu, auprPru, auprPu = [], [], [], [], [], [], []

    for r in range(N):

        # True network
        file = 'Trees{}{}/True/inter_{}.npy'.format(n, d, r + 1)
        NETWORK[r] = abs(np.load(file))
        # HARISSA
        file = 'Trees{}{}/HARISSA/score_{}.npy'.format(n, d,r + 1)
        HARISSA[r] = abs(np.load(file))
        # CARDAMOM
        file = 'Trees{}{}/CARDAMOM/score_{}.npy'.format(n, d, r + 1)
        CARDAMOM[r] = abs(np.load(file))
        # GENIE3
        file = 'Trees{}{}/GENIE3/score_{}.npy'.format(n, d, r + 1)
        GENIE3[r] = abs(np.load(file)).T
        # SINCERITIES
        file = 'Trees{}{}/SINCERITIES/score_{}.npy'.format(n, d, r + 1)
        SINCERITIES[r] = abs(np.load(file))
        # PEARSON
        file = 'Trees{}{}/PEARSON/score_{}.npy'.format(n, d, r + 1)
        PEARSON[r] = abs(np.load(file))
        # PIDC
        file = 'Trees{}{}/PIDC/score_{}.npy'.format(n, d, r + 1)
        PIDC[r] = abs(np.load(file))

        # Vectorize scores: directed version
        vTd, vGd, vHd, vCd, vSd, vPrd = [], [], [], [], [], []
        for i in range(G):
            for j in range(1,G):
                if i != j:
                    vTd.append(NETWORK[r][i,j])
                    vHd.append(HARISSA[r][i,j])
                    vCd.append(CARDAMOM[r][i, j])
                    vGd.append(GENIE3[r][i,j])
                    vSd.append(SINCERITIES[r][i, j])
                    vPrd.append(PEARSON[r][i, j])
        vTd = np.array(vTd)
        vGd = np.array(vGd)
        vHd = np.array(vHd)
        vCd = np.array(vCd)
        vSd = np.array(vSd)
        vPrd = np.array(vPrd)

        # Random estimator
        auprTd.append(np.mean(vTd))

        # HARISSA AUPR
        yHd, xHd, tHd = precision_recall_curve(vTd, vHd)
        auprHd.append(auc(xHd, yHd))

        # CARDAMOM AUPR
        yCd, xCd, tCd = precision_recall_curve(vTd, vCd)
        auprCd.append(auc(xCd, yCd))

        # GENIE3 AUPR
        yGd, xGd, tGd = precision_recall_curve(vTd, vGd)
        auprGd.append(auc(xGd, yGd))

        # SINCERITIES AUPR
        ySd, xSd, tSd = precision_recall_curve(vTd, vSd)
        auprSd.append(auc(xSd, ySd))

        # PEARSON AUPR
        yPrd, xPrd, tSd = precision_recall_curve(vTd, vPrd)
        auprPrd.append(auc(xPrd, yPrd))

        # Vectorize scores: undirected version
        vTu, vGu, vHu, vCu, vSu, vPru, vPu = [], [], [], [], [], [], []
        for i in range(G):
            for j in range(i+1,G):
                vTu.append(np.max([NETWORK[r][i,j],NETWORK[r][j,i]]))
                vHu.append(np.max([HARISSA[r][i,j],HARISSA[r][j,i]]))
                vCu.append(np.max([CARDAMOM[r][i,j], CARDAMOM[r][j,i]]))
                vSu.append(np.max([SINCERITIES[r][i,j],SINCERITIES[r][j,i]]))
                vPru.append(np.max([PEARSON[r][i, j], PEARSON[r][j, i]]))
                vGu.append(np.max([GENIE3[r][i,j], GENIE3[r][j,i]]))
                vPu.append(np.max([PIDC[r][i,j],PIDC[r][j,i]]))
        vTu = np.array(vTu)
        vGu = np.array(vGu)
        vHu = np.array(vHu)
        vCu = np.array(vCu)
        vSu = np.array(vSu)
        vPu = np.array(vPu)
        vPru = np.array(vPru)

        # Random estimator
        auprTu.append(np.mean(vTu))

        # HARISSA AUPR
        yHu, xHu, tHu = precision_recall_curve(vTu, vHu)
        auprHu.append(auc(xHu, yHu))

        # CARDAMOM AUPR
        yCu, xCu, tCu = precision_recall_curve(vTu, vCu)
        auprCu.append(auc(xCu, yCu))

        # GENIE3 AUPR
        yGu, xGu, tGu = precision_recall_curve(vTu, vGu)
        auprGu.append(auc(xGu, yGu))

        # SINCERITIES AUPR
        ySu, xSu, tSu = precision_recall_curve(vTu, vSu)
        auprSu.append(auc(xSu, ySu))

        # PEARSON AUPR
        yPru, xPru, tPru = precision_recall_curve(vTu, vPru)
        auprPru.append(auc(xPru, yPru))

        # PIDC AUPR
        yPu, xPu, tPu = precision_recall_curve(vTu, vPu)
        auprPu.append(auc(xPu, yPu))

    # Mean curves (directed)
    mauprTd.append(np.mean(auprTd))
    mauprHd.append(np.mean(auprHd))
    mauprCd.append(np.mean(auprCd))
    mauprGd.append(np.mean(auprGd))
    mauprSd.append(np.mean(auprSd))
    mauprPrd.append(np.mean(auprPrd))

    # Mean curves (undirected)
    mauprTu.append(np.mean(auprTu))
    mauprHu.append(np.mean(auprHu))
    mauprCu.append(np.mean(auprCu))
    mauprPu.append(np.mean(auprPu))
    mauprGu.append(np.mean(auprGu))
    mauprSu.append(np.mean(auprSu))
    mauprPru.append(np.mean(auprPru))


# Figure
fig = plt.figure(figsize=(12,8))
grid = gs.GridSpec(2, 2, wspace=0.25, hspace=0.58)

# Line style
style = {'ls': '--', 'marker': '.', 'ms': 8}

# 1. Without dropouts

# Directed
ax = plt.subplot(grid[0,0])
# Label A
ax.text(0, 1.03, 'A', fontsize=14, weight='bold',
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
ax.text(0.07, 1.03, 'Directed', fontsize=14,
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
# AUPR
ax.plot(size, mauprHd, color=cH, label='HARISSA', **style)
ax.plot(size, mauprCd, color=cC, label='CARDAMOM', **style)
ax.plot(size, mauprGd, color=cG, label='GENIE3', **style)
ax.plot(size, mauprSd, color=cS, label='SINCERITIES', **style)
ax.plot(size, mauprTd, color=cT, label='Random estimator', **style)


ax.set_xlim(size[0]-5,size[-1]+5)
ax.set_ylim(0,1)
ax.set_ylabel('AUPR')
ax.legend(loc='upper right', borderaxespad=0, frameon=False)
# ax.set_xticks(size)
xticks = np.sort(list(set(size + list(range(10,101,10)))))
ax.set_xticks(xticks)
ax.set_xlabel('No. of genes')

# Undirected
ax = plt.subplot(grid[0,1])
# Label A
ax.text(0, 1.03, 'B', fontsize=14, weight='bold',
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
ax.text(0.07, 1.03, 'Undirected', fontsize=14,
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
# AUPR
ax.plot(size, mauprHu, color=cH, label='HARISSA', **style)
ax.plot(size, mauprCu, color=cC, label='CARDAMOM', **style)
ax.plot(size, mauprGu, color=cG, label='GENIE3', **style)
ax.plot(size, mauprSu, color=cS, label='SINCERITIES', **style)
ax.plot(size, mauprPu, color=cP, label='PIDC', **style)
ax.plot(size, mauprPru, color=cPr, label='PEARSON', **style)
ax.plot(size, mauprTu, color=cT, label='Random estimator', **style)

ax.set_xlim(size[0]-5,size[-1]+5)
ax.set_ylim(0,1)
ax.set_ylabel('AUPR')
ax.legend(loc='upper right', borderaxespad=0, frameon=False)
# ax.set_xticks(size)
xticks = np.sort(list(set(size + list(range(10,101,10)))))
ax.set_xticks(xticks)
ax.set_xlabel('No. of genes')

# 2. With dropouts

# # Directed
# ax = plt.subplot(grid[1,0])
# # Label C
# ax.text(0, 1.03, 'C', fontsize=14, weight='bold',
#     transform=ax.transAxes,
#     horizontalalignment='left', verticalalignment='bottom')
# ax.text(0.07, 1.03, 'Directed (dropouts)', fontsize=14,
#     transform=ax.transAxes,
#     horizontalalignment='left', verticalalignment='bottom')
# # AUPR
# ax.plot(size, mauprTdD, color=cT, label='Random', **style)
# ax.plot(size, mauprHdD, color=cH, label='HARISSA', **style)

# ax.set_xlim(size[0]-5,size[-1]+5)
# ax.set_ylim(0,1)
# ax.set_ylabel('AUPR')
# ax.legend(loc='upper right', borderaxespad=0, frameon=False)
# # ax.set_xticks(size)
# xticks = np.sort(list(set(size + list(range(10,101,10)))))
# ax.set_xticks(xticks)
# ax.set_xlabel('No. of genes')

# # Undirected
# ax = plt.subplot(grid[1,1])
# # Label D
# ax.text(0, 1.03, 'D', fontsize=14, weight='bold',
#     transform=ax.transAxes,
#     horizontalalignment='left', verticalalignment='bottom')
# ax.text(0.07, 1.03, 'Undirected (dropouts)', fontsize=14,
#     transform=ax.transAxes,
#     horizontalalignment='left', verticalalignment='bottom')
# # AUPR
# ax.plot(size, mauprTuD, color=cT, label='Random', **style)
# ax.plot(size, mauprHuD, color=cH, label='HARISSA', **style)

# ax.set_xlim(size[0]-5,size[-1]+5)
# ax.set_ylim(0,1)
# ax.set_ylabel('AUPR')
# ax.legend(loc='upper right', borderaxespad=0, frameon=False)
# # ax.set_xticks(size)
# xticks = np.sort(list(set(size + list(range(10,101,10)))))
# ax.set_xticks(xticks)
# ax.set_xlabel('No. of genes')

# Export
fig.savefig('results_Trees_comp.pdf', bbox_inches='tight')

