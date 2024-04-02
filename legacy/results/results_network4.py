# Affiche les resultats pour la figure 5
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# Colors
cG = 'dodgerblue'
cH = 'orangered'
cC = 'purple'
cS = 'orange'
cP = 'tab:green'
cPr = 'black'

G = 5
N = 10

NETWORK = np.zeros((N,G,G))
HARISSA = np.zeros((N,G,G))
CARDAMOM = np.zeros((N,G,G))
SINCERITIES = np.zeros((N,G,G))
PIDC = np.zeros((N,G,G))
GENIE3 = np.zeros((N,G,G))
PEARSON = np.zeros((N,G,G))

# AUROC values
aurocTd, aurocGd, aurocHd, aurocCd, aurocSd, aurocPd, aurocPrd = [], [], [], [], [], [], []
aurocTu, aurocGu, aurocHu, aurocCu, aurocSu, aurocPu, aurocPru = [], [], [], [], [], [], []

# AUPR values
auprTd, auprGd, auprHd, auprCd, auprSd, auprPd, auprPrd = [], [], [], [], [], [], []
auprTu, auprGu, auprHu, auprCu, auprSu, auprPu, auprPru = [], [], [], [], [], [], []

for r in range(N):

    # True network
    file = 'Network4/True/inter_{}.npy'.format(r+1)
    NETWORK[r] = abs(np.load(file))
    # HARISSA
    file = 'Network4/HARISSA/score_{}.npy'.format(r+1)
    HARISSA[r] = abs(np.load(file))
    # CARDAMOM
    file = 'Network4/CARDAMOM/score_{}.npy'.format(r + 1)
    CARDAMOM[r] = abs(np.load(file))
    # GENIE3
    file = 'Network4/GENIE3/score_{}.npy'.format(r + 1)
    GENIE3[r] = abs(np.load(file)).T
    # SINCERITIES
    file = 'Network4/SINCERITIES/score_{}.npy'.format(r + 1)
    SINCERITIES[r] = abs(np.load(file))
    # PIDC
    file = 'Network4/PIDC/score_{}.npy'.format(r + 1)
    PIDC[r] = abs(np.load(file))
    # PEARSON
    file = 'Network4/PEARSON/score_{}.npy'.format(r + 1)
    PEARSON[r] = abs(np.load(file))

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
    aurocTd.append(0.5)
    auprTd.append(np.mean(vTd))

    # HARISSA ROC
    xHd, yHd, tHd = roc_curve(vTd, vHd)
    aurocHd.append(auc(xHd, yHd))

    # CARDAMOM ROC
    xCd, yCd, tCd = roc_curve(vTd, vCd)
    aurocCd.append(auc(xCd, yCd))

    # GENIE3 ROC
    xGd, yGd, tGd = roc_curve(vTd, vGd)
    aurocGd.append(auc(xGd, yGd))

    # SINCERITIES ROC
    xSd, ySd, tSd = roc_curve(vTd, vSd)
    aurocSd.append(auc(xSd, ySd))

    # PEARSON ROC
    xPrd, yPrd, tPrd = roc_curve(vTd, vPrd)
    aurocPrd.append(auc(xPrd, yPrd))

    # HARISSA AUPR
    yHd, xHd, tHd = precision_recall_curve(vTd, vHd)
    auprHd.append(auc(xHd, yHd))

    # CARDAMOM AUPR
    yCd, xCd, tCd = precision_recall_curve(vTd, vCd)
    auprCd.append(auc(xCd, yCd))

    # SINCERITIES AUPR
    ySd, xSd, tSd = precision_recall_curve(vTd, vSd)
    auprSd.append(auc(xSd, ySd))

    # PEARSON AUPR
    yPrd, xPrd, tPrd = precision_recall_curve(vTd, vPrd)
    auprPrd.append(auc(xPrd, yPrd))

    # GENIE3 AUPR
    yGd, xGd, tGd = precision_recall_curve(vTd, vGd)
    auprGd.append(auc(xGd, yGd))

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
    vPru = np.array(vPru)
    vPu = np.array(vPu)

    # Random estimator
    aurocTu.append(0.5)
    auprTu.append(np.mean(vTu))

    # HARISSA ROC
    xHu, yHu, tHu = roc_curve(vTu, vHu)
    aurocHu.append(auc(xHu, yHu))

    # HARISSA AUPR
    yHu, xHu, tHu = precision_recall_curve(vTu, vHu)
    auprHu.append(auc(xHu, yHu))

    # GENIE3 AUPR
    yGu, xGu, tGu = precision_recall_curve(vTu, vGu)
    auprGu.append(auc(xGu, yGu))

    # SINCERITIES AUPR
    ySu, xSu, tSu = precision_recall_curve(vTu, vSu)
    auprSu.append(auc(xSu, ySu))

    # PEARSON AUPR
    yPru, xPru, tPru = precision_recall_curve(vTu, vPru)
    auprPru.append(auc(xPru, yPru))

    # GENIE3 ROC
    xGu, yGu, tGu = roc_curve(vTu, vGu)
    aurocGu.append(auc(xGu, yGu))

    # SINCERITIES ROC
    xSu, ySu, tSu = roc_curve(vTu, vSu)
    aurocSu.append(auc(xSu, ySu))

    # PEARSON ROC
    xPru, yPru, tPru = roc_curve(vTu, vPru)
    aurocPru.append(auc(xPru, yPru))

    # PIDC AUPR
    yPu, xPu, tPu = precision_recall_curve(vTu, vPu)
    auprPu.append(auc(xPu, yPu))

    # PIDC ROC
    xPu, yPu, tPu = roc_curve(vTu, vPu)
    aurocPu.append(auc(xPu, yPu))

    # CARDAMOM ROC
    xCu, yCu, tCu = roc_curve(vTu, vCu)
    aurocCu.append(auc(xCu, yCu))

    # CARDAMOM AUPR
    yCu, xCu, tCu = precision_recall_curve(vTu, vCu)
    auprCu.append(auc(xCu, yCu))


# Styles
w = -0.01
h = 1.1

# Figure
fig = plt.figure(figsize=(12,5))
grid = gs.GridSpec(2, 2, width_ratios=[5,6])
grid.update(wspace=0.3, hspace=0.18)
# Directed
ax = plt.subplot(grid[0,0])
# Label A
ax.text(0+w, h, 'A', fontsize=14, weight='bold',
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
ax.text(0.091+w, h, 'Directed network inference', fontsize=14,
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')

# AUROC
b = np.mean(aurocTd)
ax.plot([0,6], [b,b], color='lightgray', ls='--')
prop = {'color':'black'}
test = ax.boxplot([aurocHd, aurocCd, aurocGd, aurocSd], medianprops=prop,
    patch_artist=True, widths=(.3, .3, .3, .3))
for j, c in enumerate([cH, cC, cG, cS]): test['medians'][j].set_color(c)
for patch in test['boxes']: patch.set(facecolor='white')
ax.set_xlim(0.5,4.5)
ax.set_ylim(0,1)
ax.set_ylabel('AUROC')
ax.tick_params(axis='x', labelbottom=False)

# AUPR
ax = plt.subplot(grid[1,0])
plt.xticks(fontsize=8)
b = np.mean(auprTd)
ax.plot([0,6], [b,b], color='lightgray', ls='--')
prop = {'color':'black'}
test = ax.boxplot([auprHd, auprCd, auprGd, auprSd],
    patch_artist=True, medianprops=prop, widths=(.3, .3, .3, .3))
for j, c in enumerate([cH, cC, cG, cS]): test['medians'][j].set_color(c)
for patch in test['boxes']: patch.set(facecolor='white')
ax.set_xlim(0.5,4.5)
ax.set_ylim(0,1)
ax.set_ylabel('AUPR')
ax.set_xticklabels(['HARISSA', 'CARDAMOM', 'GENIE3', 'SINCERITIES'])

# Undirected
ax = plt.subplot(grid[0,1])
# Label B
ax.text(0+w, h, 'B', fontsize=14, weight='bold',
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
ax.text(0.067+w, h, 'Undirected network inference', fontsize=14,
    transform=ax.transAxes,
    horizontalalignment='left', verticalalignment='bottom')
# AUROC
b = np.mean(aurocTu)
ax.plot([0,7], [b,b], color='lightgray', ls='--')
prop = {'color':'black'}
test = ax.boxplot([aurocHu, aurocCu, aurocGu, aurocSu, aurocPu, aurocPru], patch_artist=True,
    widths=(.3, .3, .3, .3, .3, .3))
for j, c in enumerate([cH, cC, cG, cS, cP, cPr]): test['medians'][j].set_color(c)
for patch in test['boxes']: patch.set(facecolor='white')
ax.set_xlim(0.5,6.5)
ax.set_ylim(0,1)
ax.set_ylabel('AUROC')
ax.tick_params(axis='x', labelbottom=False)

# AUPR
ax = plt.subplot(grid[1,1])
plt.xticks(fontsize=8)
b = np.mean(auprTu)
ax.plot([0,7], [b,b], color='lightgray', ls='--')
prop = {'color':'black'}
test = ax.boxplot([auprHu, auprCu, auprGu, auprSu, auprPu, auprPru], patch_artist=True,
    medianprops=prop, widths=(.3, .3, .3, .3, .3, .3))
for j, c in enumerate([cH, cC, cG, cS, cP, cPr]): test['medians'][j].set_color(c)
for patch in test['boxes']: patch.set(facecolor='white')
ax.set_xlim(0.5,6.5)
ax.set_ylim(0,1)
ax.set_ylabel('AUPR')
ax.set_xticklabels(['HARISSA', 'CARDAMOM', 'GENIE3', 'SINCERITIES', 'PIDC','PEARSON'])

# Export
fig.savefig('results_Network4_comp.pdf', bbox_inches='tight')