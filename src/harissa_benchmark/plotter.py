# Some utility functions for bencharking Harissa
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

def prepare_matrix(matrix):
    n = matrix.shape[0]
    # remove first column and the diagonal
    mask = ~np.hstack((
        np.ones((n, 1), dtype=bool), 
        np.eye(n, dtype=bool)[:, 1:]
    ))

    # assert matrix[mask].shape == (n*(n - 2) + 1,)

    return np.abs(matrix[mask])


class InteractionPlotter:
    def __init__(self, inter) -> None:
        self._inter = prepare_matrix(inter)

    @property
    def inter(self):
        return self._inter
    
    @inter.setter
    def inter(self, inter):
        self._inter = prepare_matrix(inter)
    

    def roc(self, score):
        """
        Compute a receiver operating characteristic (ROC) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        x, y, _ = roc_curve(self.inter, prepare_matrix(score))
        return x, y

    def auroc(self, score):
        """
        Area under ROC curve (see function `roc`).
        """
        x, y = self.roc(score)
        return auc(x, y), (x, y) 

    def plot_rocs(self, scores, ax):
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        ax.plot([0,1], [0,1], color='gray', ls='--', label='Random (0.50)')
        for score in scores:
            auc, curve = self.auroc(score)
            ax.plot(*curve, label=f'Score ({auc:.2f})')
        # ax.set_xlim(0,1)
        # ax.set_ylim(0)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc='lower right')

    def pr(self, score):
        """
        Compute a precision recall (PR) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        y, x, _ = precision_recall_curve(self.inter, prepare_matrix(score))

        return x, y

    def aupr(self, score):
        """
        Area under PR curve (see function `pr`).
        """
        x, y, _ = self.pr(score)
        return auc(x,y), (x, y)

    def plot_prs(self, scores, ax):
        """
        Plot multiple PR curves (see function `pr`).
        """
        b = np.mean(self.inter)
        ax.plot([0,1], [b,b], color='gray', ls='--')
        for score in scores:
            auc, curve = self.aupr(score)
            ax.plot(*curve, label=f'Score ({auc:.2f})')
        ax.set_xlim(0,1)
        ax.set_ylim(0)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower right')

    plot_types = {
        'roc': plot_rocs,
        'pr' : plot_prs
    }

    def show_plot(self, scores, plot_type):
        fig = plt.figure(figsize=(5,5), dpi=100)
        grid = gs.GridSpec(1,1)
        ax = fig.add_subplot(grid[0,0])

        self.plot_types[plot_type](scores, ax)

        fig.show()


    def save_plot(self, scores, plot_type, file=None):
        fig = plt.figure(figsize=(5,5), dpi=100)
        grid = gs.GridSpec(1,1)
        ax = fig.add_subplot(grid[0,0])

        self.plot_types[plot_type](scores, ax)

        if file is None: 
            file = f'{plot_type}.pdf'
        fig.savefig(file, dpi=100, bbox_inches='tight', frameon=False)