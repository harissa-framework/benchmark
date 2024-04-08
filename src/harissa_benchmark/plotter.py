"""Some utility functions for benchmarking Harissa"""

from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from harissa import NetworkParameter

def _prepare_matrix(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    n = matrix.shape[0]
    # remove first column and the diagonal
    mask = ~np.hstack((
        np.ones((n, 1), dtype=bool), 
        np.eye(n, dtype=bool)[:, 1:]
    ))

    # assert matrix[mask].shape == (n*(n - 2) + 1,)

    return np.abs(matrix[mask])


class InteractionPlotter:
    def __init__(self, network_parameter: NetworkParameter) -> None:
        self._inter = _prepare_matrix(network_parameter.interaction)

    @property
    def inter(self) -> npt.NDArray[np.float_]:
        return self._inter
    
    @inter.setter
    def inter(self, inter: npt.NDArray[np.float_]):
        self._inter = _prepare_matrix(inter)
    

    def roc(self, 
        score: npt.NDArray[np.float_]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Compute a receiver operating characteristic (ROC) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        x, y, _ = roc_curve(self.inter, _prepare_matrix(score))
        return x, y

    def auroc(self, 
        score:npt.NDArray[np.float_]
    ) -> Tuple[float, Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]:
        """
        Area under ROC curve (see function `roc`).
        """
        x, y = self.roc(score)
        return auc(x, y), (x, y) 

    def plot_rocs(self, scores: list[NetworkParameter], ax) -> None:
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        ax.plot([0,1], [0,1], color='gray', ls='--', label='Random (0.50)')
        for score in scores:
            auc, curve = self.auroc(score.interaction)
            ax.plot(*curve, label=f'Score ({auc:.2f})')
        # ax.set_xlim(0,1)
        # ax.set_ylim(0)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc='lower right')

    def pr(self,
        score: npt.NDArray[np.float_]
    ) ->  Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Compute a precision recall (PR) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        y, x, _ = precision_recall_curve(self.inter, _prepare_matrix(score))

        return x, y

    def aupr(self, 
        score: npt.NDArray[np.float_]
    ) -> Tuple[float, Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]:
        """
        Area under PR curve (see function `pr`).
        """
        x, y, _ = self.pr(score)
        return auc(x,y), (x, y)

    def plot_prs(self, scores: List[NetworkParameter], ax) -> None:
        """
        Plot multiple PR curves (see function `pr`).
        """
        b = np.mean(self.inter)
        ax.plot([0,1], [b,b], color='gray', ls='--')
        for score in scores:
            auc, curve = self.aupr(score.interaction)
            ax.plot(*curve, label=f'Score ({auc:.2f})')
        ax.set_xlim(0,1)
        ax.set_ylim(0)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower right')

    def show_plot(self,
        scores: List[NetworkParameter], 
        plot_type: str
    ) -> None:
        fig = plt.figure(figsize=(5,5), dpi=100)
        grid = gs.GridSpec(1,1)
        ax = fig.add_subplot(grid[0,0])

        getattr(self, f'plot_{plot_type}s')(scores, ax)

        fig.show()


    def save_plot(self, 
        scores: List[NetworkParameter], 
        plot_type: str,
        file: Optional[Union[str, Path]] = None
    ) -> None:
        fig = plt.figure(figsize=(5,5), dpi=100)
        grid = gs.GridSpec(1,1)
        ax = fig.add_subplot(grid[0,0])

        getattr(self, f'plot_{plot_type}s')(scores, ax)

        if file is None: 
            file = f'{plot_type}.pdf'
        fig.savefig(file, dpi=100, bbox_inches='tight', frameon=False)