"""Some utility functions for benchmarking Harissa"""

from typing import Dict, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

def _prepare_matrix(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    n = matrix.shape[0]
    # remove first column and the diagonal
    mask = ~np.hstack((
        np.ones((n, 1), dtype=bool), 
        np.eye(n, dtype=bool)[:, 1:]
    ))

    # assert matrix[mask].shape == (n*(n - 2) + 1,)

    return np.clip(np.abs(matrix[mask]), 0.0, 1.0)


class InteractionPlotter:
    def __init__(self, inter: npt.NDArray[np.float_]) -> None:
        self._inter = _prepare_matrix(inter)

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

    def plot_rocs(self, scores: Dict[str, npt.NDArray[np.float_]], ax) -> None:
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        for inference_name, score in scores.items():
            auc, curve = self.auroc(score)
            ax.plot(*curve, label=f'{inference_name} ({auc:.2f})')
        ax.plot([0,1], [0,1], color='gray', ls='--', label='Random (0.50)')
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
        x, y = self.pr(score)
        return auc(x,y), (x, y)

    def plot_prs(self, scores: Dict[str, npt.NDArray[np.float_]], ax) -> None:
        """
        Plot multiple PR curves (see function `pr`).
        """
        for inference_name, score in scores.items():
            auc, curve = self.aupr(score)
            ax.plot(*curve, label=f'{inference_name} ({auc:.2f})')
        b = np.mean(self.inter)
        ax.plot([0,1], [b,b], color='gray', ls='--', label=f'Random ({b:.2f})')
        ax.set_xlim(0,1)
        ax.set_ylim(0)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower right')

    def show_plot(self,
        scores: Dict[str, npt.NDArray[np.float_]], 
        plot_type: str
    ) -> None:
        fig = plt.figure(figsize=(5,5), dpi=100)
        grid = gs.GridSpec(1,1)
        ax = fig.add_subplot(grid[0,0])

        getattr(self, f'plot_{plot_type}s')(scores, ax)

        fig.show(warn=False)


    def save_plot(self, 
        scores: Dict[str, npt.NDArray[np.float_]], 
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