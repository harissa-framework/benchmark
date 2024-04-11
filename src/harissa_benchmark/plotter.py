"""Some utility functions for benchmarking Harissa"""

from typing import Dict, Tuple, List, Union, Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from harissa import NetworkParameter

def _prepare_score(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    n = matrix.shape[0]
    # remove first column and the diagonal
    mask = ~np.hstack((
        np.ones((n, 1), dtype=bool), 
        np.eye(n, dtype=bool)[:, 1:]
    ))

    # assert matrix[mask].shape == (n*(n - 2) + 1,)

    return np.abs(matrix[mask])

def _prepare_inter(matrix):
    return 1.0 * (_prepare_score(matrix) > 0)

class InteractionPlotter:
    def __init__(self, network: NetworkParameter) -> None:
        self._inter = _prepare_inter(network.interaction)

    @property
    def inter(self) -> npt.NDArray[np.float_]:
        return self._inter
    
    @inter.setter
    def inter(self, inter: npt.NDArray[np.float_]):
        self._inter = _prepare_inter(inter)
    

    def roc(self, 
        score: npt.NDArray[np.float_]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Compute a receiver operating characteristic (ROC) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        x, y, _ = roc_curve(self.inter, _prepare_score(score))
        return x, y

    def auroc(self, 
        score:npt.NDArray[np.float_]
    ) -> Tuple[float, Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]:
        """
        Area under ROC curve (see function `roc`).
        """
        x, y = self.roc(score)
        return auc(x, y), (x, y) 

    def plot_roc(self,
        scores: Dict[str, npt.NDArray[np.float_]],
        ax: Optional[plt.Axes] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        show_plot = path is None and ax is None
        if ax is None:
            fig = plt.figure(figsize=(5,5), dpi=100)
            grid = gs.GridSpec(1,1)
            ax = fig.add_subplot(grid[0,0])

        for inference_name, score in scores.items():
            auc, curve = self.auroc(score)
            ax.plot(*curve, label=f'{inference_name} ({auc:.2f})')
        ax.plot([0,1], [0,1], color='gray', ls='--', label='Random (0.50)')
        # ax.set_xlim(0,1)
        # ax.set_ylim(0)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc='lower right')

        if path is not None:
            plt.gcf().savefig(
                path, 
                dpi=100, 
                bbox_inches='tight', 
                frameon=False
            )
        elif show_plot:
            plt.gcf().show()
    
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    def plot_rocs(self, 
        scores: Dict[str,Tuple[List[NetworkParameter],npt.NDArray[np.float_]]],
        ax: Optional[plt.Axes] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        show_plot = path is None and ax is None
        if ax is None:
            fig = plt.figure(figsize=(5,5), dpi=100)
            grid = gs.GridSpec(1,1)
            ax = fig.add_subplot(grid[0,0])

        for name, (networks_param, _) in scores.items():
            aucs = np.empty(len(networks_param))
            x = np.linspace(0, 1, 1000)
            ys = np.empty((len(networks_param), x.size))
            for i, network_param in enumerate(networks_param):
                aucs[i], curve = self.auroc(network_param.interaction)
                interpolated_y = np.interp(x, *curve)
                interpolated_y[0] = 0.0
                # ax.plot(x, interpolated_y, alpha=0.2, label=f'{name} ({aucs[i]:.2f})')

                ys[i] = interpolated_y
            
            y = np.mean(ys, axis=0)
            y[-1] = 1.0
            std_y = np.std(ys, axis=0)
            p = ax.plot(x, y, label=f'{name} ({np.mean(aucs):.2f} $\pm$ {np.std(aucs):.2f})')
            ax.fill_between(
                x, 
                np.maximum(y - std_y, 0.0), 
                np.minimum(y + std_y, 1.0),
                color=p[-1].get_color(),
                alpha=0.2
            )
        ax.plot([0,1], [0,1], color='gray', ls='--', label='Random (0.50)')
        # ax.set_xlim(0,1)
        # ax.set_ylim(0)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc='lower right')

        if path is not None:
            plt.gcf().savefig(
                path, 
                dpi=100, 
                bbox_inches='tight', 
                frameon=False
            )
        elif show_plot:
            plt.gcf().show()
        

    def pr(self,
        score: npt.NDArray[np.float_]
    ) ->  Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Compute a precision recall (PR) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        y, x, _ = precision_recall_curve(self.inter, _prepare_score(score))

        return np.flip(x), np.flip(y)

    def aupr(self, 
        score: npt.NDArray[np.float_]
    ) -> Tuple[float, Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]:
        """
        Area under PR curve (see function `pr`).
        """
        x, y = self.pr(score)
        return auc(x,y), (x, y)

    def plot_pr(self, 
        scores: Dict[str, npt.NDArray[np.float_]], 
        ax: Optional[plt.Axes] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        show_plot = path is None and ax is None
        if ax is None:
            fig = plt.figure(figsize=(5,5), dpi=100)
            grid = gs.GridSpec(1,1)
            ax = fig.add_subplot(grid[0,0])

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

        if path is not None:
            plt.gcf().savefig(
                path, 
                dpi=100, 
                bbox_inches='tight', 
                frameon=False
            )
        elif show_plot:
            plt.gcf().show()

    def plot_prs(self,
        scores: Dict[str,Tuple[List[NetworkParameter],npt.NDArray[np.float_]]],
        ax: Optional[plt.Axes] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot multiple PR curves (see function `pr`).
        """
        show_plot = path is None and ax is None
        if ax is None:
            fig = plt.figure(figsize=(5,5), dpi=100)
            grid = gs.GridSpec(1,1)
            ax = fig.add_subplot(grid[0,0])

        for name, (networks_param, _) in scores.items():
            aucs = np.empty(len(networks_param))
            x = np.linspace(0, 1, 1000)
            ys = np.empty((len(networks_param), x.size))
            for i, network_param in enumerate(networks_param):
                aucs[i], curve = self.aupr(network_param.interaction)
                interpolated_y = np.interp(x, *curve)
                interpolated_y[0] = 1.0
                # ax.plot(x, interpolated_y, alpha=0.2, label=f'{name} ({aucs[i]:.2f})')

                ys[i] = interpolated_y
            
            y = np.mean(ys, axis=0)
            std_y = np.std(ys, axis=0)
            p = ax.plot(x, y, label=f'{name} ({np.mean(aucs):.2f} $\pm$ {np.std(aucs):.2f})')
            ax.fill_between(
                x,
                np.maximum(y - std_y, 0.0),
                np.minimum(y + std_y, 1.0),
                color=p[-1].get_color(),
                alpha=0.2
            )

        b = np.mean(self.inter)
        ax.plot([0,1], [b,b], color='gray', ls='--', label=f'Random ({b:.2f})')
        
        ax.set_xlim(0,1)
        ax.set_ylim(0)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower right')

        if path is not None:
            plt.gcf().savefig(
                path, 
                dpi=100, 
                bbox_inches='tight', 
                frameon=False
            )
        elif show_plot:
            plt.gcf().show()