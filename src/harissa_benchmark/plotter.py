"""Some utility functions for benchmarking Harissa"""

from typing import Dict, Tuple, Union, Optional
from pathlib import Path
from functools import wraps

import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from harissa import NetworkParameter
from harissa_benchmark.benchmark import ScoreInfo
from harissa_benchmark.generators import InferencesGenerator, InferenceInfo

def _plot_decorator(plot_func):
    @wraps(plot_func)
    def wrapper(self,
        scores: Dict[str, ScoreInfo],
        ax: Optional[plt.Axes] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        show_plot = path is None and ax is None
        if ax is None:
            fig = plt.figure(figsize=(5,5), dpi=100)
            grid = gs.GridSpec(1,1)
            ax = fig.add_subplot(grid[0,0])

        plot_func(self, scores, ax)

        if path is not None:
            plt.savefig(
                path, 
                bbox_inches='tight'
            )
        elif show_plot:
            plt.show()

    return wrapper

class DirectedPlotter:
    def __init__(self, 
        network: NetworkParameter, 
        alpha_curve_std: float = 0.2
        ) -> None:
        self._inter = self._prepare_inter(network.interaction)
        self.alpha_curve_std = alpha_curve_std

    @property
    def inter(self) -> npt.NDArray[np.float_]:
        return self._inter
    
    @inter.setter
    def inter(self, inter: npt.NDArray[np.float_]):
        self._inter = self._prepare_inter(inter)    

    def _prepare_score(self, 
        matrix: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        n = matrix.shape[0]
        # remove first column and the diagonal
        mask = ~np.hstack((
            np.ones((n, 1), dtype=bool), 
            np.eye(n, dtype=bool)[:, 1:]
        ))

        # assert matrix[mask].shape == (n*(n - 2) + 1,)

        return np.abs(matrix[mask])

    def _prepare_inter(self, matrix):
        return 1.0 * (self._prepare_score(matrix) > 0)
    
    def _accept_inference(self, inference_info: InferenceInfo) -> bool:
        return inference_info.is_directed_graph

    def roc(self, 
        score: npt.NDArray[np.float_]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Compute a receiver operating characteristic (ROC) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        x, y, _ = roc_curve(self.inter, self._prepare_score(score))
        return x, y

    def auroc(self, 
        score:npt.NDArray[np.float_]
    ) -> float:
        """
        Area under ROC curve (see function `roc`).
        """
        x, y = self.roc(score)
        return auc(x, y) 
    
    @_plot_decorator
    def plot_roc_curves(self, 
        scores: Dict[str, ScoreInfo],
        ax: plt.Axes
    ) -> None:
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        
        self._plot_curves(scores, self.roc, ax)
        ax.plot(
            [0,1], 
            [0,1], 
            color='lightgray',
            ls='--',
            label='Random (0.50)'
        )
        # ax.set_xlim(0,1)
        # ax.set_ylim(0)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc='lower right')

    @_plot_decorator
    def plot_roc_boxes(self,
        scores: Dict[str, ScoreInfo],
        ax: plt.Axes                   
    ) -> None:

        self._plot_boxes_auc(scores, self.auroc, ax)
        left, right = ax.get_xlim()
        ax.plot(
            [left, right], 
            [0.5,0.5], 
            color='lightgray', 
            ls='--' 
            # label=f'Random ({b:.2f})'
        )
        
        ax.set_xlim(left, right)
        ax.set_ylim(0,1)
        ax.set_ylabel('AUROC', fontsize=6)

    def pr(self,
        score: npt.NDArray[np.float_]
    ) ->  Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Compute a precision recall (PR) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        y, x, _ = precision_recall_curve(
            self.inter, 
            self._prepare_score(score)
        )

        return np.flip(x), np.flip(y)

    def aupr(self, 
        score: npt.NDArray[np.float_]
    ) -> float:
        """
        Area under PR curve (see function `pr`).
        """
        x, y = self.pr(score)
        return auc(x,y)

    @_plot_decorator
    def plot_pr_curves(self,
        scores: Dict[str, ScoreInfo],
        ax: plt.Axes
    ) -> None:
        """
        Plot multiple PR curves (see function `pr`).
        """
        self._plot_curves(scores, self.pr, ax)
        b = np.mean(self.inter)
        ax.plot(
            [0,1], 
            [b,b], 
            color='lightgray', 
            ls='--', 
            label=f'Random ({b:.2f})'
        )
        
        ax.set_xlim(0,1)
        ax.set_ylim(0)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower right')

    @_plot_decorator
    def plot_pr_boxes(self,
        scores: Dict[str, ScoreInfo],
        ax: plt.Axes                   
    ) -> None:

        self._plot_boxes_auc(scores, self.aupr, ax)
        b = np.mean(self.inter)
        left, right = ax.get_xlim()
        ax.plot(
            [left, right], 
            [b,b], 
            color='lightgray', 
            ls='--' 
            # label=f'Random ({b:.2f})'
        )
        
        ax.set_xlim(left, right)
        ax.set_ylim(0,1)
        ax.set_ylabel('AUPR', fontsize=6)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    def _plot_curves(self, scores, curve_fn, ax):
        for inf, score_info in scores.items():
            inf_info = InferencesGenerator.getInferenceInfo(inf)
            if self._accept_inference(inf_info):
                x = np.linspace(0, 1, 1000)
                ys = np.empty((score_info.results.size, x.size))
                for i, result in enumerate(score_info.results.flat):
                    curve = curve_fn(result.parameter.interaction)
                    interpolated_y = np.interp(x, *curve)
                    interpolated_y[0] = 0.0

                    ys[i] = interpolated_y
                
                y = np.mean(ys, axis=0)
                y[-1] = 1.0
                std_y = np.std(ys, axis=0)
                ax.plot(
                    x, 
                    y,
                    color=inf_info.colors[0], 
                    label=f'{inf} ({auc(x, y):.2f})'
                )
                ax.fill_between(
                    x, 
                    np.maximum(y - std_y, 0.0), 
                    np.minimum(y + std_y, 1.0),
                    color=inf_info.colors[1],
                    alpha=self.alpha_curve_std
                )

    def _plot_boxes_auc(self, scores, auc_fn, ax):
        inferences_name = []
        for i, (inf, score_info) in enumerate(scores.items()):
            inf_info = InferencesGenerator.getInferenceInfo(inf)
            if self._accept_inference(inf_info):
                inferences_name.append(inf)
                aucs = np.empty(score_info.results.size)
                for j, result in enumerate(score_info.results.flat):
                    aucs[j] = auc_fn(result.parameter.interaction)
                
                box = ax.boxplot(
                    [aucs], 
                    positions=[i+0.5], 
                    patch_artist= True,
                    widths= [.25]
                )

                w = 0.8
                for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(box[item], color=inf_info.colors[0], lw=w)
                plt.setp(box['boxes'], facecolor=inf_info.colors[1])
                plt.setp(
                    box['fliers'], 
                    markeredgecolor=inf_info.colors[0], 
                    ms=3, 
                    markerfacecolor=inf_info.colors[1],
                    markeredgewidth=w
                )
        ax.set_xticklabels(inferences_name)
        w = 0.7
        ax.tick_params(direction='out', length=3, width=w)
        ax.tick_params(axis='x', pad=2, labelsize=5.5)
        ax.tick_params(axis='y', pad=0.5, labelsize=5.5)
        for x in ['top','bottom','left','right']:
            ax.spines[x].set_linewidth(w)


class UnDirectedPlotter(DirectedPlotter):
    def __init__(self, network: NetworkParameter) -> None:
        super().__init__(network)

    def _prepare_score(self, 
        matrix: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        abs_matrix = np.abs(matrix)
        # remove lower triangle
        mask = ~np.tri(*abs_matrix.shape, dtype=bool)

        return np.maximum(abs_matrix[mask], abs_matrix.T[mask])

    def _accept_inference(self, inference_info: InferenceInfo) -> bool:
        return True