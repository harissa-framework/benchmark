import numpy as np
from scipy import stats

from harissa.core import Inference, NetworkParameter, Dataset
from harissa_benchmark.generators import InferencesGenerator, InferenceInfo

class Pearson(Inference):
    def run(self, dataset: Dataset) -> Inference.Result:
        n_gene_stim = dataset.count_matrix.shape[1]
        score = np.zeros((n_gene_stim, n_gene_stim))

        for i in range(n_gene_stim):
            for j in range(n_gene_stim):
                score[i, j] = stats.pearsonr(
                    dataset.count_matrix[:, i],
                    dataset.count_matrix[:, j]
                )[0]

        param = NetworkParameter(n_gene_stim - 1)
        param.interaction[:] = score

        return Inference.Result(param)
    

InferencesGenerator.register(
    'Pearson', 
    InferenceInfo(
        Pearson,
        False,
        np.array([InferencesGenerator.color_map(14),
                  InferencesGenerator.color_map(15)])
    )
)