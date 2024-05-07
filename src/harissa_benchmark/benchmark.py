from typing import (
    Dict,
    Union, 
    Optional
)

from pathlib import Path
from time import perf_counter
from dataclasses import dataclass, asdict

import numpy as np
import numpy.typing as npt

from alive_progress import alive_bar

from harissa.core import NetworkModel, Inference
from harissa_benchmark.generators import (
    GenericGenerator,
    DatasetsGenerator,
    InferencesGenerator
)

from harissa.plot import build_pos
from harissa_benchmark.utils import match_rec
from harissa_benchmark.plotters import plot_all_directed, plot_all_undirected

@dataclass
class ScoreInfo:
    results: npt.NDArray[Inference.Result]
    runtimes: npt.NDArray[np.float_]

class Benchmark(GenericGenerator[Dict[str, ScoreInfo]]):
    def __init__(self,
        datasets_generator: Optional[DatasetsGenerator] = None,
        inferences_generator: Optional[InferencesGenerator] = None,
        n_scores: int = 10, 
        path: Optional[Union[str, Path]] = None
    ) -> None:
        self.generators = [
            datasets_generator or DatasetsGenerator(path=path),
            inferences_generator or InferencesGenerator(path=path)
        ]

        super().__init__('scores', path)
        self.model = NetworkModel()
        self.n_scores = n_scores


    # Alias
    @property  
    def scores(self):
        return self.items
    
    @property
    def networks(self):
        return self.generators[0].networks_generator.networks
    
    @property
    def datasets(self):
        return self.generators[0].datasets
    
    @property
    def inferences(self):
        return self.generators[1].inferences

    def _load(self, path: Path) -> None:
        self._items = {}
        paths = match_rec(path)
        with alive_bar(len(paths), title='Loading scores') as bar:
            for p in paths:
                inf_name = p.stem
                for part in p.parts[-2::-1]:
                    if inf_name in InferencesGenerator.available_inferences():
                        break
                    inf_name = str(Path(part) / inf_name)
                
                network_name = str(Path().joinpath(
                    *p.relative_to(path).parts[:-len(Path(inf_name).parts)]
                ))
                
                if network_name not in self._items:
                    self._items[network_name] = {}                

                with np.load(p, allow_pickle=True) as data:
                    self._items[network_name][inf_name] = ScoreInfo(**data)

                bar()

    def _generate(self) -> None:
        self._items = {}
        with alive_bar(
            self.n_scores
            * len(self.inferences) 
            * int(np.sum([d.size for d in self.datasets.values()])),
            title='Generating scores'
        ) as bar:
            for net_name, datasets in self.datasets.items():
                self._items[net_name] = {}
                for inf_name, inference in self.inferences.items():
                    n_dataset = datasets.size
                    runtime= np.zeros((n_dataset, self.n_scores))
                    results= np.empty((n_dataset, self.n_scores), dtype=object)
                    # results = [None] * n_scores
                    self.model.inference = inference
                    for i in range(n_dataset):
                        for j in range(self.n_scores):
                            text = f'Score {net_name}-{inf_name}-{i+1}'
                            if self.n_scores > 1:
                                text += f'-{j+1}'
                            bar.text(text)
                            start = perf_counter()
                            results[i, j] = self.model.fit(datasets[i])
                            runtime[i, j] = perf_counter() - start
                            bar()

                    self._items[net_name][inf_name] = ScoreInfo(
                        results, 
                        runtime
                    )
    
    def _save(self, path: Path) -> None:
        for generator in self.generators:
            generator.save(path.parent)
        
        with alive_bar(
            int(np.sum([
                len(infos)
                for infos in self.scores.values()
            ])),
            title='Saving Scores'
        ) as bar:
            for network, inferences_scores in self.scores.items():
                network_path = path / network
                for inference, score_info  in inferences_scores.items():
                    output = network_path / inference
                    output.parent.mkdir(parents=True, exist_ok=True)
                    bar.text(f'{output.with_suffix(".npz").absolute()}')

                    np.savez_compressed(output, **asdict(score_info))
                    bar()

    def reports(self, show_networks=False):
        networks_pos = [
            build_pos(n.interaction) for n in self.networks.values()
        ]
        return [
            plot_all_directed(
                self.networks, 
                self.scores, 
                networks_pos, 
                show_networks
            ), 
            plot_all_undirected(
                self.networks, 
                self.scores, 
                networks_pos, 
                show_networks
            )
        ]

    def save_reports(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        for fname, fig in zip(
            ['directed.pdf', 'undirected.pdf'], self.reports()
        ):
            fig.savefig(path / fname)
        return path