from typing import (
    Dict, 
    List,
    Union, 
    Optional
)

from pathlib import Path
from time import perf_counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from alive_progress import alive_bar

from harissa import NetworkParameter, NetworkModel
from harissa_benchmark.generators import (
    GenericGenerator,
    DatasetsGenerator,
    InferencesGenerator
)

@dataclass
class ScoreInfo:
    results: List[NetworkParameter]
    runtimes: npt.NDArray[np.float_]

class Benchmark(GenericGenerator[Dict[str, ScoreInfo]]):
    def __init__(self,
        datasets_generator: Optional[DatasetsGenerator] = None,
        inferences_generator: Optional[InferencesGenerator] = None,
        n_scores: int = 10, 
        path: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__('scores', path)

        self.generators = [
            datasets_generator or DatasetsGenerator(path=path),
            inferences_generator or InferencesGenerator(path=path)
        ]
        self.model = NetworkModel()
        self.n_scores = n_scores


    # Alias
    @property  
    def scores(self):
        return self.items

    def _load(self, path: Path) -> None:
        self._items = {}
        networks = [network_dir for network_dir in path.iterdir()]
        with alive_bar(
            int(np.sum([
                len(list(inf_dir.iterdir())) 
                for network_dir in networks
                for inf_dir in network_dir.iterdir() 
            ])),
            title='Loading scores'
        ) as bar:
            for network_dir in networks:
                network_name = network_dir.stem
                self._items[network_name] = {}

                for inf_dir in network_dir.iterdir():
                        inf_name = inf_dir.stem
                        info_file_stem = 'extra_infos'
                        score_files = [
                            p for p in inf_dir.iterdir()
                            if p.stem != info_file_stem
                        ]
                        scores = [None] * len(score_files)
                        with np.load(inf_dir/f'{info_file_stem}.npz') as data:
                            extra_infos = dict(data)
                            bar()
                        
                        for i, score_file in enumerate(score_files):
                            bar.text(f'{score_file.absolute()}')
                            scores[i] = NetworkParameter.load(score_file)
                            bar()

                        self._items[network_name][inf_name] = ScoreInfo(
                            scores,
                            **extra_infos
                        )

    def _generate(self) -> None:
        self._items = {}
        with alive_bar(
            len(self.generators[1].items) 
            * int(np.sum([len(d) for d in self.generators[0].items.values()])),
            title='Generating scores'
        ) as bar:
            for net_name, datasets in self.generators[0].items.items():
                self._items[net_name] = {}
                for inf_name, inference in self.generators[1].items.items():
                    n_scores = len(datasets)
                    runtime = np.zeros(n_scores)
                    results = [None] * n_scores
                    self.model.inference = inference
                    for i in range(n_scores):
                        bar.text(f'Score {net_name}-{inf_name}-{i+1}')
                        start = perf_counter()
                        results[i] = self.model.fit(datasets[i]).parameter
                        runtime[i] = perf_counter() - start
                        bar()

                    self._items[net_name][inf_name] = ScoreInfo(
                        results, 
                        runtime
                    )
    
    def _save(self, path: Path) -> None:
        self.generators[0].save(path.parent)
        with alive_bar(
            int(np.sum([
                len(s.results) + 1
                for inf_s in self.scores.values() 
                for s in inf_s.values()
            ])),
            title='Saving Scores'
        ) as bar:
            for network, inferences_score in self.scores.items():
                network_path = path / network
                for inference, score_info  in inferences_score.items():
                    output = network_path / inference
                    output.mkdir(parents=True, exist_ok=True)
                    for i, ntw_param in enumerate(score_info.results):
                        output_score = ntw_param.save(output / f'score_{i+1}')
                        bar.text(f'{output_score.absolute()} saved')
                        bar()
                    extra_file = output / 'extra_infos.npz'
                    bar.text(f'{extra_file.absolute()}')
                    np.savez_compressed(
                        extra_file, 
                        runtimes=score_info.runtimes
                    )
                    bar()
        
        print(f'Scores saved at {path.absolute()}')
