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

from harissa_benchmark.utils import match_rec

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
            * len(self.generators[1].items) 
            * int(np.sum([d.size for d in self.generators[0].items.values()])),
            title='Generating scores'
        ) as bar:
            for net_name, datasets in self.generators[0].items.items():
                self._items[net_name] = {}
                for inf_name, inference in self.generators[1].items.items():
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
        
        print(f'Scores saved at {path.absolute()}')
