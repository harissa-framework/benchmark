from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import numpy.typing as npt

from harissa import Dataset, NetworkModel
from harissa_benchmark import spinner, alive_bar, alive_it
from harissa_benchmark.generators import NetworksGenerator

class DatasetsGenerator:

    def __init__(self, 
        path: Optional[Path] = None,
        include: Optional[List[str]] = None, 
        exclude: Optional[list[str]] = None,
    ) -> None:
        self.networks = NetworksGenerator(path).networks(include, exclude)
        self.datasets : Optional[Dict[str, List[Dataset]]] = None

    def generate(self, 
        time_points : npt.NDArray[np.float_] = np.array([
            0, 6, 12, 24, 36, 48, 60, 72, 84, 96
        ], dtype=float),
        n_cells_per_time_points: int = 100,
        burn_in_duration: float = 5.0,
        n_dataset_per_network : int = 10
    ) -> None:
        model = NetworkModel()
        self.datasets = {}

        with alive_bar(len(self.networks), spinner=spinner) as bar:
            for name, network in self.networks.items():
                model.parameter = network
                self.datasets[name] = [
                    model.simulate_dataset(
                        time_points, 
                        n_cells_per_time_points, 
                        burn_in_duration
                    ) for _ in alive_it(
                        range(n_dataset_per_network),
                        title=f'Generating {name} datasets',
                        spinner=spinner,
                        receipt=False
                    )
                ]
            bar()
    
    def save(self, path: Optional[Path] = Path.cwd()) -> Path:
        if self.datasets is not None:
            output = path.with_suffix('')
            output.mkdir(parents=True, exist_ok=True)

            with alive_bar(len(self.networks), spinner=spinner) as bar:
                for name, datasets in self.datasets:
                    output_network = output / name
                    output_network.mkdir(exist_ok=True)

                    with alive_bar(
                        len(datasets),
                        title=f'Saving {name} datasets', 
                        spinner=spinner,
                        receipt=False
                    ) as bar2:
                        for i, dataset in enumerate(datasets):
                            dataset.save(output_network / f'dataset_{i+1}')
                        bar2()
                bar()

            return output
        else:
            raise RuntimeError(('Nothing to save. '
                                'Please generate datasets before.'))

