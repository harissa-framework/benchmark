from typing import Dict, List, Callable, Generic, TypeVar, Union, Optional
from pathlib import Path
from alive_progress import alive_bar, alive_it

import numpy as np
import numpy.typing as npt

from harissa import Dataset, NetworkParameter, NetworkModel
from harissa.core import Inference
from harissa.simulation import BurstyPDMP

T = TypeVar('T')

class GenericGenerator(Generic[T]):
    def __init__(self,
        sub_directory_name: str, 
        path: Optional[Union[str, Path]] = None
    ) -> None:
        self._items : Optional[Dict[str, T]] = None
        self.sub_directory_name = sub_directory_name
        
        if path is not None:
            self.load(path)
    
    @property
    def items(self) -> Dict[str, T]:
        self.generate()
        return self._items

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path) / self.sub_directory_name
        if not path.is_dir():
            raise ValueError(f'{path} must be an existing directory.')
        
        self._load(path)
    
    def generate(self, force_generation: bool = False) -> None:
        if self._items is None or force_generation:
            self._generate()

    def save(self, path: Union[str, Path] = Path.cwd()) -> Path:
        self.generate()

        output = Path(path).with_suffix('') / self.sub_directory_name
        output.mkdir(parents=True, exist_ok=True)
        self._save(output)

        return output
    
    def _load(self, path: Path) -> None:
        raise NotImplementedError
    
    def _generate(self) -> None:
        raise NotImplementedError

    def _save(self, path: Path) -> None:
        raise NotImplementedError

class NetworksGenerator(GenericGenerator[NetworkParameter]):

    _networks : Dict[str, Callable[[], NetworkParameter]] = {}

    def __init__(self,
        path: Optional[Union[str, Path]] = None, 
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None
    ) -> None:
        self._include = include or list(self._networks.keys())
        self._exclude = exclude or []

        super().__init__('networks', path)

    @classmethod
    def register(cls, 
        name: str, 
        network_callable: Callable[[], NetworkParameter]
    ) -> None:
        if isinstance(network_callable, Callable):
            if name not in cls._networks:
                cls._networks[name] = network_callable
            else:
                raise ValueError((f'{name} is already taken. '
                                  f'Cannot register {network_callable}.'))
        else:
            raise TypeError(('network_callable must be a callable '
                             'that returns a NetworkParameter.'))
    
    # Alias
    @property
    def networks(self) -> Dict[str, NetworkParameter]:
        return self.items

    @classmethod
    def available_networks(cls) -> List[str]:
        return list(cls._networks.keys())

    def _load(self, path: Path) -> None:
        self._items = {}
        for p in path.iterdir():
            name = p.stem
            if name in self._include and name not in self._exclude:
                self._items[name] = NetworkParameter.load(p)
        
    def _generate(self) -> None:
        self._items = {}
        for name, network_callable in self._networks.items():
            if name in self._include and name not in self._exclude:
                network = network_callable()
                if isinstance(network, NetworkParameter):
                    self._items[name] = network
                else:
                    raise RuntimeError((f'{network_callable} is not a callable'
                                         ' that returns a NetworkParameter.'))
        
    def _save(self, path: Path) -> None:
        with alive_bar(len(self.networks)) as bar:
            for name, network in self.networks.items():
                network.save(path / name)
            bar()
        print(f'Networks saved at {path.absolute()}')
        
    
class DatasetsGenerator(GenericGenerator[List[Dataset]]):
    def __init__(self, 
        networks_generator: Optional[NetworksGenerator] = None,
        time_points : npt.NDArray[np.float_] = np.array([
            0, 6, 12, 24, 36, 48, 60, 72, 84, 96
        ], dtype=float),
        n_cells: int = 100,
        burn_in_duration: float = 5.0,
        n_dataset_per_network : int = 10,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__('datasets', path)
            
        self.networks_generator = networks_generator or NetworksGenerator(path)
    
        self.model = NetworkModel(simulation=BurstyPDMP(
            # use_numba=True
        ))
        self.simulate_dataset_parameters = {
            'time_points': time_points, 
            'n_cells' : n_cells,
            'burn_in_duration': burn_in_duration
        }
        self.n_dataset_per_network = n_dataset_per_network

    # Alias
    @property
    def datasets(self) -> Dict[str, List[Dataset]]:
        return self.items
    
    def _load(self, path: Path) -> None:
        self._items = {}
        for network_dir in path.iterdir():
            name = network_dir.stem
            self._items[name] = [
                Dataset.load(dataset_file)
                for dataset_file in alive_it(
                    network_dir.iterdir(),
                    title=f'Loading {name} datasets'
                )
            ]

    def _generate(self) -> None:
        self._items = {}
        for name, network in self.networks_generator.networks.items():
            self.model.parameter = network
            self._items[name] = [
                self.model.simulate_dataset(**self.simulate_dataset_parameters) 
                for _ in alive_it(
                    range(self.n_dataset_per_network),
                    title=f'Generating {name} datasets'
                )
            ]
    
    def _save(self, path: Path) -> None:
        self.networks_generator.save(path.parent)
        for name, datasets in self.datasets.items():
            output = path / name
            output.mkdir(parents=True, exist_ok=True)

            with alive_bar(
                len(datasets),
                title=f'Saving {name} datasets', 
            ) as bar:
                for i, dataset in enumerate(datasets):
                    dataset.save(output / f'dataset_{i + 1}')
                bar()

        print(f'Datasets saved at {path.absolute()}')

class InferencesGenerator(GenericGenerator[Inference]):
    _inferences : Dict[str, Callable[[], Inference]] = {}

    def __init__(self,
        # path: Optional[Union[str, Path]] = None, 
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None
    ) -> None:
        self._include = include or list(self._inferences.keys())
        self._exclude = exclude or []

        super().__init__('networks',
            # path
        )

    @classmethod
    def register(cls, 
        name: str, 
        inference_callable: Callable[[], Inference]
    ) -> None:
        if isinstance(inference_callable, Callable):
            if name not in cls._inferences:
                cls._inferences[name] = inference_callable
            else:
                raise ValueError((f'{name} is already taken. '
                                  f'Cannot register {inference_callable}.'))
        else:
            raise TypeError(('inference_callable must be a callable '
                             'that returns a Inference sub class.'))

    # Alias
    @property
    def inferences(self):
        return self.items

    @classmethod
    def available_inferences(cls) -> List[str]:
        return list(cls._inferences.keys())
    
    def _generate(self) -> None:
        self._items = {}
        for name, inference_callable in self._networks.items():
            if name in self._include and name not in self._exclude:
                inference = inference_callable()
                if isinstance(inference, Inference):
                    self._items[name] = inference
                else:
                    raise RuntimeError(
                        (f'{inference_callable} is not a callable'
                          ' that returns a Inference sub class.')
                    )
                
    

