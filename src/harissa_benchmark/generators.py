from typing import (
    Dict, 
    List, 
    Tuple, 
    Callable, 
    Generic, 
    TypeVar, 
    Union, 
    Optional
)

from dataclasses import dataclass
from pathlib import Path
from alive_progress import alive_bar

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import matplotlib

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
            try:
                self._generate()
            except BaseException as e:
                self._items = None
                raise e


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
        paths = [
            p for p in path.iterdir() 
            if p.stem in self._include and p.stem not in self._exclude
        ]
        with alive_bar(len(paths), title='Loading Networks parameters') as bar:
            for p in path.iterdir():
                bar.text(f'Loading {p.absolute()}')
                self._items[p.stem] = NetworkParameter.load(p)
                bar()
        
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

@dataclass
class InferenceInfo:
    inference: Union[Inference, Callable[[], Inference]]
    is_directed_graph: bool
    colors: npt.NDArray

class InferencesGenerator(GenericGenerator[Inference]):
    _inferences : Dict[str, InferenceInfo] = {}
    color_map: matplotlib.colors.Colormap = matplotlib.pyplot.get_cmap('tab20')

    def __init__(self,
        path: Optional[Union[str, Path]] = None, 
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None
    ) -> None:
        self._include = include or list(self._inferences.keys())
        self._exclude = exclude or []

        super().__init__('inferences', path)

    @classmethod
    def register(cls, name: str, inference_info: InferenceInfo) -> None:
        if name not in cls._inferences:
            if isinstance(inference_info.inference, (Inference, Callable)):
                cls._inferences[name] = inference_info
            else:
                raise TypeError(('inference_callable must be a callable '
                             'that returns a Inference sub class.'))
        else:
            raise ValueError((f'{name} is already taken. '
                              f'Cannot register {inference_info}.'))

    # Alias
    @property
    def inferences(self):
        return self.items

    @classmethod
    def available_inferences(cls) -> List[str]:
        return list(cls._inferences.keys())
    
    @classmethod
    def getInferenceInfo(cls, name: str) -> InferenceInfo:
        return cls._inferences[name]
    
    def _load(self, path: Path) -> None:
        with alive_bar(
            len(list(path.iterdir())),
            title='Loading inferences info'
        ) as bar:
            for p in path.iterdir():
                name = p.stem
                if name not in self._include and name not in self._include:
                    self._include.append(name)
                    if name not in self._inferences:
                        with np.load(p, allow_pickle=True) as data:
                            self.register(name, InferenceInfo(
                                data['inference'].item(),
                                data['is_directed_graph'].item(),
                                data['colors']
                            ))
                bar()
        self.generate(force_generation=True)


    def _generate(self) -> None:
        self._items = {}
        for name, inf_info in self._inferences.items():
            if name in self._include and name not in self._exclude:
                if isinstance(inf_info.inference, Inference):
                    inference = inf_info.inference
                else:
                    inference = inf_info.inference()
                
                if isinstance(inference, Inference):
                    self._items[name] = inference
                else:
                    raise RuntimeError(
                        (f'{inf_info.inference} is not a callable'
                          ' that returns a Inference sub class.')
                    )
                
    def _save(self, path: Path) -> None:
        with alive_bar(
            len(self.inferences), 
            title='Saving Inferences Info'
        ) as bar:
            for inf_name in self.inferences:
                output = (path / inf_name).with_suffix('.npz')
                bar.text(f'{output.absolute()}')
                info = self._inferences[inf_name]
                np.savez_compressed(
                    output,
                    inference=np.array(info.inference),
                    is_directed_graph=np.array(info.is_directed_graph),
                    colors=info.colors
                ) 

    
class DatasetsGenerator(GenericGenerator[List[Dataset]]):
    def __init__(self, 
        networks_generator: Optional[NetworksGenerator] = None,
        time_points : npt.NDArray[np.float_] = np.array([
            0, 6, 12, 24, 36, 48, 60, 72, 84, 96
        ], dtype=float),
        n_cells: int = 100,
        burn_in_duration: float = 5.0,
        n_datasets : int = 10,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__('datasets', path)
            
        self.networks_generator = (
            networks_generator or NetworksGenerator(path=path)
        )
    
        self.model = NetworkModel(simulation=BurstyPDMP(
            use_numba=True
        ))
        self.simulate_dataset_parameters = {
            'time_points': time_points, 
            'n_cells' : n_cells,
            'burn_in_duration': burn_in_duration
        }
        self.n_datasets = n_datasets

    # Alias
    @property
    def datasets(self) -> Dict[str, List[Dataset]]:
        return self.items
    
    def _load(self, path: Path) -> None:
        self._items = {}
        with alive_bar(
            int(np.sum([len(list(p.iterdir())) for p in path.iterdir()])),
            title='Loading datasets'
        ) as bar:
            for network_dir in path.iterdir():
                name = network_dir.stem
                self._items[name] = [None] * len(list(network_dir.iterdir()))
                for i, dataset_file in enumerate(network_dir.iterdir()):
                    bar.text(f'Loading {name} datasets {i}')
                    self._items[name][i] = Dataset.load(dataset_file) 
                    bar()

    def _generate(self) -> None:
        self._items = {}
        with alive_bar(
            len(self.networks_generator.networks) * self.n_datasets,
            title='Generating datasets'
        ) as bar:
            for name, network in self.networks_generator.networks.items():
                self.model.parameter = network
                self._items[name] = [None] * self.n_datasets
                for i in range(self.n_datasets):
                    bar.text(f'Generating {name} - dataset {i+1}')
                    self._items[name][i] = self.model.simulate_dataset(
                        **self.simulate_dataset_parameters
                    )
                    bar()
    
    def _save(self, path: Path) -> None:
        self.networks_generator.save(path.parent)
        with alive_bar(
            int(np.sum([len(d) for d in self.datasets.values()])),
            title='Saving datasets'
        ) as bar:
            for name, datasets in self.datasets.items():
                output = path / name
                output.mkdir(parents=True, exist_ok=True)

                for i, dataset in enumerate(datasets):
                    output_dataset = output / f'dataset_{i + 1}'
                    bar.text(f'Saving {output_dataset.absolute()}')
                    dataset.save(output_dataset)
                    bar()

        print(f'Datasets saved at {path.absolute()}')

