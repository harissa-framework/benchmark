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
from harissa_benchmark.utils import match, match_rec

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
        
        try:
            self._load(path)
        except BaseException as e:
            self._items = None
            raise e
    
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
        self._include = include or ['**']
        self._exclude = exclude or []

        super().__init__('networks', path)

    @classmethod
    def register(cls, 
        name: str, 
        network: Union[NetworkParameter, Callable[[], NetworkParameter]]
    ) -> None:
        if isinstance(network, (NetworkParameter, Callable)):
            if name not in cls._networks:
                cls._networks[name] = network
            else:
                raise ValueError((f'{name} is already taken. '
                                  f'Cannot register {network}.'))
        else:
            raise TypeError(('network must be a NetworkParameter or a '
                             'callable that returns a NetworkParameter.'))
    
    # Alias
    @property
    def networks(self) -> Dict[str, NetworkParameter]:
        return self.items

    @classmethod
    def available_networks(cls) -> List[str]:
        return list(cls._networks.keys())

    def _load(self, path: Path) -> None:
        self._items = {}
        
        paths = match_rec(path, self._include, self._exclude)

        with alive_bar(len(paths), title='Loading Networks parameters') as bar:
            for p in paths:
                bar.text(f'Loading {p.absolute()}')
                name = str(p.relative_to(path).with_suffix(''))
                self._items[name] = NetworkParameter.load(p)
                bar()
        
    def _generate(self) -> None:
        self._items = {}
        networks = {
            k:n for k,n  in self._networks.items() 
            if match(k, self._include, self._exclude) 
        }
        with alive_bar(len(networks), title='Generating networks') as bar:
            for name, network in networks.items():
                if isinstance(network, Callable):
                    network = network()
                if isinstance(network, NetworkParameter):
                    self._items[name] = network
                else:
                    raise RuntimeError((f'{network} is not a callable'
                                        ' that returns a NetworkParameter.'))
                bar()
        
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
        self._include = include or ['**']
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
    
    def _generate_inference(self, name, inf_info):
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
    
    def _load(self, path: Path) -> None:
        self._items = {}
        paths = match_rec(path, self._include, self._exclude)
        with alive_bar(
            len(paths),
            title='Loading inferences info'
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                name = str(p.relative_to(path).with_suffix(''))
                with np.load(p, allow_pickle=True) as data:
                    inf_info = InferenceInfo(
                            data['inference'].item(),
                            data['is_directed_graph'].item(),
                            data['colors']
                        )
                    if name not in self._inferences:
                        self.register(name, inf_info)
                    
                    self._generate_inference(name, inf_info)
                
                bar()


    def _generate(self) -> None:
        self._items = {}
        inferences = {
            k:i for k,i in self._inferences.items()
            if match(k, self._include, self._exclude)
        }
        with alive_bar(len(inferences), title='Generating inferences') as bar:
            for name, inf_info in inferences.items():
                bar.text(f'{name}')
                self._generate_inference(name, inf_info)
                bar()
    
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

        print(f'Inferences saved at {path.absolute()}')     

    
class DatasetsGenerator(GenericGenerator[npt.NDArray[Dataset]]):
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
    def datasets(self) -> Dict[str, npt.NDArray[Dataset]]:
        return self.items
    
    def _load(self, path: Path) -> None:
        self._items = {}
        paths = match_rec(path, suffix='.npy')
        with alive_bar(
            len(paths),
            title='Loading datasets'
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                name = str(p.relative_to(path).with_suffix(''))
                
                self._items[name] = np.load(p, allow_pickle=True)
                bar()

    def _generate(self) -> None:
        self._items = {}
        with alive_bar(
            len(self.networks_generator.networks) * self.n_datasets,
            title='Generating datasets'
        ) as bar:
            for name, network in self.networks_generator.networks.items():
                self.model.parameter = network
                self._items[name] = np.empty(self.n_datasets, dtype=object)
                for i in range(self.n_datasets):
                    bar.text(f'Generating {name} - dataset {i+1}')
                    self._items[name][i] = self.model.simulate_dataset(
                        **self.simulate_dataset_parameters
                    )
                    bar()
    
    def _save(self, path: Path) -> None:
        self.networks_generator.save(path.parent)
        with alive_bar(len(self.datasets), title='Saving datasets') as bar:
            for name, datasets in self.datasets.items():
                output = path / name
                output.parent.mkdir(parents=True, exist_ok=True)

                bar.text(f'{output.absolute()}')
                np.save(output, datasets)
                bar()

        print(f'Datasets saved at {path.absolute()}')

