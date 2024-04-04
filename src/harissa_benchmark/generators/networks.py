from typing import Dict, List, Optional
from pathlib import Path
from harissa import NetworkParameter

class NetworksGenerator:

    _networks : Dict[str, NetworkParameter] = {}

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is not None and not path.is_dir():
            raise ValueError('path must be an existing directory.')
        
        self.path = path

    @classmethod
    def register_networks(cls, name: str, network: NetworkParameter):
        if isinstance(network, NetworkParameter):
            if name not in cls._networks:
                cls._networks[name] = network
            else:
                raise ValueError((f'{name} is already taken. '
                                  f'Cannot register {network}.'))
        else:
            raise TypeError('network must be a NetworkParameter object.')
        
    def networks(self, 
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None
    ) -> Dict[str, NetworkParameter]:
        if include is None:
            include = list(self._networks.keys())
        if exclude is None:
            exclude = []
        
        networks = self._networks
        if self.path is not None:
            networks = {}
            for p in self.path.iterdir():
                networks[p.stem] = NetworkParameter.load(p)

        filtered_networks = {}
        for name, network in self._networks.items():
            if name in include and name not in exclude:
                filtered_networks[name] = network 

        return filtered_networks
        
    def save(self, path: Path = Path.cwd()) -> Path:
        output = path.with_suffix('')
        output.mkdir(parents=True, exist_ok=True)

        for name, network in self._networks.items():
            network.save(output / name)

        return output