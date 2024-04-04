from pathlib import Path
from importlib import import_module

__all__ = []

for path in Path(__file__).parent.iterdir():
    network_module = path.stem
    if not network_module.startswith('_'):
        import_module(f'harissa_benchmark.networks.{network_module}')
        __all__.append(network_module)

del globals()['Path']
del globals()['import_module']
del globals()['path']
del globals()['network_module']


