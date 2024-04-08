from harissa_benchmark.generators import NetworksGenerator
from harissa_benchmark.utils import collect_modules

__all__ = ['available_networks']

collect_modules(__file__, __all__)

available_networks = NetworksGenerator.available_networks