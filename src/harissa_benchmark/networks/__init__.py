from harissa_benchmark.generators import NetworksGenerator
from harissa_benchmark.utils import collect_submodules

__all__ = ['available_networks']

collect_submodules(__file__, __all__)

available_networks = NetworksGenerator.available_networks