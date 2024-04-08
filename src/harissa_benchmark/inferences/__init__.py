from harissa_benchmark.generators import InferencesGenerator
from harissa_benchmark.utils import collect_submodules

__all__ = ['available_inferences']

collect_submodules(__file__, __all__)

available_inferences = InferencesGenerator.available_inferences