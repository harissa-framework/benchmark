from harissa.inference import Hartree
from harissa_benchmark.generators import InferencesGenerator

InferencesGenerator.register('Hartree', Hartree)
