from harissa_benchmark import ScoresGenerator
from harissa_benchmark.generators import DatasetsGenerator

gen = ScoresGenerator(datasets_generator=DatasetsGenerator(path='datagen'))

gen.save('datagen')
