from harissa_benchmark import Benchmark
from harissa_benchmark.generators import DatasetsGenerator

gen = Benchmark(datasets_generator=DatasetsGenerator(path='datagen'))

gen.save('datagen')
