from harissa_benchmark import Benchmark, available_networks
from harissa_benchmark.generators import NetworksGenerator, DatasetsGenerator, InferencesGenerator

print(available_networks())

gen = NetworksGenerator(
    # n_scores=1,
    # datasets_generator=DatasetsGenerator(path='cardamom_benchmark')
)

gen.save('test_datagen')
