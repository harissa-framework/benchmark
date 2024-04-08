from harissa import NetworkParameter
from harissa_benchmark.generators import NetworksGenerator

def create_cn5():
    cn5 = NetworkParameter(5)

    cn5.degradation_rna[:] = 0.5
    cn5.degradation_protein[:] = 0.1

    scale = cn5.scale()
    cn5.creation_rna[:] = cn5.degradation_rna * scale
    cn5.creation_protein[:] = cn5.degradation_protein * scale

    cn5.basal[1:] = [-5, 4, 4, -5, -5]
    cn5.interaction[0, 1] = 10
    cn5.interaction[1, 2] = -10
    cn5.interaction[2, 3] = -10
    cn5.interaction[3, 4] = 10
    cn5.interaction[4, 5] = 10
    cn5.interaction[5, 1] = -10

    return cn5

NetworksGenerator.register('CN5', create_cn5)
