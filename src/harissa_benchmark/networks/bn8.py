from harissa import NetworkParameter
from harissa_benchmark.generators import NetworksGenerator

def create_bn8():
    bn8 = NetworkParameter(8)

    bn8.degradation_rna[:] = 0.25
    bn8.degradation_protein[:] = 0.05

    scale = bn8.scale()
    bn8.creation_rna[:] = bn8.degradation_rna * scale
    bn8.creation_protein[:] = bn8.degradation_protein * scale 

    bn8.basal[:] = -4

    bn8.interaction[0, 1] = 10
    bn8.interaction[1, 2] = 10
    bn8.interaction[1, 3] = 10
    bn8.interaction[3, 2] = -10
    bn8.interaction[2, 3] = -10
    bn8.interaction[2, 2] = 5
    bn8.interaction[3, 3] = 5
    bn8.interaction[2, 4] = 10
    bn8.interaction[3, 5] = 10
    bn8.interaction[2, 5] = -10
    bn8.interaction[3, 4] = -10
    bn8.interaction[4, 7] = -10
    bn8.interaction[5, 6] = -10
    bn8.interaction[4, 6] = 10
    bn8.interaction[5, 7] = 10
    bn8.interaction[7, 8] = 10
    bn8.interaction[6, 8] = -10

    return bn8

NetworksGenerator.register('BN8', create_bn8)