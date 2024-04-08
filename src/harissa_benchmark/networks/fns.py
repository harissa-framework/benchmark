from harissa import NetworkParameter
from harissa_benchmark.generators import NetworksGenerator

def create_fn4():
    fn = NetworkParameter(4)
    
    fn.degradation_rna[:] = 1
    fn.degradation_protein[:] = 0.2
    fn.d[:] /= 5

    scale = fn.scale()
    fn.creation_rna[:] = fn.degradation_rna * scale
    fn.creation_protein[:] = fn.degradation_protein * scale 

    fn.basal[:] = -5
    fn.interaction[0,1] = 10
    fn.interaction[1,2] = 10
    fn.interaction[1,3] = 10
    fn.interaction[3,4] = 10
    fn.interaction[4,1] = -10
    fn.interaction[2,2] = 10
    fn.interaction[3,3] = 10

    return fn

def create_fn8():
    fn = NetworkParameter(8)

    fn.degradation_rna[:] = 0.4
    fn.degradation_protein[:] = 0.08

    scale = fn.scale()
    fn.creation_rna[:] = fn.degradation_rna * scale
    fn.creation_protein[:] = fn.degradation_protein * scale

    fn.basal[:] = -5
    fn.interaction[0, 1] = 10
    fn.interaction[1, 2] = 10
    fn.interaction[2, 3] = 10
    fn.interaction[3, 4] = 10
    fn.interaction[3, 5] = 10
    fn.interaction[3, 6] = 10
    fn.interaction[4, 1] = -10
    fn.interaction[5, 1] = -10
    fn.interaction[6, 1] = -10
    fn.interaction[4, 4] = 10
    fn.interaction[5, 5] = 10
    fn.interaction[6, 6] = 10
    fn.interaction[4, 8] = -10
    fn.interaction[4, 7] = -10
    fn.interaction[6, 7] = 10
    fn.interaction[7, 6] = 10
    fn.interaction[8, 8] = 10

    return fn

NetworksGenerator.register('FN4', create_fn4)
NetworksGenerator.register('FN8', create_fn8)