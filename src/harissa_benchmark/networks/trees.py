from harissa.networks import random_tree
from harissa_benchmark.generators import NetworksGenerator

def create_tree(n_genes):
    tree = random_tree(n_genes)

    tree.degradation_rna[:] = 1
    tree.degradation_protein[:] = 0.2
    tree.d[:] /= 4

    scale = tree.scale()
    tree.creation_rna[:] = tree.degradation_rna[:] * scale
    tree.creation_protein[:] = tree.degradation_protein[:] * scale

    return tree


for n in [5, 10, 20, 50, 100]:
    NetworksGenerator.register(f'Trees{n}', lambda: create_tree(n))