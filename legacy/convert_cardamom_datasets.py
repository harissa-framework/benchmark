from pathlib import Path
import numpy as np

from harissa import Dataset
from harissa_benchmark.generators import NetworksGenerator
from harissa_benchmark.networks.trees import create_tree
from alive_progress import alive_bar

cur_file_dir = Path(__file__).parent
cardamom_article = cur_file_dir / 'results_cardamom_article'
output_dir = cur_file_dir.parent / 'cardamom_benchmark'

deterministic_networks = ['BN8', 'CN5', 'FN4', 'FN8']
network_gen = NetworksGenerator(include=deterministic_networks)
network_gen.save(output_dir)

with alive_bar(len(list(cardamom_article.iterdir()))) as bar:
    for folder in cardamom_article.iterdir():
        folder_name = folder.name
        old_datasets = [
            p for p in (folder / 'Data').iterdir() if p.suffix == '.txt'
        ]
        datasets_output = output_dir / 'datasets' / folder_name
        
        if folder_name in deterministic_networks:
            # save datasets
            datasets = np.empty(len(old_datasets), dtype=object)
            for i, path in enumerate(old_datasets):
                old_dataset = np.loadtxt(path, dtype=int, delimiter='\t')
                datasets[i] = Dataset(
                    old_dataset[0, 1:].astype(np.float_), 
                    old_dataset[1:, 1:].T.astype(np.uint)
                )
            datasets_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(datasets_output.with_suffix('.npy'), datasets)
        else:
            assert folder_name.startswith('Trees')

            inters = [
                p for p in (folder / 'True').iterdir() if p.suffix == '.npy'
            ]
            networks_output = output_dir / 'networks' / folder_name
            # save networks interaction
            for path in inters:
                tree_name = f'{path.stem.split("_")[1]}.npz'
                inter = np.load(path)
                network = create_tree(inter.shape[1] - 1)
                # override interaction matrix
                network.interaction[:] = inter
                network.save(networks_output / tree_name)
            # save datasets
            for path in old_datasets:
                old_dataset = np.loadtxt(path, dtype=int, delimiter='\t')
                datasets = np.array([
                    Dataset(
                        old_dataset[0, 1:].astype(np.float_), 
                        old_dataset[1:, 1:].T.astype(np.uint)
                    )
                ], dtype=object)
                output = datasets_output / f'{path.stem.split("_")[1]}.npy'
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, datasets)
        bar()

print(f'Datasets saved at {(output_dir / "datasets").absolute()}')