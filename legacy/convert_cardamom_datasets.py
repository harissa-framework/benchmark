from pathlib import Path
import numpy as np
import argparse as ap
from shutil import make_archive
from tempfile import TemporaryDirectory
from alive_progress import alive_bar

from harissa import Dataset
from harissa_benchmark.generators import NetworksGenerator
from harissa_benchmark.networks.trees import create_tree

def convert(cardamom_article, output_dir):
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
                    p for p in (folder / 'True').iterdir() if p.suffix=='.npy'
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

def archive(cardamom_article, output_dir, archive_format):
    with TemporaryDirectory() as tmp_dir:
        convert(cardamom_article, Path(tmp_dir))
        with alive_bar(title='Archiving', monitor=False, stats=False) as bar:
            make_archive(str(output_dir), archive_format, tmp_dir)
            bar()

def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-o', '--output',
        type=Path
    )
    parser.add_argument(
        '-f', '--format',
        choices=('zip', 'tar', 'gztar'),
    )
    args = parser.parse_args()

    cur_file_dir = Path(__file__).parent
    cardamom_article = cur_file_dir / 'results_cardamom_article'
    output_dir = (
        cur_file_dir.parent / 'cardamom_datasets' 
        if args.output is None else args.output
    )

    if args.format is None:
        convert(cardamom_article, output_dir)
    else:
        archive(cardamom_article, output_dir, args.format)

main()