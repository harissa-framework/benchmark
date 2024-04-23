from pathlib import Path
from importlib import import_module

def collect_submodules(init_file, submodules):
    module = Path(init_file).parent
    for path in module.iterdir():
        submodule = path.stem
        if not submodule.startswith('_'):
            import_module(f'harissa_benchmark.{module.stem}.{submodule}')
            submodules.append(submodule)


def match(path, include_patterns = ['**'], exclude_patterns = [], suffix = ''):
    path = Path(path)
    return (
        any(
            [path.match(f'{pattern}{suffix}') for pattern in include_patterns]
        ) and all(
            [~path.match(f'{pattern}{suffix}') for pattern in exclude_patterns]
        )
    )

def match_rec(
    path, 
    include_patterns = ['**'] , 
    exclude_patterns = [],
    suffix='.npz'
):
    def add_paths(p, acc):
        if p is not None:
            if p.is_dir():
                for sub_p in p.iterdir():
                    acc = add_paths(sub_p, acc)
            elif match(p, include_patterns, exclude_patterns, suffix):
                acc = add_paths(None, acc + [p])
        
        return acc
        
    return add_paths(path, [])