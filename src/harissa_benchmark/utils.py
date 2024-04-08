from pathlib import Path
from importlib import import_module

def collect_submodules(init_file, modules):
    module = Path(init_file).parent
    for path in module.iterdir():
        submodule = path.stem
        if not submodule.startswith('_'):
            import_module(f'harissa_benchmark.{module.stem}.{submodule}')
            modules.append(submodule)