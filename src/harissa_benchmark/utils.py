from pathlib import Path
from importlib import import_module

def collect_modules(init_file, modules):
    for path in Path(init_file).parent.iterdir():
        module = path.stem
        if not module.startswith('_'):
            import_module(f'harissa_benchmark.networks.{module}')
            modules.append(module)