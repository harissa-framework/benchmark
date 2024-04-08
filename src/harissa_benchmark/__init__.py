from harissa_benchmark.networks import available_networks
from harissa_benchmark.generators import NetworksGenerator, DatasetsGenerator

from alive_progress.animations.spinners import bouncing_spinner_factory as _bsf
from alive_progress import config_handler as _cfh

__all__ = ['NetworksGenerator', 'DatasetsGenerator' 'available_networks']

_cfh.set_global(
    spinner=_bsf('🌶', 6, hide=False),
    dual_line=True,
    receipt=False
)
