from harissa_benchmark.networks import available_networks
from harissa_benchmark.inferences import available_inferences
from harissa_benchmark.benchmark import Benchmark

from alive_progress.animations.spinners import bouncing_spinner_factory as _bsf
from alive_progress import config_handler as _cfh

__all__ = ['Benchmark', 'available_networks', 'available_inferences']

_cfh.set_global(
    bar='smooth',
    spinner=_bsf('🌶', 6, hide=False),
    unknown='horizontal',
    # dual_line=True,
    receipt=False,
    force_tty=True,
    length=20,
    max_cols=100
)
