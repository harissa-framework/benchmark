import numpy as np

from harissa.inference import Hartree
from harissa_benchmark.generators import InferencesGenerator, InferenceInfo

InferencesGenerator.register(
    'Hartree', 
    InferenceInfo(
        Hartree,
        True,
        np.array([InferencesGenerator.color_map(6),
                  InferencesGenerator.color_map(7)])
    )

)
