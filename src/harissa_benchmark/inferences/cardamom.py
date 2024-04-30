try:
    import numpy as np
    from harissa.inference import Cardamom
    from harissa_benchmark.generators import InferencesGenerator, InferenceInfo

    InferencesGenerator.register(
        'Cardamom', 
        InferenceInfo(
            Cardamom, 
            True, 
            np.array([InferencesGenerator.color_map(8), 
                    InferencesGenerator.color_map(9)])
        )
    )
except ImportError as e:
    from sys import stderr
    print(e.msg, file=stderr)