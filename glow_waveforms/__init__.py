"""
glow_waveforms: wave-optics lensed GW waveforms with glow2.

    glow_waveforms.amplification   F(w) with glow2 (shared)
    glow_waveforms.lens_models          lens models, default parameters, frame conversions (shared)
    glow_waveforms.glow_bilby      bilby / bilby_pipe source models, priors, PE scripts   [needs bilby]
    glow_waveforms.glow_pycbc      pycbc approximants GLoW_<model>                         [needs pycbc]
"""
__version__ = "0.1.0"

from . import amplification, lens_models
from .lens_models import LENS_MODELS, DEFAULT_LENS_PARAMS, apply_lensing
