"""glow_waveforms.glow_pycbc: wave-optics lensed waveforms (glow2) as pycbc approximants GLoW_<model>."""
from .. import amplification, lens_models
from . import waveform
from .waveform import (LENS_MODELS, LENS_DEFAULTS, APPROXIMANTS, GLoWAmplificationError,
                       lens_frequency_series, get_fd_lensed_waveform)


def register(force=True):
    """
    Register the GLoW_<model> approximants with pycbc (only needed when the
    package is not pip-installed; installation registers them as pycbc plugins).
    """
    from pycbc.waveform.plugin import add_custom_waveform
    for name, func in APPROXIMANTS.items():
        add_custom_waveform(name, func, 'frequency', force=force)
    return list(APPROXIMANTS)
