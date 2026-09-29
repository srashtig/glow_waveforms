import numpy as np
import pytest

from glow_waveforms import glow_pycbc
from glow_waveforms.glow_pycbc import waveform as W

BBH = dict(mass1=36., mass2=29., spin1z=0.3, spin2z=-0.1, distance=410., inclination=0.4,
           coa_phase=1.3, f_lower=20., delta_f=1. / 16, f_ref=20.)
MODELS = W.LENS_MODELS


@pytest.mark.parametrize("model", MODELS)
def test_approximant_via_pycbc(model):
    """ GLoW_<model> through pycbc.waveform.get_fd_waveform == base x F*(w). """
    from pycbc.waveform import get_fd_waveform
    glow_pycbc.register()
    lp = dict(W.LENS_DEFAULTS[model], y=0.7)
    hp, hc = get_fd_waveform(approximant='GLoW_' + model, base_approximant='IMRPhenomXAS', **BBH, **lp)
    hp0, hc0 = get_fd_waveform(approximant='IMRPhenomXAS', **BBH)
    hp1, hc1 = W.lens_frequency_series(model, hp0, hc0, **lp)
    assert np.allclose(hp.numpy(), hp1.numpy(), rtol=1e-10, atol=0)
    assert np.allclose(hc.numpy(), hc1.numpy(), rtol=1e-10, atol=0)
    assert not np.allclose(hp.numpy(), hp0.numpy(), rtol=1e-3, atol=0)


@pytest.mark.parametrize("model", MODELS)
def test_same_amplification_as_glow_bilby(model):
    """ The applied factor F*(w) equals glow_bilby's on the same frequency grid. """
    glow_bilby = pytest.importorskip("glow_waveforms.glow_bilby")
    from pycbc.waveform import get_fd_waveform
    hp0, hc0 = get_fd_waveform(approximant='IMRPhenomXAS', **BBH)
    lp = dict(W.LENS_DEFAULTS[model], y=0.7)
    hp1, _ = W.lens_frequency_series(model, hp0, hc0, **lp)
    f = hp0.sample_frequencies.numpy()
    ref = glow_bilby.source.lens_polarizations(model, f, {'plus': hp0.numpy(), 'cross': hc0.numpy()}, **lp)
    assert np.allclose(hp1.numpy(), ref['plus'], rtol=1e-10, atol=0)


def test_defaults_match_glow_bilby():
    glow_bilby = pytest.importorskip("glow_waveforms.glow_bilby")
    for model in MODELS:
        assert W.LENS_DEFAULTS[model] == glow_bilby.source.DEFAULT_LENS_PARAMS[model]


def test_unknown_lens_parameter():
    from pycbc.waveform import get_fd_waveform
    hp0, hc0 = get_fd_waveform(approximant='IMRPhenomXAS', **BBH)
    with pytest.raises(ValueError):
        W.lens_frequency_series('PL', hp0, hc0, kp=0.1)
