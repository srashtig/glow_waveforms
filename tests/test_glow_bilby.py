import inspect
import os

import numpy as np
import pytest
import bilby

from glow_waveforms import amplification, lens_models
from glow_waveforms.glow_bilby import source
from glow_waveforms.glow_bilby.bilby_pe import priors

INJ = dict(mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7,
           phi_jl=0.3, luminosity_distance=1640., theta_jn=0.4, phase=1.3,
           ra=1.375, dec=1.12108, psi=2.659, geocent_time=1126259642.413)
WF = dict(waveform_approximant="IMRPhenomXPHM", reference_frequency=20., minimum_frequency=20.)
MODELS = list(source.LENSED_WAVEFORMS)
PRIOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glow_waveforms', 'glow_bilby', 'bilby_pipe_pe', 'prior_files')


def test_registries_consistent():
    assert set(source.LENSED_WAVEFORMS) == set(priors.LENS_PRIORS) == set(source.DEFAULT_LENS_PARAMS)
    assert set(source.LENSED_WAVEFORMS) == set(source.LENSED_WAVEFORMS_GWSIGNAL)


@pytest.mark.parametrize("model", MODELS)
def test_lal_and_gwsignal_signatures_match(model):
    lal = inspect.signature(source.LENSED_WAVEFORMS[model])
    gws = inspect.signature(source.LENSED_WAVEFORMS_GWSIGNAL[model])
    assert str(lal) == str(gws)


@pytest.mark.parametrize("model", MODELS)
def test_prior_covers_signature(model):
    """ WaveformGenerator needs every signature parameter (KeyError otherwise). """
    keys = set(bilby.core.utils.infer_parameters_from_function(source.LENSED_WAVEFORMS[model]))
    lens_keys = keys - set(source.BBH_KEYS)
    assert lens_keys == set(priors.LENS_PRIORS[model])


@pytest.mark.parametrize("model", MODELS)
def test_waveform_generator_default_params(model):
    wfg = bilby.gw.WaveformGenerator(
        duration=4, sampling_frequency=2048,
        frequency_domain_source_model=source.LENSED_WAVEFORMS[model],
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=WF)
    pols = wfg.frequency_domain_strain({**INJ, **source.DEFAULT_LENS_PARAMS[model]})
    assert pols is not None
    for key in pols:
        assert np.all(np.isfinite(pols[key]))


@pytest.mark.parametrize("model", MODELS)
def test_prior_files_round_trip(model):
    """ Shipped bilby_pipe .prior files are up to date and load into LensedBBHPriorDict. """
    import os, tempfile
    shipped = os.path.join(PRIOR_DIR, model + '.prior')
    with tempfile.TemporaryDirectory() as tmp:
        priors.write_prior_files(tmp)
        assert open(os.path.join(tmp, model + '.prior')).read() == open(shipped).read()
    from_file = priors.LensedBBHPriorDict(filename=shipped)
    from_code = priors.get_priors(model)
    assert set(from_file) == set(from_code)
    sample = from_file.sample()
    assert all(k in sample for k in priors.LENS_PRIORS[model])


def test_lensing_whole_band():
    """ The first and last non-zero bins are lensed too (fix wrt glowpe). """
    f = amplification.get_freq_array(4, 2048)
    ul = source.LENSED_WAVEFORMS['UL'](f, **{k: INJ[k] for k in source.BBH_KEYS}, **WF)
    pl = source.LENSED_WAVEFORMS['PL'](f, **{k: INJ[k] for k in source.BBH_KEYS}, **WF)
    idx = np.where(np.abs(ul['plus']) > 0)[0]
    assert not np.isclose(pl['plus'][idx[0]], ul['plus'][idx[0]], atol=0)
    assert not np.isclose(pl['plus'][idx[-1]], ul['plus'][idx[-1]], atol=0)


def test_mu_conversion():
    p, added = priors.convert_to_lensed_bbh_parameters(dict(INJ, g1=0.3, kp=0.1))
    assert np.isclose(p['mu'], 1 / (0.9**2 - 0.09)) and 'mu' in added
    p, _ = priors.convert_to_lensed_bbh_parameters(dict(INJ, g1=0.3))  # kpeqg1
    assert np.isclose(p['mu'], 1 / (1 - 0.6))


def test_shear_models_mishra_setup():
    """
    PL + shear (Mishra et al fig 2, source 0.4 from the lens) with the package
    PLshear builder, after mapping the source position to the package
    (y, theta), agrees with an independent glow2 set-up (lens at the origin,
    source displaced, eval_mode='full').
    """
    from glow2 import lenses
    ws = amplification.GMsun8pi * 150 * np.linspace(10, 3e3, 2000)
    rot = np.array([[np.cos(-np.pi/8), -np.sin(-np.pi/8)], [np.sin(-np.pi/8), np.cos(-np.pi/8)]])
    gamma_rot = rot @ np.array([[0.5, 0], [0, -0.5]]) @ rot.T
    g1, g2 = gamma_rot[1, 1], gamma_rot[0, 1]
    y, theta = source.lens_offset_from_source_position(0.4, 0., g1, g2, 0.)
    Fw_func, Fw_kwargs = source._LENS_BUILDERS['PLshear'](psi0=1., y=y, theta=theta, g1=g1, g2=g2, kp=0.)
    F = Fw_func(w=ws, **Fw_kwargs)
    ref = lenses.Psi_ChangRefsdal({'psi0': 1, 'xc1': 0, 'xc2': 0, 'gamma1': g1, 'gamma2': g2, 'kappa': 0})
    F_ref = amplification.get_Fw(ref, ws, y1=0.4, y2=0., im_mode='full')
    assert np.median(np.abs(F - F_ref) / np.abs(F_ref)) < 1e-3

def test_source_position_mapping():
    """ Package convention == lens at origin with the source at the mapped position. """
    from glow2 import lenses
    y, theta, g1, kp = 1.2, 0.2, 0.5, 0.1
    ws = np.linspace(0.1, 20, 500)
    Fw_func, Fw_kwargs = source._LENS_BUILDERS['PLshear'](psi0=1., y=y, theta=theta, g1=g1, g2=0., kp=kp)
    F_pkg = Fw_func(w=ws, **Fw_kwargs)
    y_src, th_src = source.source_position_from_lens_offset(y, theta, g1, 0., kp)
    std = lenses.Psi_ChangRefsdal({'psi0': 1, 'xc1': 0, 'xc2': 0, 'gamma1': g1, 'gamma2': 0, 'kappa': kp})
    F_std = amplification.get_Fw(std, ws, y1=y_src*np.cos(th_src), y2=y_src*np.sin(th_src))
    assert np.median(np.abs(np.abs(F_std) - np.abs(F_pkg)) / np.abs(F_pkg)) < 1e-3
    assert np.allclose(source.lens_offset_from_source_position(y_src, th_src, g1, 0., kp), (y, theta))


@pytest.mark.parametrize("model", MODELS)
def test_lens_polarizations_matches_source(model):
    """ lens_polarizations on a precomputed UL waveform == full source function. """
    f = amplification.get_freq_array(4, 2048)
    bbh = {k: INJ[k] for k in source.BBH_KEYS}
    ul = source.unlensed_binary_black_hole(f, **bbh, **WF)
    lp = dict(source.DEFAULT_LENS_PARAMS[model])
    if 'y' in lp:
        lp['y'] = 0.7
    new = source.lens_polarizations(model, f, ul, **lp)
    ref = source.LENSED_WAVEFORMS[model](f, **bbh, **lp, **WF)
    for key in ref:
        assert np.allclose(new[key], ref[key], rtol=1e-10, atol=0)
    assert np.all(ul['plus'] == source.unlensed_binary_black_hole(f, **bbh, **WF)['plus'])  # not modified


@pytest.mark.parametrize("model", MODELS)
def test_signature_defaults_are_shared_defaults(model):
    """ Keyword defaults of the bilby source functions == glow_waveforms.lens_models.DEFAULT_LENS_PARAMS. """
    assert source.signature_defaults(source.LENSED_WAVEFORMS[model]) == lens_models.DEFAULT_LENS_PARAMS[model]
    assert source.signature_defaults(source.LENSED_WAVEFORMS_GWSIGNAL[model]) == lens_models.DEFAULT_LENS_PARAMS[model]
