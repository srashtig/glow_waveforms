"""
Lensed BBH frequency-domain source models for bilby / bilby_pipe (glow2).

Every model has the standard bilby BBH signature followed by *explicit* lens
parameters. bilby's WaveformGenerator only passes parameters that appear in
the signature, so every lens parameter below must be present in the prior (a
fixed value is fine, see glow_waveforms.glow_bilby.bilby_pe.priors). The keyword defaults are the
default lens parameters (``DEFAULT_LENS_PARAMS``) used for injections.

Two variants exist per lens model; both contain ``binary_black_hole`` in their
name so that bilby_pipe picks the BBH conversion/generation functions:
    lensed_binary_black_hole_<model>            (lal_binary_black_hole)
    lensed_gwsignal_binary_black_hole_<model>   (gwsignal_binary_black_hole)

bilby_pipe usage:
    frequency-domain-source-model = glow_waveforms.glow_bilby.source.lensed_binary_black_hole_SIS

Lens builders, default lens parameters and the shear-model frame conversions
live in glow_waveforms.lens_models (shared with glow_waveforms.glow_pycbc).
"""
import inspect

import numpy as np
from bilby.gw.source import lal_binary_black_hole, gwsignal_binary_black_hole

from ..amplification import lensed_pols
from ..lens_models import (LENS_BUILDERS, LENS_MODELS, DEFAULT_LENS_PARAMS, apply_lensing,
                      source_position_from_lens_offset, lens_offset_from_source_position)

_LENS_BUILDERS = LENS_BUILDERS   # backwards-compatible name

BBH_KEYS = ('mass_1', 'mass_2', 'luminosity_distance', 'a_1', 'tilt_1', 'phi_12',
            'a_2', 'tilt_2', 'phi_jl', 'theta_jn', 'phase')


def _lensed_bbh(bbh_model, lens_model, frequency_array, args):
    """
    args = locals() of a public source function: BBH params, lens params and
    the **kwargs dict (waveform arguments, passed only to the BBH model).
    """
    args = dict(args)
    waveform_kwargs = args.pop('kwargs')
    args.pop('frequency_array')
    bbh_params  = {key:args.pop(key) for key in BBH_KEYS}
    lens_params = args
    MLz = lens_params.pop('MLz')

    pols = bbh_model(frequency_array, **bbh_params, **waveform_kwargs)
    Fw_func, Fw_kwargs = LENS_BUILDERS[lens_model](**lens_params)
    return lensed_pols(frequency_array, pols, MLz, Fw_func, **Fw_kwargs)
## ========================================================


## ====  Unlensed
## ========================================================
def unlensed_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                               tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                               **kwargs):
    return lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                 tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                 **kwargs)

def unlensed_gwsignal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                        tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                        **kwargs):
    return gwsignal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                      tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                      **kwargs)
## ========================================================


## ====  Point lens (PL), analytical F(w)
## ========================================================
def lensed_binary_black_hole_PL(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                MLz=1e3, y=0.2,
                                **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'PL', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_PL(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                         tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                         MLz=1e3, y=0.2,
                                         **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'PL', frequency_array, locals())


## ====  Singular Isothermal Sphere (SIS)
## ========================================================
def lensed_binary_black_hole_SIS(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                 tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                 MLz=1e3, psi0=1., y=0.2,
                                 **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'SIS', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_SIS(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                          tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                          MLz=1e3, psi0=1., y=0.2,
                                          **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'SIS', frequency_array, locals())


## ====  Generalized SIS (gSIS)
## ========================================================
def lensed_binary_black_hole_gSIS(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                  tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                  MLz=1e3, psi0=1., y=0.2, k=1.,
                                  **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'gSIS', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_gSIS(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                           tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                           MLz=1e3, psi0=1., y=0.2, k=1.,
                                           **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'gSIS', frequency_array, locals())


## ====  Navarro Frenk White (NFW)
## ========================================================
def lensed_binary_black_hole_NFW(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                 tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                 MLz=1e3, psi0=2., y=0.2, xs=1.,
                                 **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'NFW', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_NFW(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                          tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                          MLz=1e3, psi0=2., y=0.2, xs=1.,
                                          **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'NFW', frequency_array, locals())


## ====  Point lens + external shear/convergence (Chang-Refsdal)
## ========================================================
def lensed_binary_black_hole_PLshear(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                     tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                     MLz=1e3, psi0=1., y=1.2, theta=0.2, g1=0.5, g2=0., kp=0.,
                                     **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'PLshear', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_PLshear(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                              tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                              MLz=1e3, psi0=1., y=1.2, theta=0.2, g1=0.5, g2=0., kp=0.,
                                              **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'PLshear', frequency_array, locals())

def lensed_binary_black_hole_PLshear_kpeqg1(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                            tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                            MLz=1e3, psi0=1., y=1.2, theta=0.2, g1=0.4, g2=0.,
                                            **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'PLshear_kpeqg1', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_PLshear_kpeqg1(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                                     tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                                     MLz=1e3, psi0=1., y=1.2, theta=0.2, g1=0.4, g2=0.,
                                                     **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'PLshear_kpeqg1', frequency_array, locals())


## ====  gSIS + external shear/convergence
## ========================================================
def lensed_binary_black_hole_gSISshear(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                       tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                       MLz=1e3, psi0=1., y=1.2, theta=0.2, k=1., g1=0.2, g2=0., kp=0.,
                                       **kwargs):
    return _lensed_bbh(lal_binary_black_hole, 'gSISshear', frequency_array, locals())

def lensed_gwsignal_binary_black_hole_gSISshear(frequency_array, mass_1, mass_2, luminosity_distance, a_1,
                                                tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                                                MLz=1e3, psi0=1., y=1.2, theta=0.2, k=1., g1=0.2, g2=0., kp=0.,
                                                **kwargs):
    return _lensed_bbh(gwsignal_binary_black_hole, 'gSISshear', frequency_array, locals())
## ========================================================


## ====  Registries =============
LENSED_WAVEFORMS = {'UL':unlensed_binary_black_hole,
                    **{m:globals()['lensed_binary_black_hole_'+m] for m in LENS_MODELS}}

LENSED_WAVEFORMS_GWSIGNAL = {'UL':unlensed_gwsignal_binary_black_hole,
                             **{m:globals()['lensed_gwsignal_binary_black_hole_'+m] for m in LENS_MODELS}}

def signature_defaults(func):
    """ Keyword defaults of a source function (== DEFAULT_LENS_PARAMS, checked in the tests). """
    params = inspect.signature(func).parameters
    return {k:p.default for k, p in params.items() if p.default is not inspect.Parameter.empty}

def source_model_path(model, gwsignal=False):
    """ Python path of a source model, e.g. for bilby_pipe ini files. """
    func = (LENSED_WAVEFORMS_GWSIGNAL if gwsignal else LENSED_WAVEFORMS)[model]
    return '{}.{}'.format(func.__module__, func.__name__)

def lens_polarizations(model, frequency_array, pols, **lens_params):
    """
    Apply the lens model's F*(w) to precomputed (unlensed) polarisations.

    Same result as LENSED_WAVEFORMS[model](frequency_array, <bbh>, **lens_params)
    when pols = unlensed_binary_black_hole(frequency_array, <bbh>), but the BBH
    waveform can be reused, e.g. for grids over lens parameters. Missing lens
    parameters take their DEFAULT_LENS_PARAMS values; pols is not modified.
    Returns None if F(w) could not be computed. (= glow_waveforms.lens_models.apply_lensing)
    """
    return apply_lensing(model, frequency_array, pols, **lens_params)
