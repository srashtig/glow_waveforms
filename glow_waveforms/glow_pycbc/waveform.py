"""
Lensed frequency-domain waveforms for pycbc (glow2 amplification factor).

Same lens models, parameters and code as glow_waveforms.glow_bilby (both use
glow_waveforms.lens_models): the polarisations of a base pycbc approximant are
multiplied by F*(w) with w = 8 pi G MLz f / c^3, in the band where the strain
is non-zero.

Approximants (registered with pycbc, see glow_waveforms.glow_pycbc.register):
    GLoW_PL, GLoW_SIS, GLoW_gSIS, GLoW_NFW, GLoW_PLshear, GLoW_PLshear_kpeqg1, GLoW_gSISshear
Extra waveform arguments: the lens parameters of the model (defaults in
LENS_DEFAULTS) and base_approximant (default IMRPhenomXPHM).

    from pycbc.waveform import get_fd_waveform
    hp, hc = get_fd_waveform(approximant='GLoW_PLshear_kpeqg1', base_approximant='IMRPhenomXPHM',
                             mass1=36, mass2=29, distance=410, f_lower=20, delta_f=1/64,
                             MLz=1e3, y=0.8, theta=np.pi/6, g1=0.2)

Shear models: (y, theta) is the lens-plane offset of the microlens from the
macro-image (see glow_waveforms.lens_models).
"""
from ..lens_models import LENS_MODELS, DEFAULT_LENS_PARAMS, apply_lensing

DEFAULT_BASE_APPROXIMANT = 'IMRPhenomXPHM'
LENS_DEFAULTS = {m: DEFAULT_LENS_PARAMS[m] for m in LENS_MODELS}   # shared with glow_waveforms.glow_bilby


class GLoWAmplificationError(RuntimeError):
    """ F(w) could not be computed for these lens parameters. """


## ====  Lensing of pycbc FrequencySeries
## ========================================================
def lens_frequency_series(model, hp, hc, **lens_params):
    """
    Return copies of the pycbc FrequencySeries hp, hc multiplied by F*(w) of
    `model`; missing lens parameters take their LENS_DEFAULTS values.
    Raises GLoWAmplificationError if F(w) fails.
    """
    freqs = hp.sample_frequencies.numpy()
    pols = apply_lensing(model, freqs, {'plus': hp.numpy(), 'cross': hc.numpy()}, **lens_params)
    if pols is None:
        raise GLoWAmplificationError("F(w) failed for {} with {}".format(model, lens_params))
    hp_l, hc_l = hp.copy(), hc.copy()
    hp_l.data[:] = pols['plus']
    hc_l.data[:] = pols['cross']
    return hp_l, hc_l

def get_fd_lensed_waveform(lens_model, base_approximant=DEFAULT_BASE_APPROXIMANT, **kwds):
    """
    Lensed (hp, hc): pycbc get_fd_waveform(base_approximant, **bbh kwds) x F*(w).
    Lens parameters of `lens_model` are taken from kwds (defaults otherwise).
    """
    from pycbc.waveform import get_fd_waveform
    kwds = dict(kwds)
    kwds.pop('approximant', None)
    lens_params = {k: kwds.pop(k) for k in list(kwds) if k in LENS_DEFAULTS[lens_model]}
    hp, hc = get_fd_waveform(approximant=base_approximant, **kwds)
    return lens_frequency_series(lens_model, hp, hc, **lens_params)


## ====  pycbc approximant functions: GLoW_<model>(**kwds) -> (hp, hc)
## ========================================================
def _make_approximant(model):
    def approximant(**kwds):
        base = kwds.pop('base_approximant', None) or DEFAULT_BASE_APPROXIMANT
        return get_fd_lensed_waveform(model, base_approximant=base, **kwds)
    approximant.__name__ = 'fd_' + model
    approximant.__doc__ = "pycbc FD approximant GLoW_{}: base approximant x F*(w).".format(model)
    return approximant

APPROXIMANTS = {}
for _model in LENS_MODELS:
    globals()['fd_' + _model] = APPROXIMANTS['GLoW_' + _model] = _make_approximant(_model)
