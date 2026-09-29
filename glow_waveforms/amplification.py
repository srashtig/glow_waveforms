"""
Amplification factor F(w) with glow2 and its application to GW polarisations.

Shared by glow_waveforms.glow_bilby and glow_waveforms.glow_pycbc. Adapted from
glowpe/sources.py (glow2 version); see REVIEW.md for the changes.
"""
import numpy as np
import logging

from glow2 import images, time_domain, freq_domain
from glow2.common import GLoWException

_logger = logging.getLogger('glow_waveforms')

# 8 pi G Msun / c^3 [s]: w = GMsun8pi * MLz * f
GMsun8pi = 1.23791089411462657335e-04


## ====  General functions to lens waveforms
## ========================================================
def get_freq_array(duration=4, sampling_frequency=2048):
    freq_array = 1./duration*np.arange(0, int(duration*sampling_frequency/2.)+1)
    return freq_array

def restrict_lens_freqs(pols, freq_array):
    """ Mask of the frequency band where the strain is non-zero (inclusive). """
    # either one of them is not zero
    ids = (np.abs(pols['plus'])>0) | (np.abs(pols['cross'])>0)
    if not np.any(ids):
        return ids

    fs  = freq_array[ids]
    ids = (freq_array >= fs[0]) & (freq_array <= fs[-1]) & (freq_array > 0)

    return ids

def _log_failure(logger, failed_parameters, wmin, wmax):
    if logger is None:
        return
    failed_parameters = dict(failed_parameters)
    failed_parameters['wmin'] = wmin
    failed_parameters['wmax'] = wmax
    logger.debug("Evaluating the amplification factor failed\n" +
                 "The parameters were {}\n".format(failed_parameters) +
                 "Likelihood will be set to -inf.")

def _check_finite(Fws):
    """ glow2 may return NaN/inf without raising: treat it as a failure. """
    if Fws is None or not np.all(np.isfinite(Fws)):
        raise GLoWException("non-finite amplification factor")
    return Fws

def get_Fw(Psi, w, y1=0., y2=0., logger=_logger, im_mode='slim', debug=False):
    """
    Compute F(w) for non-sym lenses with the source at (y1, y2).

    NB: in glow2 external fields (Psi_Ext, the shear/convergence of
    Psi_ChangRefsdal) are centred on the coordinate origin, so the lens must
    sit at the origin and the source be displaced (not the other way round).
    """
    wmin = w[0] if not w[0]==0 else w[1]
    wmax = w[-1]

    Cprec = {} #{'no_output':1}
    p_prec_im = {'eval_mode':im_mode}
    p_prec_t  = {}
    p_prec_w  = {'wmin':wmin, 'wmax':wmax}

    Img_method = images.Images_2d
    It_method  = time_domain.It_Contour2d
    Fw_method  = freq_domain.Fw_MixedFT

    ## ----------------------------------------------------------------------

    It = None
    try:
        Img = Img_method(Psi, y1, y2, p_prec_im, Cprec=Cprec)
        It  = It_method(Img, p_prec_t)
        Fw  = Fw_method(It, p_prec_w)
        Fws = _check_finite(np.conj(Fw(w)))
    except GLoWException:
        Fws = None
        _log_failure(logger, {**Psi.p_phys, 'y1':y1, 'y2':y2}, wmin, wmax)

    if debug is True:
        return It, Fws
    else:
        return Fws

def get_Fw_sym(Psi, y, w, logger=_logger, use_Int1d=True, debug=False):
    """ Compute F(w) for sym lenses. Falls back to the 2d contour method. """
    wmin = w[0] if not w[0]==0 else w[1]
    wmax = w[-1]

    Cprec = {} #{'no_output':1}
    p_prec_im = {}
    p_prec_t  = {}
    p_prec_w  = {'wmin':wmin, 'wmax':wmax}

    if use_Int1d is True:
        Img_method = images.Images_1d
        It_method  = time_domain.It_Integral1d
    else:
        # will be used as a fallback method
        p_prec_im['eval_mode'] = 'full'
        Img_method = images.Images_2d
        It_method  = time_domain.It_Contour2d
    Fw_method  = freq_domain.Fw_MixedFT

    ## ----------------------------------------------------------------------

    It = None
    try:
        Img = Img_method(Psi, y, 0, p_prec_im, Cprec=Cprec)
        It  = It_method(Img, p_prec_t)
        Fw  = Fw_method(It, p_prec_w)
        Fws = _check_finite(np.conj(Fw(w)))
    except GLoWException:
        if use_Int1d is True:
            # use fallback method
            return get_Fw_sym(Psi, y, w, logger=logger, use_Int1d=False, debug=debug)
        Fws = None
        _log_failure(logger, {**Psi.p_phys, 'y':y}, wmin, wmax)

    if debug is True:
        return It, Fws
    else:
        return Fws

def get_Fw_analytical(Fw, w, logger=_logger, debug=False):
    """ Compute F(w) for analytical lenses. """
    try:
        Fws = _check_finite(np.conj(Fw(w)))
    except GLoWException:
        Fws = None
        wmin = w[0] if not w[0]==0 else w[1]
        _log_failure(logger, {'y':Fw.y}, wmin, w[-1])

    if debug is True:
        return Fw, Fws
    else:
        return Fws

def lensed_pols(freq_array, pols, MLz, Fw_func, debug=False, **Fw_kwargs):
    """
    Multiply the polarisations by F(w) in the band where the strain is non-zero.

    Fw_func is one of get_Fw, get_Fw_sym, get_Fw_analytical; Fw_kwargs are its
    lens arguments (Psi / y / Fw, im_mode ...). Returns None if pols is None or
    F(w) could not be computed. With debug=True returns (ws, Fws, It, fs, pols).
    """
    if pols is None:
        return (None,)*5 if debug else None

    ids = restrict_lens_freqs(pols, freq_array)  # non-zero strain
    ws  = GMsun8pi*MLz*freq_array[ids]
    if ws.size == 0:
        return (ws, None, None, freq_array[ids], pols) if debug else pols

    It, Fws = Fw_func(w=ws, debug=True, **Fw_kwargs)

    if Fws is None:
        pols = None
    else:
        for key in pols:
            pols[key][ids] *= Fws

    if debug is True:
        return ws, Fws, It, freq_array[ids], pols
    return pols
## ========================================================
