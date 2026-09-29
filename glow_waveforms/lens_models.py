"""
Lens models shared by glow_waveforms.glow_bilby and glow_waveforms.glow_pycbc:
lens builders (lens parameters -> F(w) evaluator), default lens parameters,
generic lensing of precomputed polarisations, and the frame conversions of the
shear models.

Shear models (PLshear, PLshear_kpeqg1, gSISshear) keep the glowpe convention:
the source is at the origin, the lens at c = y (cos theta, sin theta), and glow2
centres the external shear/convergence on the origin. c is therefore the
microlens offset from the unperturbed macro-image in the lens plane; the
source-plane offset is u = -((1-kp) I - Gamma) c (|det| = 1/|mu|). Convert with
source_position_from_lens_offset / lens_offset_from_source_position.
"""
import numpy as np

from glow2 import lenses, freq_domain

from .amplification import lensed_pols, get_Fw, get_Fw_sym, get_Fw_analytical, GMsun8pi


## ====  Lens builders: lens parameters -> (Fw_func, Fw_kwargs)
## ========================================================
def _lens_PL(y, **_):
    return get_Fw_analytical, dict(Fw=freq_domain.Fw_PL(y))

def _lens_SIS(psi0, y, **_):
    return get_Fw_sym, dict(Psi=lenses.Psi_SIS({'psi0':psi0}), y=y)

def _lens_gSIS(psi0, y, k, **_):
    return get_Fw_sym, dict(Psi=lenses.Psi_gSIS({'psi0':psi0, 'k':k}), y=y)

def _lens_NFW(psi0, y, xs, **_):
    return get_Fw_sym, dict(Psi=lenses.Psi_NFW({'psi0':psi0, 'xs':xs}), y=y)

def _lens_PLshear(psi0, y, theta, g1, g2, kp, **_):
    # glowpe convention: source at the origin, lens displaced by y along theta.
    # glow2 centres the external shear/convergence on the origin, so (y, theta)
    # is the microlens offset from the unperturbed macro-image in the *lens
    # plane*; the source-plane offset is source_position_from_lens_offset.
    p_phys = {'psi0':psi0, 'xc1':y*np.cos(theta), 'xc2':y*np.sin(theta),
              'gamma1':g1, 'gamma2':g2, 'kappa':kp}
    return get_Fw, dict(Psi=lenses.Psi_ChangRefsdal(p_phys), im_mode='slim')

def _lens_PLshear_kpeqg1(psi0, y, theta, g1, g2, **_):
    # convergence tied to the shear: kappa = gamma1
    return _lens_PLshear(psi0, y, theta, g1, g2, kp=g1)

def _lens_gSISshear(psi0, y, theta, k, g1, g2, kp, **_):
    # same convention as PLshear
    Psi1 = lenses.Psi_gSIS({'psi0':psi0, 'xc1':y*np.cos(theta), 'xc2':y*np.sin(theta), 'k':k})
    Psi2 = lenses.Psi_Ext({'gamma1':g1, 'gamma2':g2, 'kappa':kp})
    return get_Fw, dict(Psi=lenses.CompositeLens([Psi1, Psi2]), im_mode='full')

LENS_BUILDERS = {'PL':_lens_PL, 'SIS':_lens_SIS, 'gSIS':_lens_gSIS, 'NFW':_lens_NFW,
                 'PLshear':_lens_PLshear, 'PLshear_kpeqg1':_lens_PLshear_kpeqg1,
                 'gSISshear':_lens_gSISshear}
LENS_MODELS = list(LENS_BUILDERS)

# default lens parameters (keyword defaults of the glow_waveforms.glow_bilby source functions)
DEFAULT_LENS_PARAMS = {
    'UL':             {},
    'PL':             dict(MLz=1e3, y=0.2),
    'SIS':            dict(MLz=1e3, psi0=1., y=0.2),
    'gSIS':           dict(MLz=1e3, psi0=1., y=0.2, k=1.),
    'NFW':            dict(MLz=1e3, psi0=2., y=0.2, xs=1.),
    'PLshear':        dict(MLz=1e3, psi0=1., y=1.2, theta=0.2, g1=0.5, g2=0., kp=0.),
    'PLshear_kpeqg1': dict(MLz=1e3, psi0=1., y=1.2, theta=0.2, g1=0.4, g2=0.),
    'gSISshear':      dict(MLz=1e3, psi0=1., y=1.2, theta=0.2, k=1., g1=0.2, g2=0., kp=0.),
}


## ====  F(w) and lensing of precomputed polarisations
## ========================================================
def amplification_factor(model, w, debug=False, **lens_params):
    """
    F*(w) (the factor multiplying the polarisations) of `model` at dimensionless
    frequencies w; lens parameters without MLz. None if glow2 fails.
    With debug=True returns (It, Fw*) where available.
    """
    Fw_func, Fw_kwargs = LENS_BUILDERS[model](**lens_params)
    return Fw_func(w=w, debug=debug, **Fw_kwargs)

def apply_lensing(model, frequency_array, pols, **lens_params):
    """
    Multiply copies of the polarisations {'plus', 'cross'} (arrays on
    frequency_array [Hz]) by F*(w), w = GMsun8pi * MLz * f, in the band where
    the strain is non-zero. Missing lens parameters take DEFAULT_LENS_PARAMS.
    Returns None if F(w) could not be computed; pols is not modified.
    """
    unknown = set(lens_params) - set(DEFAULT_LENS_PARAMS[model])
    if unknown:
        raise ValueError("unknown lens parameters for {}: {}".format(model, sorted(unknown)))
    if pols is None:
        return None
    pols = {key: np.array(val, copy=True) for key, val in pols.items()}
    if model == 'UL':
        return pols
    params = {**DEFAULT_LENS_PARAMS[model], **lens_params}
    MLz = params.pop('MLz')
    Fw_func, Fw_kwargs = LENS_BUILDERS[model](**params)
    return lensed_pols(frequency_array, pols, MLz, Fw_func, **Fw_kwargs)


## ====  Shear models: package (y, theta) <-> source position relative to the lens
## ========================================================
def _macro_matrix(g1, g2, kp):
    """ (1-kp) I - Gamma """
    return np.array([[1 - kp - g1, -g2], [-g2, 1 - kp + g1]])

def _polar(v1, v2, fold):
    r, phi = np.hypot(v1, v2), np.arctan2(v2, v1)
    if fold:  # g2 = 0: lens symmetric under x1 -> -x1 and x2 -> -x2
        phi = np.arctan(np.abs(np.tan(phi)))
    return r, phi

def source_position_from_lens_offset(y, theta, g1, g2=0., kp=0.):
    """
    Source-plane offset (|u|, arg u) of the source from the lens, with the
    external field centred on the lens (convention of Mishra et al),
    for the package (lens-plane) parameters (y, theta) of the shear models:
        u = -((1-kp) I - Gamma) c,   c = y (cos theta, sin theta).
    Areas scale by |det| = 1/|mu_macro|: sigma_source = sigma_lens / |mu|.
    The angle is folded into [0, pi/2] when g2 = 0 (reflection symmetry).
    """
    c = np.array([y*np.cos(theta), y*np.sin(theta)])
    u = -_macro_matrix(g1, g2, kp) @ c
    return _polar(u[0], u[1], g2 == 0)

def lens_offset_from_source_position(y_src, theta_src, g1, g2=0., kp=0.):
    """ Inverse of source_position_from_lens_offset: package (y, theta). """
    u = np.array([y_src*np.cos(theta_src), y_src*np.sin(theta_src)])
    c = -np.linalg.solve(_macro_matrix(g1, g2, kp), u)
    return _polar(c[0], c[1], g2 == 0)
## ========================================================
