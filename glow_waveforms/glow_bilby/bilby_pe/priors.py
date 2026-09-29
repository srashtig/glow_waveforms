"""
Default priors for glow_waveforms.glow_bilby (adapted from glowpe/default_priors_injections.py).

The example injection is in glow_waveforms.glow_bilby.bilby_pe.injections and the default
lens parameters in glow_waveforms.glow_bilby.source.DEFAULT_LENS_PARAMS.

Every lens parameter in the source signature must be present in the prior
(fixed values allowed), since bilby's WaveformGenerator only passes the
parameters named in the signature.

bilby:      priors = get_priors('PLshear', trigger_time=t0)
bilby_pipe: prior-file = <glow_waveforms/glow_bilby/bilby_pipe_pe/prior_files/PLshear.prior>
            (written by write_prior_files(outdir))
            default-prior = glow_waveforms.glow_bilby.bilby_pe.priors.LensedBBHPriorDict
"""
import os

import numpy as np
import bilby
from bilby.core.prior import Uniform, LogUniform, PowerLaw, Sine, Cosine, Constraint
from bilby.gw.conversion import (convert_to_lal_binary_black_hole_parameters,
                                 generate_mass_parameters)
from bilby.gw.prior import BBHPriorDict, UniformComovingVolume



## ====  Conversion (adds the magnification of the type-I image, mu)
## ========================================================
def convert_to_lensed_bbh_parameters(parameters):
    """
    convert_to_lal_binary_black_hole_parameters + macro-model magnification
    mu = 1/((1-kp)^2 - g1^2 - g2^2) when g1 is available (kp = g1 for the
    kpeqg1 model, g2 = 0 if absent). mu can then be
    used as a Constraint prior. Same I/O as the bilby conversion functions.
    """
    converted, added_keys = convert_to_lal_binary_black_hole_parameters(parameters)
    if 'g1' in converted:
        g1 = converted['g1']
        g2 = converted.get('g2', 0.)
        kp = converted.get('kp', g1)
        converted['mu'] = 1 / ((1 - kp) ** 2 - g1 ** 2 - g2 ** 2)
        added_keys = added_keys + ['mu']
    return converted, added_keys

class LensedBBHPriorDict(BBHPriorDict):
    """ BBHPriorDict whose conversion function also generates mu (see above). """
    def default_conversion_function(self, sample):
        out_sample = bilby.gw.prior.fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lensed_bbh_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        return out_sample
## ========================================================


## ====  BBH priors
## ========================================================
def bbh_priors(trigger_time=None, deltaT=0.2):
    """ Default BBH priors; geocent_time prior only if trigger_time is given. """
    priors = LensedBBHPriorDict()  # bilby precessing_spins_bbh defaults

    priors["chirp_mass"] = Uniform(name="chirp_mass", minimum=22.0, maximum=80.0,
                                   latex_label="$\\mathcal{M}$", unit="$M_{\\odot}$")
    priors["mass_ratio"] = Uniform(name="mass_ratio", minimum=0.125, maximum=1.0,
                                   latex_label="$q$")
    priors["mass_1"] = Constraint(name="mass_1", minimum=1.001398, maximum=1000)
    priors["mass_2"] = Constraint(name="mass_2", minimum=1.001398, maximum=1000)
    priors["luminosity_distance"] = UniformComovingVolume(
        name="luminosity_distance", latex_label="$d_L$", unit="Mpc", minimum=100.0, maximum=5000)
    priors["theta_jn"] = Sine(name="theta_jn", latex_label="$\\theta_{JN}$",
                              minimum=0.0, maximum=np.pi)
    priors["ra"] = Uniform(name="ra", latex_label="$\\mathrm{RA}$", minimum=0.0,
                           maximum=2 * np.pi, boundary="periodic")
    priors["dec"] = Cosine(name="dec", latex_label="$\\mathrm{DEC}$",
                           minimum=-np.pi / 2.0, maximum=np.pi / 2.0)
    if trigger_time is not None:
        priors["geocent_time"] = Uniform(
            minimum=trigger_time - deltaT, maximum=trigger_time + deltaT,
            name="geocent_time", latex_label="$t_c$", unit="$s$")
    return priors
## ========================================================


## ====  Lens priors
## ========================================================
MLz_prior = LogUniform(10, 1e6, name="MLz", latex_label="$M_{Lz}$", unit="$M_{\\odot}$")
# p(y) ~ y: uniform in area (y dy dtheta) of the impact parameter / lens offset.
# For the shear models, with theta uniform, this is area-uniform in both the
# lens plane and the source plane (the map between them is linear).
y_prior   = PowerLaw(alpha=1, minimum=0.05, maximum=5.0, name="y", latex_label="$y$")

PL_p  = dict(MLz=MLz_prior, y=y_prior)
SIS_p = dict(MLz=MLz_prior, y=y_prior, psi0=1.)
gSIS_p = dict(SIS_p, k=Uniform(0.5, 1.5, name="k", latex_label="$k$"))
NFW_p = dict(SIS_p, psi0=Uniform(0.1, 5, name="psi0", latex_label="$\\psi_0$"), xs=1.)

PLshear_p = dict(
    MLz=MLz_prior, y=y_prior, psi0=1.,
    theta=Uniform(0.0, np.pi / 2, name="theta", latex_label="$\\theta$"),
    g1=Uniform(0, 0.8, name="g1", latex_label="$\\gamma_1$"),
    g2=0., kp=0.)

PLshear_kpeqg1_p = {k:v for k, v in PLshear_p.items() if k != 'kp'}
PLshear_kpeqg1_p["g1"] = Uniform(0, 0.49, name="g1", latex_label="$\\gamma_1=\\kappa$")

gSISshear_p = dict(PLshear_p, k=1.)

LENS_PRIORS = {
    "UL": {},
    "PL": PL_p,
    "SIS": SIS_p,
    "gSIS": gSIS_p,
    "NFW": NFW_p,
    "PLshear": PLshear_p,
    "PLshear_kpeqg1": PLshear_kpeqg1_p,
    "gSISshear": gSISshear_p,
}


def get_priors(model, trigger_time=None, deltaT=0.2, **overrides):
    """ BBH + lens priors for `model`; overrides replace individual entries. """
    priors = bbh_priors(trigger_time, deltaT)
    priors.update(LENS_PRIORS[model])
    priors.update(overrides)
    return priors

def write_prior_files(outdir):
    """
    Regenerate <model>.prior files (for bilby_pipe) from the definitions above.
    geocent_time is left out: bilby_pipe adds it from trigger-time and deltaT.
    """
    os.makedirs(outdir, exist_ok=True)
    for model in LENS_PRIORS:
        priors = get_priors(model)
        with open(os.path.join(outdir, "{}.prior".format(model)), "w") as f:
            f.write("# glow_waveforms.glow_bilby default priors: {} (generated by write_prior_files)\n".format(model))
            for key, prior in priors.items():
                f.write("{} = {}\n".format(key, repr(prior)))
## ========================================================
