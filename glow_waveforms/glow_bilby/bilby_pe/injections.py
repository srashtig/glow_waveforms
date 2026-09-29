"""
Injection parameters for glow_waveforms.glow_bilby PE (bilby_pe/run_bilby.py, bilby_pipe_pe/make_ini.py,
example notebooks): one BBH injection + the default lens parameters of each model,
glow_waveforms.glow_bilby.source.DEFAULT_LENS_PARAMS[model].
"""
import bilby
import numpy as np

from ..source import DEFAULT_LENS_PARAMS

# typical precessing BBH (chi_eff = 0.101, chi_p = 0.405); d_L is rescaled to the
# target SNR by run_bilby.py / make_ini.py
injection_parameters = dict(
    mass_1=35.0,
    mass_2=30.0,      # q ~ 0.86 (Typical)
    a_1=0.42,         # Moderate spin to drive chi_p
    a_2=0.1,
    tilt_1=1.3,       # Highly tilted (~75 degrees)
    tilt_2=1.3,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1560.0, # Adjusted for SNR ~30
    theta_jn=0.4,
    phase=1.3,
    ra=1.375,
    dec=1.12108,
    psi=2.659,
    geocent_time=1126259642.413,
)

def get_injection(model, **overrides):
    """ BBH injection + default lens parameters of `model` + overrides. """
    params = dict(injection_parameters)
    params.update(DEFAULT_LENS_PARAMS[model])
    params.update(overrides)
    return params


def network_snr(parameters, waveform_generator, detectors=("H1", "L1", "V1")):
    """ Optimal network SNR in zero noise with bilby's default PSDs. """
    ifos = bilby.gw.detector.InterferometerList(list(detectors))
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=waveform_generator.sampling_frequency,
        duration=waveform_generator.duration,
        start_time=parameters["geocent_time"] - waveform_generator.duration / 2,
    )
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters)
    return np.sqrt(np.sum([np.real(ifo.optimal_snr_squared(ifo.frequency_domain_strain)) for ifo in ifos]))

def scale_distance_to_snr(parameters, waveform_generator, snr, detectors=("H1", "L1", "V1")):
    """ Rescale luminosity_distance (in place) so the network SNR is `snr`. """
    snr0 = network_snr(parameters, waveform_generator, detectors)
    parameters["luminosity_distance"] *= snr0 / snr
    return parameters
