#!/usr/bin/env python
"""
Bayesian PE with bilby for simulated injections/recoveries with glow2 lens models.
Port of scripts/glow2/pe_run_include_phi_spins_HM.py to glow_waveforms.glow_bilby.

Ex: python -m glow_waveforms.glow_bilby.bilby_pe.run_bilby PL PL outdir/PL_PL --npool 4
    python -m glow_waveforms.glow_bilby.bilby_pe.run_bilby PLshear_kpeqg1 SIS outdir/kpeqg1_0.4 --lens-override g1=0.4

models : UL, PL, SIS, gSIS, NFW, PLshear, PLshear_kpeqg1, gSISshear
"""
import argparse

import bilby
import numpy as np

from glow_waveforms.glow_bilby.source import LENSED_WAVEFORMS, LENSED_WAVEFORMS_GWSIGNAL
from glow_waveforms.glow_bilby.bilby_pe.priors import get_priors
from glow_waveforms.glow_bilby.bilby_pe.injections import get_injection, scale_distance_to_snr


def parse_overrides(items):
    """ ['g1=0.4', 'y=1'] -> {'g1': 0.4, 'y': 1.0} """
    return {k: float(v) for k, v in (item.split('=') for item in items)}

def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inj_model', choices=list(LENSED_WAVEFORMS))
    parser.add_argument('rec_model', choices=list(LENSED_WAVEFORMS))
    parser.add_argument('outdir')
    parser.add_argument('--npool', type=int, default=1)
    parser.add_argument('--lens-override', nargs='*', default=[],
                        help='injected lens parameters, e.g. g1=0.4 y=1.0')
    parser.add_argument('--injection-snr', type=float, default=30,
                        help='rescale d_L to this network SNR (0: unchanged)')
    parser.add_argument('--approximant', default='IMRPhenomXPNR')
    parser.add_argument('--gwsignal', action='store_true', help='use gwsignal instead of lalsimulation')
    parser.add_argument('--duration', type=float, default=4.)
    parser.add_argument('--sampling-frequency', type=float, default=2048.)
    parser.add_argument('--detectors', nargs='+', default=['H1', 'L1', 'V1'])
    parser.add_argument('--nlive', type=int, default=1000)
    parser.add_argument('--naccept', type=int, default=60)
    parser.add_argument('--distance-marginalization', action='store_true')
    parser.add_argument('--time-marginalization', action='store_true')
    parser.add_argument('--seed', type=int, default=88888881)
    return parser.parse_args()


def main():
    args = get_args()
    inj_model, rec_model, outdir = args.inj_model, args.rec_model, args.outdir

    np.random.seed(args.seed)
    bilby.core.utils.random.seed(args.seed)
    print("seed = ", args.seed)

    injection_parameters = get_injection(inj_model, **parse_overrides(args.lens_override))

    duration = args.duration
    sampling_frequency = args.sampling_frequency

    # set up output
    label = "zero_noise_" + inj_model + "_inj_" + rec_model + "_rec" + str(args.seed)
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # waveform generations
    waveform_arguments = dict(
        waveform_approximant=args.approximant,
        reference_frequency=20.0,
        minimum_frequency=20.0,
    )
    models = LENSED_WAVEFORMS_GWSIGNAL if args.gwsignal else LENSED_WAVEFORMS

    def waveform_generator(model):
        return bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=models[model],
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )
    waveform_generator_inj = waveform_generator(inj_model)
    waveform_generator_rec = waveform_generator(rec_model)

    if args.injection_snr != 0: # rescaling DL to set the injection_SNR
        scale_distance_to_snr(injection_parameters, waveform_generator_inj, args.injection_snr, args.detectors)

    ifos1 = bilby.gw.detector.InterferometerList(args.detectors)
    ifos1.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos1.inject_signal(waveform_generator=waveform_generator_inj, parameters=injection_parameters)
    snr_inj = np.sqrt(np.sum([ifo.optimal_snr_squared(ifo.frequency_domain_strain) for ifo in ifos1]))
    print('Injected SNR: %.3f' % np.real(snr_inj))

    # setup the priors (BBH + lens priors of the recovery model)
    priors = get_priors(rec_model, trigger_time=injection_parameters["geocent_time"])

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos1,
        waveform_generator=waveform_generator_rec,
        priors=priors,
        distance_marginalization=args.distance_marginalization,
        phase_marginalization=False,
        time_marginalization=args.time_marginalization,
    )

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        naccept=args.naccept,
        clean=True,
        sample="acceptance-walk",
        check_point_delta_t=1800,
        check_point_plot=True,
        resume=False,
        npool=args.npool,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )

    # Make a corner plot.
    result.plot_corner()

    # Plot the inferred waveform superposed on the actual data.
    result.plot_waveform_posterior(n_samples=1000)


if __name__ == "__main__":
    main()
