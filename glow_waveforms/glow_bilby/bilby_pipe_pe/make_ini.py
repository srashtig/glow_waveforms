#!/usr/bin/env python
"""
Write a bilby_pipe ini for a glow_waveforms.glow_bilby injection/recovery pair.

The injected lens parameters are DEFAULT_LENS_PARAMS[inj_model] (+ overrides)
and d_L is rescaled to the requested network SNR (bilby_pipe cannot do this),
with the same zero-noise set-up as glow_waveforms.glow_bilby.bilby_pe.run_bilby.

The ini is written to --ini-dir (default: current directory).

Ex: python -m glow_waveforms.glow_bilby.bilby_pipe_pe.make_ini PLshear_kpeqg1 PLshear_kpeqg1 --snr 30
    python -m glow_waveforms.glow_bilby.bilby_pipe_pe.make_ini PLshear SIS --snr 30
    python -m glow_waveforms.glow_bilby.bilby_pipe_pe.make_ini PLshear_kpeqg1 SIS --lens-override g1=0.4 --label kpeqg1_0.4_SIS
    bilby_pipe PLshear_kpeqg1_inj_PLshear_kpeqg1_rec.ini --submit
"""
import argparse
import os

import bilby

from glow_waveforms.glow_bilby.source import LENSED_WAVEFORMS, source_model_path

HERE = os.path.dirname(os.path.abspath(__file__))
# <model>.prior files; regenerate with glow_waveforms.glow_bilby.bilby_pe.priors.write_prior_files(PRIOR_DIR)
PRIOR_DIR = os.path.join(HERE, 'prior_files')

from glow_waveforms.glow_bilby.bilby_pe.injections import get_injection, scale_distance_to_snr


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inj_model', choices=list(LENSED_WAVEFORMS))
    parser.add_argument('rec_model', choices=list(LENSED_WAVEFORMS))
    parser.add_argument('--label', default=None)
    parser.add_argument('--outdir', default=None, help='default: outdir_<label>')
    parser.add_argument('--ini-dir', default='.')
    parser.add_argument('--lens-override', nargs='*', default=[], help='e.g. g1=0.4 y=1.0')
    parser.add_argument('--snr', type=float, default=30, help='0: keep d_L')
    parser.add_argument('--approximant', default='IMRPhenomXPNR')
    parser.add_argument('--gwsignal', action='store_true')
    parser.add_argument('--detectors', nargs='+', default=['H1', 'L1', 'V1'])
    parser.add_argument('--duration', type=float, default=4.)
    parser.add_argument('--post-trigger-duration', type=float, default=2.,
                        help='data after the trigger [s]; must cover lensing time delays (large MLz)')
    parser.add_argument('--sampling-frequency', type=float, default=2048.)
    parser.add_argument('--nlive', type=int, default=1000)
    parser.add_argument('--naccept', type=int, default=60)
    parser.add_argument('--request-cpus', type=int, default=16)
    parser.add_argument('--distance-marginalization', action='store_true')
    parser.add_argument('--time-marginalization', action='store_true')
    parser.add_argument('--accounting', default='ligo.dev.o4.cbc.lensing.multi')
    return parser.parse_args()


def make_ini(args):
    overrides = {k: float(v) for k, v in (item.split('=') for item in args.lens_override)}
    injection = get_injection(args.inj_model, **overrides)

    if args.snr != 0:
        wfg = bilby.gw.WaveformGenerator(
            duration=args.duration, sampling_frequency=args.sampling_frequency,
            frequency_domain_source_model=LENSED_WAVEFORMS[args.inj_model],
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(waveform_approximant=args.approximant,
                                    reference_frequency=20., minimum_frequency=20.))
        scale_distance_to_snr(injection, wfg, args.snr, args.detectors)

    others = ["## other recovery models:"]
    for model in LENSED_WAVEFORMS:
        if model != args.rec_model:
            others += ["# prior-file = " + os.path.join(PRIOR_DIR, model + '.prior'),
                       "# frequency-domain-source-model = " + source_model_path(model, args.gwsignal)]

    label = args.label or "{}_inj_{}_rec".format(args.inj_model, args.rec_model)
    outdir = args.outdir or "outdir_" + label

    fill = dict(
        label=label,
        outdir=outdir,
        accounting=args.accounting,
        detectors="[" + ", ".join(args.detectors) + "]",
        duration=args.duration,
        post_trigger_duration=args.post_trigger_duration,
        sampling_frequency=args.sampling_frequency,
        maximum_frequency=args.sampling_frequency / 2,
        trigger_time=injection["geocent_time"],
        injection_dict="{" + ", ".join("'{}': {!r}".format(k, float(v)) for k, v in injection.items()) + "}",
        injection_source_model=source_model_path(args.inj_model, args.gwsignal),
        approximant=args.approximant,
        prior_file=os.path.join(PRIOR_DIR, args.rec_model + '.prior'),
        source_model=source_model_path(args.rec_model, args.gwsignal),
        other_recoveries="\n".join(others),
        distance_marginalization=args.distance_marginalization,
        time_marginalization=args.time_marginalization,
        nlive=args.nlive,
        naccept=args.naccept,
        request_cpus=args.request_cpus,
    )
    with open(os.path.join(HERE, "template.ini")) as f:
        ini = f.read().format(**fill)

    path = os.path.join(args.ini_dir, label + ".ini")
    with open(path, "w") as f:
        f.write(ini)
    print("written", path)
    return path


if __name__ == "__main__":
    make_ini(get_args())
