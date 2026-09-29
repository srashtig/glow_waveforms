#!/usr/bin/env python
"""
Waveform evaluation time per lens model, for parameters drawn from the
glow_waveforms.glow_bilby default priors (BBH + lens priors of each model).

Ex: python timing_waveforms.py --n-samples 100 --outdir timing
Writes timing/timing_<approximant>.csv and timing/timing_<approximant>.png
"""
import argparse
import os
import time

import bilby
import numpy as np

from glow_waveforms.glow_bilby.source import LENSED_WAVEFORMS
from glow_waveforms.glow_bilby.bilby_pe.priors import get_priors

TRIGGER_TIME = 1126259642.413


def time_model(model, n_samples=100, duration=4., sampling_frequency=2048.,
               approximant="IMRPhenomXPNR", seed=1234):
    """ Evaluation times [s] of frequency_domain_strain for prior samples. """
    bilby.core.utils.random.seed(seed)
    priors = get_priors(model, trigger_time=TRIGGER_TIME)
    samples = priors.sample(n_samples + 1)
    samples = [{k: samples[k][i] for k in samples} for i in range(n_samples + 1)]
    warmup, samples = samples[0], samples[1:]

    wfg = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=LENSED_WAVEFORMS[model],
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant=approximant,
                                reference_frequency=20., minimum_frequency=20.,
                                catch_waveform_errors=True))
    # warm-up on a separate sample (a repeated sample would hit bilby's cache)
    wfg.frequency_domain_strain(warmup)

    times, failed = np.zeros(n_samples), np.zeros(n_samples, dtype=bool)
    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        pols = wfg.frequency_domain_strain(sample)
        times[i] = time.perf_counter() - t0
        failed[i] = pols is None
    return times, failed, samples

def summary(times, failed):
    ok = times[~failed] if np.any(~failed) else times
    return dict(mean_ms=1e3*np.mean(ok), std_ms=1e3*np.std(ok),
                median_ms=1e3*np.median(ok), min_ms=1e3*np.min(ok), max_ms=1e3*np.max(ok),
                p5_ms=1e3*np.percentile(ok, 5), p95_ms=1e3*np.percentile(ok, 95),
                fail_rate=np.mean(failed))

def run(models=None, n_samples=100, approximant="IMRPhenomXPNR", **kwargs):
    models = models or list(LENSED_WAVEFORMS)
    rows, all_times = {}, {}
    for model in models:
        times, failed, _ = time_model(model, n_samples, approximant=approximant, **kwargs)
        rows[model], all_times[model] = summary(times, failed), (times, failed)
        r = rows[model]
        print("{:15s} {:8.1f} +- {:7.1f} ms  [{:.1f}, {:.1f}]  fail {:.0%}".format(
            model, r['mean_ms'], r['std_ms'], r['min_ms'], r['max_ms'], r['fail_rate']))
    return rows, all_times

def plot(rows, all_times, filename=None):
    import matplotlib.pyplot as plt
    models = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.bar(models, [rows[m]['mean_ms'] for m in models],
           yerr=[rows[m]['std_ms'] for m in models], capsize=4, color='C0')
    ax.set_yscale('log'); ax.set_ylabel('time per waveform [ms]'); ax.set_title('mean $\\pm$ std')
    ax.tick_params(axis='x', rotation=45)
    ax = axes[1]
    ax.boxplot([1e3*all_times[m][0] for m in models], tick_labels=models, whis=(0, 100))
    ax.set_yscale('log'); ax.set_ylabel('time per waveform [ms]'); ax.set_title('full range')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
    return fig

def write_csv(rows, filename):
    keys = list(next(iter(rows.values())))
    with open(filename, 'w') as f:
        f.write(','.join(['model'] + keys) + '\n')
        for model, r in rows.items():
            f.write(','.join([model] + ['{:.4g}'.format(r[k]) for k in keys]) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='*', default=None, choices=list(LENSED_WAVEFORMS))
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--approximant', default='IMRPhenomXPNR')
    parser.add_argument('--outdir', default='timing')
    args = parser.parse_args()

    bilby.core.utils.logger.setLevel('WARNING')
    os.makedirs(args.outdir, exist_ok=True)
    rows, all_times = run(args.models, args.n_samples, args.approximant)
    base = os.path.join(args.outdir, 'timing_' + args.approximant)
    write_csv(rows, base + '.csv')
    plot(rows, all_times, base + '.png')
