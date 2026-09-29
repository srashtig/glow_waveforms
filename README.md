# glow_waveforms

Wave-optics lensed GW waveforms, with the amplification factor F(w) computed by [glow2](<GLOW2_GIT_URL>). One package serves **bilby / bilby_pipe** and **pycbc**, and both use the same lens code:

```
glow_waveforms/
  amplification.py      F(w) with glow2 (get_Fw, get_Fw_sym, get_Fw_analytical, lensed_pols)    [shared]
  lens_models.py        lens builders, LENS_MODELS, DEFAULT_LENS_PARAMS, apply_lensing,
                        source_position_from_lens_offset / lens_offset_from_source_position     [shared]
  glow_bilby/           bilby / bilby_pipe layer                                                  [needs bilby]
      source.py             bilby source functions (lal + gwsignal), LENSED_WAVEFORMS, lens_polarizations
      bilby_pe/             priors.py (default priors, LensedBBHPriorDict), injections.py, run_bilby.py
      bilby_pipe_pe/        make_ini.py, template.ini, example ini, prior_files/<model>.prior
  glow_pycbc/           pycbc layer: approximants GLoW_<model>, lens_frequency_series             [needs pycbc]
examples/               example_glow_bilby.ipynb, example_glow_pycbc_mismatch.ipynb, timing_waveforms.py/.ipynb, validations_comparisons_glow2.ipynb
tests/                  test_glow_bilby.py, test_glow_pycbc.py
REVIEW.md               issues found in the original glowpe code and what changed
```

This is a clean, glow2-only rewrite of `package/glowpe` (`sources.py` + `default_priors_injections.py`). The bilby and pycbc layers contain no lens physics of their own: both call `glow_waveforms.lens_models.apply_lensing`. The tests check that they apply the identical F*(w), and that the bilby source-function defaults equal `DEFAULT_LENS_PARAMS`.

## Install
glow2 is **not on PyPI**, so install it first:
```bash
pip install git+<GLOW2_GIT_URL>                    # or: pip install /path/to/glow2
pip install "glow_waveforms[bilby]"     # bilby layer
pip install "glow_waveforms[pipe]"      # + bilby_pipe
pip install "glow_waveforms[pycbc]"     # pycbc layer
pip install "glow_waveforms[all,test]"  # everything + pytest; add -e for development
```
The core needs only numpy and glow2. If glow2 is missing, pip fails with `No matching distribution found for glow2`.

Tested with bilby 2.6, bilby_pipe 1.6, pycbc 2.10 and glow2 0.2. 

## Models
The same parameter names are used everywhere: the bilby signatures, the priors, the ini files, the pycbc keyword arguments and the injections. The defaults are `glow_waveforms.lens_models.DEFAULT_LENS_PARAMS`.

| key | lens parameters (defaults) | default prior (bilby) |
|---|---|---|
| `UL` | none | none |
| `PL` | `MLz=1e3, y=0.2` (analytic `Fw_PL`) | MLz ~ LogU(10, 1e6), y ~ PowerLaw(α=1) on [0.05, 5], i.e. p(y) ∝ y |
| `SIS` | `MLz=1e3, psi0=1, y=0.2` | MLz, y; psi0 = 1 |
| `gSIS` | `MLz=1e3, psi0=1, y=0.2, k=1` | + k ~ U(0.5, 1.5) |
| `NFW` | `MLz=1e3, psi0=2, y=0.2, xs=1` | + psi0 ~ U(0.1, 5); xs = 1 |
| `PLshear` | `MLz=1e3, psi0=1, y=1.2, theta=0.2, g1=0.5, g2=0, kp=0` | theta ~ U(0, π/2), g1 ~ U(0, 0.8); g2 = kp = 0 |
| `PLshear_kpeqg1` | `…, g1=0.4, g2=0` (κ = γ₁) | g1 ~ U(0, 0.49) |
| `gSISshear` | `…, k=1, g1=0.2, g2=0, kp=0` | PLshear priors, k = 1 |

w = 8πG M_Lz f/c³. Each waveform is multiplied by F*(w) in the band where the strain is non-zero.

### Shear models: `(y, θ)` are lens-plane coordinates
`PLshear`, `PLshear_kpeqg1` and `gSISshear` keep the `glowpe` convention: the source is at the origin and the lens at **c** = y(cos θ, sin θ). glow2 centres the external shear/convergence on the origin, so:

- **c is the microlens offset from the unperturbed macro-image, in the lens plane**, in units of the microlens Einstein radius.
- The source-plane offset, with the external field centred on the lens as in Mishra et al., is **u** = −M**c**, with M = (1−κ)𝟙 − Γ and |det M| = 1/|μ_macro|. Convert with `source_position_from_lens_offset` and `lens_offset_from_source_position`. For an isolated point lens the two frames coincide.
- The default prior p(y) ∝ y with θ uniform is area-uniform, so it is uniform in both frames; it corresponds to a constant microlens surface density.

**glow2 edge cases.** F(w) fails (log L = −∞) at exactly θ = 0 for κ > 0, and at exactly y = 1 for the SIS. glow2 can segfault at exactly θ = π/2. All three have zero measure in PE.

## bilby
```python
import bilby
from glow_waveforms.glow_bilby.source import LENSED_WAVEFORMS
from glow_waveforms.glow_bilby.bilby_pe.priors import get_priors

wfg = bilby.gw.WaveformGenerator(
    duration=4, sampling_frequency=2048,
    frequency_domain_source_model=LENSED_WAVEFORMS['SIS'],
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(waveform_approximant='IMRPhenomXPHM', reference_frequency=20., minimum_frequency=20.))
priors = get_priors('SIS', trigger_time=1126259642.413)
```
- **Function names:** `lensed_binary_black_hole_<model>` uses lalsimulation and `lensed_gwsignal_binary_black_hole_<model>` uses gwsignal. The names contain `binary_black_hole`, so bilby_pipe automatically picks the BBH conversion and generation functions.
- **Priors must cover every signature parameter.** A fixed value is fine, and the shipped priors already do this. bilby's `WaveformGenerator` passes only named parameters and raises `KeyError` if one is missing.
- **Reusing an unlensed waveform** (e.g. for mismatch grids): `lens_polarizations(model, frequency_array, unlensed_pols, **lens_params)`. bilby's lal sources need `frequency_array` to start at 0 with uniform spacing.
- **Full injection/recovery** (zero noise, H1/L1/V1, SNR 30, dynesty):
  ```bash
  python -m glow_waveforms.glow_bilby.bilby_pe.run_bilby PLshear SIS outdir/PLshear_SIS --npool 16
  python -m glow_waveforms.glow_bilby.bilby_pe.run_bilby PLshear_kpeqg1 PLshear_kpeqg1 outdir/kpeqg1 --lens-override g1=0.4
  ```

## bilby_pipe
```bash
python -m glow_waveforms.glow_bilby.bilby_pipe_pe.make_ini PLshear_kpeqg1 PLshear_kpeqg1 --snr 30
python -m glow_waveforms.glow_bilby.bilby_pipe_pe.make_ini PLshear SIS --snr 30 --ini-dir inis/
bilby_pipe PLshear_kpeqg1_inj_PLshear_kpeqg1_rec.ini --submit
```
The package must be importable in every environment that bilby_pipe jobs run in. The ini settings:
- **`injection-dict`:** the default lens parameters, with `d_L` rescaled to the target SNR by `make_ini`.
- **Source models:** `injection-frequency-domain-source-model` and `frequency-domain-source-model = glow_waveforms.glow_bilby.source.lensed_binary_black_hole_<model>`.
- **`prior-file`:**
 default prior files at `glow_waveforms/glow_bilby/bilby_pipe_pe/prior_files/<model>.prior`.
  
- **Prior class and conversion:** `default-prior = glow_waveforms.glow_bilby.bilby_pe.priors.LensedBBHPriorDict` and `conversion-function = glow_waveforms.glow_bilby.bilby_pe.priors.convert_to_lensed_bbh_parameters`. Both are needed only for a `mu` Constraint.
- **`--duration` / `--post-trigger-duration`:** must contain the micro-image time delays (relevant for large MLz).
- **Recovery models:** in the example ini the other models are listed as commented `prior-file` / `frequency-domain-source-model` pairs.

## pycbc
Installing the package registers the approximants as pycbc waveform plugins (entry point `pycbc.waveform.fd`). Without installing, call `glow_waveforms.glow_pycbc.register()`.
```python
import numpy as np
from pycbc.waveform import get_fd_waveform

hp, hc = get_fd_waveform(approximant='GLoW_PLshear_kpeqg1', base_approximant='IMRPhenomXPHM',
                         mass1=36, mass2=29, distance=410, f_lower=20, delta_f=1/64,
                         MLz=1e3, y=0.8, theta=np.pi/6, g1=0.2)
```
- **Approximants:** `GLoW_PL`, `GLoW_SIS`, `GLoW_gSIS`, `GLoW_NFW`, `GLoW_PLshear`, `GLoW_PLshear_kpeqg1`, `GLoW_gSISshear`.
- **Arguments:** the usual pycbc arguments go to `base_approximant` (default IMRPhenomXPHM); the lens parameters are as in the table.
- **Lensing existing waveforms:** `glow_waveforms.glow_pycbc.lens_frequency_series(model, hp, hc, **lens_params)`.
- **Failures:** if F(w) fails, `GLoWAmplificationError` is raised. The bilby layer instead returns `None`, which bilby treats as log L = −∞.

## Examples and validation
- **`examples/example_glow_bilby.ipynb`:** waveforms, F(w), a bilby likelihood, and pointers to bilby_pipe.
- **`examples/example_glow_pycbc_mismatch.ipynb`:** the `GLoW_<model>` pycbc approximants: lensed waveforms of a GW150914-like binary and the mismatch against M_Lz between UL and PL, and UL and PLshear_kpeqg1 (`optimized_match`, runs in about 20 s).
- **`examples/timing_waveforms.ipynb`:** time per `frequency_domain_strain` call for prior draws. This takes about 10–35 ms per model (4 s, 2048 Hz, IMRPhenomXPNR); see the notebook for the table.
- **`examples/validations_comparisons_glow2.ipynb`:** reproduces Gravelamps fig 1, GLWORIA fig 3 and Mishra et al. figs 1–2 with glow2 only. It checks the numerical F(w) against:
  - `Fw_PL` / `Fw_SIS` (median relative difference 4e-5 to 5e-3);
  - the geometric-optics limit at w ≫ 1;
  - an independent source-displaced set-up for the shear models (2e-7 to 4e-4).

