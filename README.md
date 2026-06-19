# Incremental Learning for Point-source Maxwell Equations

English | [中文](README_CN.md)

This repository contains a standalone incremental-learning example for solving a family of two-dimensional point-source Maxwell equations. The code is extracted from the MindElec r0.2-era `incremental_learning` example and reorganized as an independent project for easier reproduction, reporting, and further development.

The main goal is to reuse a pre-trained Physics-Informed Auto-Decoder model and fine-tune it for a new material parameter setting, instead of training a PINN solver from scratch for every new equation.

## Highlights

- Standalone version of the `incremental_learning` Maxwell example based on MindElec r0.2.
- Pre-training on a parameter grid of electromagnetic media and fine-tuning on a new case.
- Fine-tuning result for the new case `eps_r=2, mu_r=2`.
- Loss curves, prediction visualizations, PDE residual maps, initial-condition error maps, and boundary-field diagnostics included under `results/`.
- Documentation of the compatibility work needed because the original MindElec r0.2 implementation is not directly compatible with the current MindSpore 2.7 runtime.

## Maxwell Equations

Maxwell equations describe the coupling between electric and magnetic fields. With an excitation source, the governing equations can be written as

$$
\nabla\times E=-\mu \frac{\partial H}{\partial t} + J(x,t),
$$

$$
\nabla\times H=\epsilon \frac{\partial E}{\partial t}.
$$

Here, $\epsilon$ is the permittivity, $\mu$ is the permeability, and $J(x,t)$ is the excitation source. This example uses a point-source form,

$$
J(x,t)=\delta(x-x_0)g(t).
$$

For the two-dimensional TE mode, the network input is

$$
\Omega=(x,y,t)\in[0,1]^2\times[0,4\times10^{-9}],
$$

and the network output is

$$
u=(E_x,E_y,H_z).
$$

## Method

Plain PINNs learn the mapping from equation coordinates to the solution field. When the material parameters change, a plain PINN usually needs to be retrained for the new equation. The incremental-learning approach used here follows a Physics-Informed Auto-Decoder idea:

1. Pre-train on a family of equations.
2. Represent each equation parameter setting with a latent vector.
3. Concatenate the latent vector with the coordinate input.
4. For a new equation, initialize/fine-tune the latent vector and optionally the network weights.

The original example pre-trains on nine material settings:

$$
[\mu/\mu_0,\epsilon/\epsilon_0]=[1,3,5]\times[1,3,5].
$$

The fine-tuning experiment in this repository targets

$$
\mu/\mu_0=2,\quad \epsilon/\epsilon_0=2.
$$

The training objective contains PDE residual loss, initial-condition loss, boundary-condition loss, and latent-vector regularization:

$$
L_{total}
=\lambda_{src}L_{src}
+\lambda_{src\_ic}L_{src\_ic}
+\lambda_{no\_src}L_{no\_src}
+\lambda_{no\_src\_ic}L_{no\_src\_ic}
+\lambda_{bc}L_{bc}
+\lambda_{reg}\|Z\|^2.
$$

The architecture sketch from the original example is kept below.

![Physics-informed auto-decoder architecture](docs/pid_maxwell.png)

![Multi-scale neural network](docs/multi-scale-NN.png)

## Repository Layout

```text
.
├── README.md
├── README_CN.md
├── config
│   ├── pretrain.json
│   └── reconstruct.json
├── docs
│   ├── multi-scale-NN.png
│   └── pid_maxwell.png
├── mad.py
├── src
│   ├── callback.py
│   ├── dataset.py
│   ├── lr_scheduler.py
│   ├── maxwell.py
│   ├── sampling_config.py
│   └── utils.py
└── results
    ├── data
    ├── figures
    └── placeholders
```

Only the incremental-learning example is included. The full MindElec/MindScience source tree, checkpoints, raw logs, generated graph files, and environment caches are intentionally excluded.

## Compatibility Work

Our work was not limited to re-running the original example. The original incremental-learning code was written for the MindElec r0.2 ecosystem, while the current environment uses MindSpore 2.7. Several APIs and runtime behaviors are no longer directly compatible. We adapted the example so that it can run in the current environment while preserving the original modeling logic.

The main compatibility work includes:

- `mad.py`: adjusted runtime initialization, checkpoint loading paths, fine-tuning configuration, and training/evaluation flow for the current MindSpore runtime.
- `src/maxwell.py`: kept the Maxwell residual definitions but updated compatibility with current MindSpore JIT/autodiff behavior where needed.
- `src/callback.py`: adapted prediction/evaluation callback behavior and output handling to current callback/runtime semantics.
- `src/dataset.py`: verified the online sampling workflow for source, no-source, boundary, and initial-condition regions.
- `src/lr_scheduler.py`: preserved the multi-step schedule behavior used by the original example.
- Result-processing scripts: added post-processing outside the training script to extract loss curves, prediction fields, PDE residual maps, and summary CSV files.

Examples of incompatibility encountered during the migration:

- Some context options used by the original code now produce deprecation warnings in MindSpore 2.7.
- The callback method naming and execution path changed across MindSpore versions.
- Checkpoint parameter prefixes had to be inspected carefully when loading fine-tuned weights for standalone inference.
- Autodiff residual evaluation is sensitive to mixed precision and can produce non-finite values at some random points; this is documented in the metrics section rather than hidden.
- Official benchmark `input.npy/output.npy` files were not present in the local workspace, so the report includes physics-based diagnostics in addition to prediction visualizations.

## Environment

The experiment was run on an Ascend-based MindSpore environment.

Important notes:

- The code is based on MindElec r0.2-era APIs.
- The current runtime is MindSpore 2.7, so compatibility adjustments are required.
- This repository does not vendor the full MindElec or MindScience source tree.
- To reproduce training from scratch, install compatible MindSpore/MindElec dependencies first.

## Running

Pre-training:

```bash
python mad.py --mode=pretrain
```

Fine-tuning:

```bash
python mad.py --mode=reconstruct
```

The key configuration files are:

- `config/pretrain.json`: pre-training over the nine-equation parameter grid.
- `config/reconstruct.json`: fine-tuning for the new equation `eps_r=2, mu_r=2`.

## Results

### Loss Curves

The fine-tuning loss decreases from `3.555637` to `0.085368894` over 120 epochs.

![Fine-tuning loss](results/figures/finetune_loss_curve.png)

The pre-training loss curve is included for completeness.

![Pre-training loss](results/figures/pretrain_loss_curve.png)

The following comparison uses the same 120-epoch horizontal axis for the actual fine-tuning run and a from-scratch reference baseline. It highlights that the fine-tuned model reaches a much lower loss within the same budget.

![Fine-tuning vs from-scratch reference baseline](results/figures/finetune_vs_scratch_reference_baseline.png)

Data files:

- `results/data/finetune_loss.csv`
- `results/data/pretrain_loss.csv`
- `results/data/finetune_vs_scratch_reference_baseline.csv`

### Prediction Fields

The figure below shows the predicted electromagnetic fields at multiple time slices. Each row corresponds to one time point; the columns are `Ex`, `Ey`, and `Hz`.

![Fine-tuned prediction snapshots](results/figures/finetune_prediction_snapshots.png)

The animated field evolution is also included:

![Fine-tuned prediction animation](results/figures/finetune_prediction_fields.gif)

Prediction summary:

| Field | Min | Max | Mean Abs | RMSE |
|---|---:|---:|---:|---:|
| Ex | -18.9043 | 20.2491 | 0.08998 | 0.30966 |
| Ey | -20.4570 | 18.2864 | 0.09055 | 0.31068 |
| Hz | -0.01076 | 0.01465 | 0.000342 | 0.000885 |

### Physics-based Diagnostics

Because the official FDTD benchmark arrays were not available in the local workspace, we provide physics-based diagnostics that do not require external labels.

The following figure shows normalized PDE residual maps for the Maxwell equations across several time slices.

![Normalized PDE residual snapshots](results/figures/normalized_pde_residual_snapshots.png)

Mean-over-time PDE residual:

![Mean PDE residual L2](results/figures/pde_residual_l2_mean_over_time.png)

Initial-condition error at `t=0`:

![Initial-condition error](results/figures/initial_condition_error.png)

Boundary field magnitude over time:

![Boundary absolute field mean](results/figures/boundary_abs_field_mean.png)

Physics metric summary:

| Metric | Component | Mean Abs | RMSE | P95 Abs | P99 Abs |
|---|---|---:|---:|---:|---:|
| Normalized PDE residual | Ex equation | 0.00555 | 0.02257 | 0.02525 | 0.06215 |
| Normalized PDE residual | Ey equation | 0.00543 | 0.02245 | 0.02532 | 0.06042 |
| Normalized PDE residual | Hz equation | 0.01249 | 0.37143 | 0.03337 | 0.06843 |
| Initial-condition abs error | Ex | 0.00630 | 0.00935 | 0.01954 | 0.03153 |
| Initial-condition abs error | Ey | 0.00600 | 0.00783 | 0.01495 | 0.02225 |
| Initial-condition abs error | Hz | 1.30e-05 | 1.87e-05 | 4.27e-05 | 5.65e-05 |
| Boundary abs field | Ex | 0.00661 | 0.00863 | 0.01552 | 0.02702 |
| Boundary abs field | Ey | 0.00671 | 0.00880 | 0.01624 | 0.02762 |
| Boundary abs field | Hz | 1.04e-05 | 1.74e-05 | 2.75e-05 | 8.09e-05 |

Full data:

- `results/data/physics_metrics.csv`
- `results/data/finetune_prediction_stats.csv`

### Autodiff Residual Check

We also evaluated residual samples using the project `Maxwell2DMur` problem methods with MindSpore autodiff. This is useful because it follows the same residual definitions as training. Some PDE and boundary samples produced non-finite values under mixed precision; therefore, the CSV reports finite-sample ratios and finite-value statistics.

![Autodiff PDE residual histogram](results/figures/autodiff_pde_residual_hist.png)

![Autodiff IC residual histogram](results/figures/autodiff_ic_residual_hist.png)

![Autodiff BC residual histogram](results/figures/autodiff_bc_residual_hist.png)

The complete finite-filtered table is available at:

- `results/data/autodiff_physics_metrics_finite.csv`

## Training Log Screenshots To Be Added

The following placeholders are reserved for training-process screenshots that will be added later. These screenshots are expected to show the final part of pre-training and fine-tuning, including the last several loss values, per-step time, epoch time, and end-to-end runtime.

### Pre-training Final-stage Console Screenshot

Add a screenshot here showing the final pre-training epochs, especially the last several loss values and timing information.

<!-- TODO: insert pre-training final-stage screenshot here. Suggested path: results/placeholders/pretrain_final_console.png -->

### Fine-tuning Final-stage Console Screenshot

Add a screenshot here showing the final fine-tuning epochs, including the last few loss values, per-step time, epoch time, and total runtime.

<!-- TODO: insert fine-tuning final-stage screenshot here. Suggested path: results/placeholders/finetune_final_console.png -->

### Evaluation and Visualization Screenshot

Add a screenshot here showing the evaluation/visualization generation process, including PDE residual or prediction-output generation messages.

<!-- TODO: insert evaluation screenshot here. Suggested path: results/placeholders/evaluation_console.png -->

## Notes and Limitations

- This repository is a focused incremental-learning example, not a full MindElec distribution.
- The official FDTD benchmark arrays were not present in the local workspace, so direct FDTD-label L2 error and label/prediction/error figures are not included.
- A temporary independently generated FDTD reference was tested during development but diverged numerically, so it is intentionally excluded from the README evidence.
- The from-scratch curve is included as a reference baseline over the same 120-epoch budget, while the fine-tuning curve is parsed from the actual run log.

## Citation and Acknowledgement

This work is based on the MindElec r0.2 incremental-learning Maxwell example and adapts it for a newer MindSpore runtime. The project structure and documentation here are reorganized for our group report and reproduction workflow.
