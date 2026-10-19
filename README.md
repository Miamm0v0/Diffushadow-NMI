# DiffuShadow

> **Repository status: work in progress.** This repository is being assembled
> from the research codebase and is still being cleaned and organized.
> The current snapshot contains the main model and selected training,
> generation, evaluation, and plotting code, but it is **not yet a complete or
> end-to-end reproducibility package**. Files, dependencies, and documentation
> are still being checked and organized.

## Scope

The code in this snapshot studies a diffusion-based model for generating
quantum-measurement sequences. The material currently collected here focuses
on the ordered-sequence (`oseq`) representation and its rotary-position-
embedding (`oseq_rope`) model. The evaluation code also contains routines for
the transverse-field Ising (TFI), Heisenberg, XXZ, J1-J2, and ANNNI models.

This repository was extracted from a larger experimental repository.
Consequently, some scripts still contain branches and imports for exploratory
or ablation variants that have not yet been copied into this snapshot.

## Files currently included

| File | Current role |
| --- | --- |
| `Qdmodel_oseq_rope.py` | PyTorch definition of the ordered-sequence diffusion model with RoPE. |
| `train_oseq_2.py` | Training entry point, dataset loaders, masking process, losses, checkpoint loading, and learning-rate scheduling. It includes both 1D and experimental 2D branches. |
| `eval_gen_mea_data.py` | Generates ordered measurement samples from a trained checkpoint and saves them as JSON. |
| `eval_new.py` | Main sampling and observable-evaluation script for several quantum models. |
| `eval_cal_exact.py` | Computes and caches exact-reference observables using dense or sparse diagonalization routines. |
| `eval_utils.py` | Hamiltonian construction, observable estimators, exact-value utilities, and median-of-means estimators. |
| `draw_fig_properties.py` | Plots observables from saved evaluation `.npz` files, optionally together with exact-reference caches. |
| `test_1.py` | Sampling/decoding helper functions used by the generation and evaluation scripts; despite the historical filename, this is not a unit-test suite. |

No datasets, trained checkpoints, generated result files, or paper figures are
included in the current snapshot.

## Current limitations

The repository is still being cleaned for release. In particular:

- Several local modules referenced by the collected scripts are not present
  yet. These include legacy/ablation model definitions such as
  `Qdmodel_oseq.py`, `Qdmodel_nseq.py`, `Qdmodel_nseq_rope.py`, the 2D and GNN
  variants, `Qmultimodel.py`, and exact-evaluation helpers
  `eval_data_heisenberg.py` and `eval_data_xxz.py`.
- Because those imports currently occur at module load time, the training and
  evaluation entry points should not yet be expected to run from a fresh clone,
  even when only `oseq_rope` is requested.
- Environment versions have not yet been pinned, and a `requirements.txt` or
  environment file has not yet been added.
- Some filenames, comments, default paths, and command-line options still
  reflect the internal research repository and will be normalized.
- Reproduction commands and the mapping from paper experiments to
  configurations/checkpoints are still being prepared.

Until these items are resolved, this repository should be treated as a source
snapshot under active cleanup rather than a turnkey reproduction package.

## Data convention used by the included 1D code

The 1D training and generation scripts use JSON files containing a list of
numeric rows. Each row has the ordered form

```text
[condition, P1, b1, P2, b2, ..., PN, bN]
```

where `condition` is the scalar physical parameter associated with the sample,
and each `(Pi, bi)` pair stores the measurement basis and outcome for one
qubit. `eval_gen_mea_data.py` writes this layout, and `train_oseq_2.py` reads it.

The training script also contains support for an experimental 2D JSON format
named `tfi2d_grid_v1`; the corresponding 2D model definitions are not included
in this snapshot.

## Software dependencies

The direct third-party imports in the current files are:

- PyTorch
- Transformers (for Llama rotary embeddings)
- NumPy
- SciPy
- Matplotlib
- tqdm

Exact supported versions and installation instructions will be added after the
environment has been validated from a clean setup. CUDA is optional for parts
of the code but is expected for the larger training and sampling runs.

## Intended workflow after cleanup

The intended reproducibility workflow is:

1. Prepare measurement data in the ordered JSON format.
2. Train the `oseq_rope` model with `train_oseq_2.py`.
3. Generate samples or evaluate observables with `eval_gen_mea_data.py` and
   `eval_new.py`.
4. Compute exact-reference caches with `eval_cal_exact.py`.
5. Plot comparisons with `draw_fig_properties.py`.

Validated commands will be documented here once all required files and
artifacts have been organized.

## Planned repository updates

- Add or remove unresolved local dependencies so the documented workflow runs
  from a clean clone.
- Add a pinned environment specification and installation instructions.
- Add representative data/configuration examples and checkpoint information.
- Document the commands corresponding to the main experiments.
- Add validation checks and clarify the expected outputs of each stage.
- Finalize filenames, comments, directory layout, and licensing information.

This README will be updated as the repository is completed.
