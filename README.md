# PX-SGMCMC

[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue.svg)](https://openreview.net/forum?id=exgLs4snap)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Code for the paper [Parameter Expanded Stochastic Gradient Markov Chain Monte Carlo (ICLR 2025)](https://openreview.net/forum?id=exgLs4snap).

## Setup

We tested the following minimal environment with Python 3.12:

```bash
pip install -U pip setuptools wheel tabulate
pip install -U "jax[cuda12]" # 0.5.1
pip install -U "flax[all]" # 0.10.3
```

## Experiment

To run the SGHMC and PX-SGHMC _without_ data augmentation (as described in Section 5.2.1), use the following command lines.
For results _with_ data augmentation (as described in Section 5.2.2), set `--data_augmentation simple` and `--posterior_temperature 0.01`.

```bash
# SGHMC
python scripts/run_sghmc.py \
  --data_name cifar10 --data_augmentation none \
  --num_samples 100 --num_updates 5000 --num_batch 256 \
  --step_size 0.0003 --step_size_min 0.0 \
  --posterior_temperature 1.0 --prior_variance 0.05 --friction 100.0 \
  --seed 42 --save save/cifar10/sghmc/42/
```

```bash
# PX-SGHMC (c=1, d=1)
python scripts/run_pxsghmc.py \
  --data_name cifar10 --data_augmentation none \
  --num_samples 100 --num_updates 5000 --num_batch 256 \
  --step_size 0.0001 --step_size_min 0.0 \
  --posterior_temperature 1.0 --prior_variance 0.02 --friction 100.0 \
  --px_friction 1.0 --px_l2_regularizer 2.0 \
  --seed 42 --save save/cifar10/pxsghmc/42/
```

We executed the above command lines on a single RTX A6000 machine, with each run taking approximately 2 hours, and obtained the following results: PX-SGHMC outperforms SGHMC in both classification error (`val/ens_err`) and negative log-likelihood (`val/ens_nll`), owing to the ensemble gain from improved ensemble diversity (`val/ens_emb`).

- [`save/cifar10/sghmc/42/console.log`](save/cifar10/sghmc/42/console.log)
  ```
  (...)
  [Sample    100/   100] (...) val/ens_err: 1.343e-01, val/ens_nll: 4.212e-01, val/ens_amb: 1.878e-01
  ```

- [`save/cifar10/pxsghmc/42/console.log`](save/cifar10/pxsghmc/42/console.log)
  ```
  (...)
  [Sample    100/   100] (...) val/ens_err: 1.185e-01, val/ens_nll: 3.763e-01, val/ens_amb: 2.408e-01
  ```

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{kim2025parameter,
  title     = {Parameter Expanded Stochastic Gradient Markov Chain Monte Carlo},
  author    = {Hyunsu Kim and Giung Nam and Chulhee Yun and Hongseok Yang and Juho Lee},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=exgLs4snap},
}
```