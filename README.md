# Information-Dense Synthesis for Molecular Discovery

Code for reproducing results in the paper: [Information-Dense Synthesis for Molecular Discovery](https://doi.org/10.48550/arXiv.2610.08495)

## Installation

You can clone this repository by running:
```bash
git clone --recurse-submodules https://github.com/kkj15dk/LIDS.git
```

Afterwards, the python environment can be set up with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```
This requires Linux with an NVIDIA GPU and driver version 580 or newer (PyTorch and JAX are installed as CUDA 13 builds).

### Submodules (if you did not use `--recurse-submodules` when cloning)

We use git submodules to handle the LOCK kernel, and ProGen2 oracle. 
To initialize the submodules, run:
```bash
git submodule update --init --recursive
```

#### ProGen2 weights

To download the weights for progen (progen2-base):
```bash
wget -P progen2-base https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz
tar -xvf progen2-base/progen2-base.tar.gz -C progen2-base
```

## Reproducing results

The main entrypoint is [run_rounds.py](run_rounds.py), which uses the config set in [conf/config.yaml](conf/config.yaml).

### Bash scripts

Every simulation in the paper has a corresponding bash script in [scripts/](scripts/). For example, to run LIDS on a sparse activity map at low noise-level, run:
```bash
bash scripts/run_rounds_lifted_categorical_bump10-4.sh
```
Be aware that running all scripts will be time-consuming, especially for the PLM oracle.

### Plotting

After running any subset of the simulations, the results can be plotted with [plot_MAP_over_rounds.py](plot_MAP_over_rounds.py) by:
```bash
uv run plot_MAP_over_rounds.py --average --ci
```
The plotting scripts render text with LaTeX, so they require a LaTeX installation.

### wandb

If you want to use wandb for logging, you need to provide an API key.
You can do so by creating a file `secret.txt` in the repository root with the key as its only content. 
By default wandb logging is disabled, it can be enabled by setting `use_wandb: true` in the config.

# Cite
If you find this work useful, please cite our preprint:
```
@misc{jakobsen2026informationdensesynthesismoleculardiscovery,
      title={Information-Dense Synthesis for Molecular Discovery}, 
      author={Kasper K. Jakobsen and Eli N. Weinstein},
      year={2026},
      eprint={2610.08495},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2610.08495}, 
}
```