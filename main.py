from typing import Sequence, Union, Tuple, Dict, Optional
import os
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import ProfilerActivity
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from omegaconf import DictConfig, OmegaConf

import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist_np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker

import wandb

from eval import (
    compute_top_k_true_metrics,
    top_k_fitness,
    visualize_umap_discrete,
    visualize_tau_and_pi,
    visualize_function,
)
from experiment import Experiment, thompson_samples_to_experiment
from function import GLM, NormalGLM, one_hot_decode, FitnessFunction
from model import ExperimentCategorical

from MCMC import (
    predictive,
    construct_posterior_samples,
    _build_weights_dist,
    load_theta_gpu_splits,
    GPUThetaBatches,
    construct_observationblock_from_experiment,
    visualize_graphical_model,
)


def _resolve_duplicate_index(cfg: DictConfig) -> int:
    sim_cfg = cfg.get("simulation", None)
    if sim_cfg is None:
        return 0
    try:
        return int(sim_cfg.theta_index)
    except Exception:
        return 0


def _compose_round_seed(base_seed: int, duplicate_idx: int, round_idx: int) -> int:
    # Keep it simple and explicit: each component influences the final seed.
    return int(base_seed) + 1_000 * int(duplicate_idx) + 10_000 * int(round_idx)


def _compose_true_theta_seed(base_seed: int, duplicate_idx: int) -> int:
    # Do not depend on round so the cached true theta stays consistent.
    return int(base_seed) + 1_000 * int(duplicate_idx)


def _checkpoint_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoints")

def _latest_checkpoint_path(run_dir: str) -> str:
    return os.path.join(_checkpoint_dir(run_dir), "latest.pt")


def _best_checkpoint_path(run_dir: str) -> str:
    return os.path.join(_checkpoint_dir(run_dir), "best.pt")


def _true_theta_path(project_path: str) -> str:
    return os.path.join(project_path, "true_theta.npz")

def _sample_log_w_from_weights(
    weights_dist: Union[float, dist_np.Distribution],
    num_components: int,
    seed: int,
) -> np.ndarray:
    if isinstance(weights_dist, (float, int)):
        gamma_np = np.full((num_components,), float(weights_dist), dtype=np.float32)
    elif isinstance(weights_dist, dist_np.Distribution):
        # Covers dist_np.Gamma as well as composite priors built on top of it
        # (e.g. a shifted Gamma via TransformedDistribution).
        key = jax.random.PRNGKey(int(seed))
        gamma_n = weights_dist.sample(key, sample_shape=(num_components,))  # type: ignore[arg-type]
        gamma_np = np.asarray(jax.device_get(gamma_n), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported weights distribution type: {type(weights_dist)}")

    if not np.all(gamma_np > 0):
        raise ValueError("Sampled weights must be positive to take log.")
    return np.log(gamma_np).astype(np.float32)


def _load_or_create_true_theta(
    project_path: str,
    val_dl: GPUThetaBatches,
    theta_index: int,
    device: str,
    true_weights_dist: Optional[Union[float, dist_np.Distribution]] = None,
    true_num_components: Optional[int] = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, GPUThetaBatches]:
    """Create campaign-level true theta from the held-out validation partition only."""
    path = _true_theta_path(project_path)

    if os.path.exists(path):
        fixed = np.load(path)
        log_w_true = torch.as_tensor(fixed["log_w_true"], dtype=torch.float32, device=device)
        log_v_true = torch.as_tensor(fixed["log_v_true"], dtype=torch.float32, device=device)
        if "mode_true" in fixed:
            mode_true = torch.as_tensor(fixed["mode_true"], dtype=torch.long, device=device)
        else:
            mode_true = torch.argmax(log_v_true, dim=-1)
        return log_w_true, log_v_true, mode_true, path, val_dl

    val_size = len(val_dl)
    if not (0 <= int(theta_index) < val_size):
        raise ValueError(
            f"true_theta_index must be in [0, {val_size - 1}] for the held-out validation split; "
            f"got {theta_index}."
        )

    log_w_true_t, log_v_true_t, mode_true_t = val_dl[int(theta_index)]
    if true_num_components is None:
        true_num_components = int(log_v_true_t.shape[0])
    if true_num_components < 1:
        raise ValueError(f"true_num_components must be >= 1, got {true_num_components}")
    if true_num_components > int(log_v_true_t.shape[0]):
        raise ValueError(
            f"true_num_components={true_num_components} exceeds available components "
            f"{int(log_v_true_t.shape[0])} in the sampled theta."
        )

    log_w_true_t = log_w_true_t[:true_num_components]
    log_v_true_t = log_v_true_t[:true_num_components]
    mode_true_t = mode_true_t[:true_num_components]
    if true_weights_dist is None:
        log_w_true_np = log_w_true_t.detach().cpu().numpy().astype(np.float32)
    else:
        log_w_true_np = _sample_log_w_from_weights(
            weights_dist=true_weights_dist,
            num_components=true_num_components,
            seed=seed,
        )
    log_v_true_np = log_v_true_t.detach().cpu().numpy().astype(np.float32)
    mode_true_np = mode_true_t.detach().cpu().numpy().astype(np.int64)

    np.savez(
        path,
        log_w_true=log_w_true_np,
        log_v_true=log_v_true_np,
        mode_true=mode_true_np,
        theta_index=np.asarray(theta_index),
    )

    log_w_true = torch.as_tensor(log_w_true_np, dtype=torch.float32, device=device)
    log_v_true = torch.as_tensor(log_v_true_np, dtype=torch.float32, device=device)
    mode_true = torch.as_tensor(mode_true_np, dtype=torch.long, device=device)

    filtered_val_dl = val_dl.exclude(int(theta_index))
    return log_w_true, log_v_true, mode_true, path, filtered_val_dl


def _ensure_true_theta_exists(
    project_path: str,
    run_dir: str,
    run_seed: int,
    true_theta_seed: int,
    sequence_length: int,
    vocab_size: int,
    num_samples_prior: int,
    num_samples_eval: int,
    num_components: int,
    weights_dist: Union[float, dist_np.Distribution],
    sigma: float,
    overlap_reg_strength: float,
    true_weights_dist: Union[float, dist_np.Distribution],
    true_num_components: int,
    true_theta_index: int,
    device: str,
) -> None:
    """
    Create project_path/true_theta.npz, if it doesn't already exist.
    """
    if os.path.exists(_true_theta_path(project_path)):
        return

    glm = NormalGLM()
    key = jax.random.PRNGKey(int(run_seed))
    key, predictive_key = jax.random.split(key)

    theta_path = os.path.join(run_dir, "prior_samples.npz")
    if os.path.exists(theta_path):
        print(f"Theta samples file {theta_path} already exists. Loading existing samples...", flush=True)
    else:
        num_theta_samples = int(num_samples_prior) + int(num_samples_eval)
        start_time = time.time()
        print(f"Sampling {num_theta_samples} prior samples for true_theta initialization (this may take a while)...", flush=True)
        predictive(
            key=predictive_key,
            N=int(num_components),
            L=int(sequence_length),
            K=int(vocab_size),
            gamma=weights_dist,
            inverse_link_function=glm.inverse_link_function,
            sigma=float(sigma),
            overlap_reg_strength=float(overlap_reg_strength),
            path=theta_path,
            num_samples=num_theta_samples,
        )
        print(f"Saved theta samples to {theta_path} (took {time.time() - start_time:.1f}s)", flush=True)

    _, val_dl = load_theta_gpu_splits(path=theta_path, test_size=int(num_samples_eval), device=device)
    _, _, _, true_theta_file, _ = _load_or_create_true_theta(
        project_path=project_path,
        val_dl=val_dl,
        theta_index=true_theta_index,
        device=device,
        true_weights_dist=true_weights_dist,
        true_num_components=true_num_components,
        seed=true_theta_seed,
    )
    print(f"Created campaign true theta at: {true_theta_file}")


def _prev_round_theta_path(project_path: str, round_idx: int) -> str:
    prev_round = round_idx - 1
    if prev_round < 1:
        raise ValueError(f"No previous round exists for round {round_idx}")

    return os.path.join(project_path, f"round_{prev_round}_obs", "posterior_samples_conditioned.npz")

def _save_checkpoint(
    run_dir: str,
    epoch: int,
    step: int,
    eig_model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    best_eval_loss: float,
    best_eval_epoch: int,
    is_best: bool = False,
) -> str:
    ckpt_dir = _checkpoint_dir(run_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "best_eval_loss": float(best_eval_loss),
        "best_eval_epoch": int(best_eval_epoch),
        "eig_model": eig_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }

    step_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    latest_path = _latest_checkpoint_path(run_dir)
    torch.save(state, step_path)
    torch.save(state, latest_path)
    if is_best:
        torch.save(state, _best_checkpoint_path(run_dir))
    return latest_path

def _maybe_resume_from_checkpoint(
    run_dir: str,
    eig_model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    device: str,
) -> tuple[int, int, float, int]:
    ckpt_path = _latest_checkpoint_path(run_dir)
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Starting from step 0.")
        return 0, 0, float("inf"), -1

    checkpoint = torch.load(ckpt_path, map_location=device)
    eig_model.load_state_dict(checkpoint["eig_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    scheduler_state = checkpoint.get("lr_scheduler", None)
    if lr_scheduler is not None and scheduler_state is not None:
        lr_scheduler.load_state_dict(scheduler_state)

    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    checkpoint_step = int(checkpoint.get("step", 0))
    resumed_best_eval_loss = float(checkpoint.get("best_eval_loss", float("inf")))
    resumed_best_eval_epoch = int(checkpoint.get("best_eval_epoch", -1))
    start_epoch = checkpoint_epoch + 1
    start_step = checkpoint_step
    print(
        f"Resumed from checkpoint: {ckpt_path} at epoch {checkpoint_epoch}, step {checkpoint_step}. "
        f"Continuing from epoch {start_epoch}."
    )
    return start_epoch, start_step, resumed_best_eval_loss, resumed_best_eval_epoch


def _resolve_wandb_resume_id(run_dir: str) -> Optional[str]:
    """Resolve previous wandb run id from checkpoint or latest-run symlink."""
    ckpt_path = _latest_checkpoint_path(run_dir)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        run_id = checkpoint.get("wandb_run_id", None)
        if run_id is not None:
            run_id_str = str(run_id).strip()
            if len(run_id_str) > 0:
                return run_id_str

    latest_run = os.path.join(run_dir, "wandb", "latest-run")
    if os.path.exists(latest_run):
        run_dir_name = os.path.basename(os.path.realpath(latest_run.rstrip(os.sep)))
        if run_dir_name.startswith("run-") and "-" in run_dir_name:
            return run_dir_name.rsplit("-", 1)[-1]

    return None


def _record_wandb_run_metadata(run_dir: str) -> None:
    """Persist wandb run metadata in run_dir for easy run mapping."""
    if wandb.run is None:
        return

    run_id = str(wandb.run.id)
    run_name = str(wandb.run.name) if wandb.run.name is not None else "(unnamed)"
    run_url = str(wandb.run.url) if wandb.run.url is not None else ""
    run_local_dir = str(wandb.run.dir) if wandb.run.dir is not None else ""

    metadata_path = os.path.join(run_dir, "wandb_run_info.txt")
    with open(metadata_path, "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"run_url: {run_url}\n")
        f.write(f"wandb_local_dir: {run_local_dir}\n")

    print(f"wandb run name: {run_name}")
    print(f"wandb run id: {run_id}")
    if run_url:
        print(f"wandb run url: {run_url}")
    print(f"wandb run metadata file: {metadata_path}")

def _is_thompson_complete(run_dir: str, round_idx: int) -> bool:
    thompson_samples_path = os.path.join(run_dir, "thompson_samples.npz")
    return os.path.exists(thompson_samples_path)

def _is_training_complete(run_dir: str, num_epochs: int) -> bool:
    ckpt_path = _latest_checkpoint_path(run_dir)
    if not os.path.exists(ckpt_path):
        return False
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch >= int(num_epochs) - 1:
        return True

    # Infer early-stop completion from best_eval_epoch and configured patience.
    best_eval_epoch = int(checkpoint.get("best_eval_epoch", -1))
    if best_eval_epoch < 0:
        return False

    hydra_cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(hydra_cfg_path):
        return False

    try:
        run_cfg_obj = OmegaConf.load(hydra_cfg_path)
        run_cfg = OmegaConf.to_container(run_cfg_obj, resolve=True)
        if not isinstance(run_cfg, dict):
            return False
        patience = int(run_cfg.get("early_stopping_patience", 0))
    except Exception:
        return False

    return patience > 0 and (epoch - best_eval_epoch) >= patience


### EIG for x_0 = argmax_x f_theta(x)
class EIG(nn.Module):
    def __init__(self, 
                 variational_dist: ExperimentCategorical,
                 experiment: Experiment,
    ):
        """
        Initialize the EIG module.
        - fitness_function: the fitness function f_theta(x)
        - variational_dist: the variational distribution r_phi(y*|x*,{y,q}_t)
        - experiment: the experiment
        """
        super().__init__()
        self.variational_dist = variational_dist
        self.experiment = experiment

    def forward(self, 
                x_0: torch.Tensor,
                y_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a lower bound on EIG given the inputs:
         - x_0: [B_theta, sequence_length]
         - y_e: [B_theta, num_experiments]
        """

        log_probs = self.variational_dist(x_0=x_0, y_e=y_e) # [B_theta]

        return log_probs # Return the log probability


def _amp_device_type(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"

def make_sigma_schedule(
    sigma: float,
    sigma_zero_warmup: int,
    sigma_warmup: int,
    device: str,
):
    """Precompute the constant endpoints once; each call afterward does only
    device-side tensor arithmetic (or returns a cached tensor), never a fresh
    host->device transfer."""
    eps_t = torch.tensor(1e-8, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(float(sigma), dtype=torch.float32, device=device)
    sigma_warmup_steps = max(int(sigma_warmup), 1)
    sigma_warmup_end = int(sigma_zero_warmup) + int(sigma_warmup)

    def sigma_at_step(step: int) -> torch.Tensor:
        if step < sigma_zero_warmup:
            return eps_t
        if step >= sigma_warmup_end:
            return sigma_t
        # sigma_t * python_float + eps_t: scalar multiply/add on an existing
        # device tensor, no new host->device copy.
        warmup_frac = (step - sigma_zero_warmup) / sigma_warmup_steps
        return sigma_t * warmup_frac + eps_t

    return sigma_at_step


class EIGObjective(nn.Module):
    def __init__(
        self,
        eig_model: EIG,
        fitness_function: FitnessFunction,
        deterministic: bool,
    ):
        super().__init__()
        self.eig_model = eig_model
        self.fitness_function = fitness_function
        self.deterministic = deterministic

    def sample_y_e(
        self,
        log_w: torch.Tensor,
        log_v: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return self.fitness_function.rsample_y(
            log_tau=self.eig_model.experiment.log_probs_syntheses,
            log_pi=self.eig_model.experiment.log_weights_experiments,
            log_w=log_w,
            log_v=log_v,
            sigma=sigma,
            deterministic=self.deterministic,
        )

    def forward(
        self,
        log_w: torch.Tensor,
        log_v: torch.Tensor,
        x_0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        y_e = self.sample_y_e(log_w=log_w, log_v=log_v, sigma=sigma)
        log_probs = self.eig_model(x_0=x_0, y_e=y_e)
        
        return -log_probs.mean(), y_e # negate for gradient descent

# def construct_prior(
        
# )
#     '''
#     Construct the prior distribution p(theta), and draw samples from it.
#     '''

### Main training loop
def train(
    # data parameters
    run_seed: int,
    true_theta_seed: int,
    round_idx: int,
    batch_size: int,
    theta_path: Optional[str],
    sequence_length: int,
    vocab_size: int,
    vocabulary: Sequence[str],
    # function parameters
    num_samples_train: int,
    num_samples_prior: int,
    num_components: int,
    weights_dist: Union[float, dist_np.Distribution],
    oracle_type: str, # plm | bump
    true_weights_dist: Optional[Union[float, dist_np.Distribution]],
    true_num_components: Optional[int],
    overlap_reg_strength: float,
    # experiment parameters
    num_experiments: int,
    num_syntheses: int,
    initialize_syntheses_per_experiment: bool,
    freeze_experiment_weights: bool,
    tau_warmup_steps: int,
    # model parameters
    model_type: str,
    model_hidden_dim: int,            
    num_heads: int,
    num_layers: int,
    mlp_ratio: int,
    dropout: float,
    use_rope: bool,
    use_pos_embed: bool,
    # training parameters
    num_epochs: int,
    sigma: float,
    sigma_zero_warmup: int,
    sigma_warmup: int,
    device: str,
    optimizer_name: str,
    momentum: float,
    lr: float,
    synthesis_lr_scale: float,
    lr_warmup_steps: int,
    weight_decay: float,
    synthesis_weight_decay: float,
    dtype: str,
    clip_grad_value: Optional[float],
    # logging and evaluation parameters
    num_samples_eval: int,
    batch_size_eval: int,
    diagnostic_every: int,
    log_every: int,
    eval_every: int,
    top_k_every: int,
    umap_every: int,
    visualize_every: int,
    num_visualized_syntheses: int,
    k_for_top_k: int,
    beam_width_for_top_k: int,
    beam_chunk_size: int,
    use_umap_eval: bool,
    recompute_umap: bool,
    umap_together: bool,
    umap_num_q_samples_per_experiment: int,
    umap_num_x_0_samples: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_simulated_true: bool,
    num_samples_simulation: int,
    num_warmup_mcmc: int,
    num_chains_mcmc: int,
    mcmc_chain_method: str,
    enable_torch_profiler: bool,
    profiler_wait_steps: int,
    profiler_warmup_steps: int,
    profiler_active_steps: int,
    profiler_repeat: int,
    profiler_record_shapes: bool,
    profiler_profile_memory: bool,
    profiler_with_stack: bool,
    use_wandb: bool,
    run_dir: str,
    project_path: str,
    true_theta_index: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
):
    # Shape
    B = batch_size
    N = num_components
    E = num_experiments
    S = num_syntheses
    L = sequence_length
    K = vocab_size

    # Initialize the variational synthesis model
    experiment = Experiment(
        vocab_size=K,
        sequence_length=L,
        num_experiments=E,
        num_syntheses=S,
        tau_warmup_steps=tau_warmup_steps,
        initialize_syntheses_per_experiment=initialize_syntheses_per_experiment,
        freeze_experiment_weights=freeze_experiment_weights,
    ).to(device)

    # Initialize the variational distribution (placeholder dimensions)
    if model_type == "categorical":
        variational_dist = ExperimentCategorical(
            vocab_size=K,
            sequence_length=L,
            num_experiments=E,
        ).to(device)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Use one of: categorical")

    # randomness
    torch.manual_seed(run_seed)
    key = jax.random.PRNGKey(run_seed)

    if num_chains_mcmc < 1:
        raise ValueError(f"num_chains_mcmc must be >= 1, got {num_chains_mcmc}")
    if mcmc_chain_method not in ("auto", "parallel", "vectorized", "sequential"):
        raise ValueError(
            f"Invalid mcmc_chain_method: {mcmc_chain_method}. "
            "Use one of: auto, parallel, vectorized, sequential"
        )
    resolved_chain_method = "vectorized" if mcmc_chain_method == "auto" else mcmc_chain_method

    # GLM
    glm = NormalGLM()
    assert glm.monotone_increasing_link, "Currently only monotone increasing link functions are supported for the true theta construction logic, as we need it to find argmax_x"

    # load p(theta) or p(theta | data)
    # for round 1 and round > 1 respectively
    using_prior_samples = theta_path is None
    if using_prior_samples:
        key, predictive_key = jax.random.split(key)
        print("No theta path provided. Generating new prior samples...", flush=True)
        theta_path = os.path.join(run_dir, "prior_samples.npz")

        if os.path.exists(theta_path):
            print(f"Theta samples file {theta_path} already exists. Loading existing samples...", flush=True)
            theta_samples = np.load(theta_path)
        else:
            num_theta_samples = num_samples_prior + num_samples_eval
            start_time = time.time()
            print(f"Sampling {num_theta_samples} prior samples (this may take a while)...", flush=True)
            theta_samples = predictive(
                key=predictive_key,
                N=N,
                L=L,
                K=K,
                gamma=weights_dist,
                inverse_link_function=glm.inverse_link_function,
                sigma=sigma,
                overlap_reg_strength=overlap_reg_strength,
                path=theta_path,
                num_samples=num_theta_samples
            )
            elapsed = time.time() - start_time
            print(f"Saved theta samples to {theta_path} (took {elapsed:.1f}s)", flush=True)
    theta_samples = np.load(theta_path)

    if using_prior_samples:
        num_prior_function_plots = min(5, int(theta_samples["log_w"].shape[0]))
        if num_prior_function_plots > 0:
            rng = np.random.default_rng(int(run_seed))
            selected_prior_indices = rng.choice(
                int(theta_samples["log_w"].shape[0]),
                size=num_prior_function_plots,
                replace=False,
            )
            for vis_rank, sample_idx in enumerate(selected_prior_indices):
                sample_idx_int = int(sample_idx)
                log_w_prior = torch.as_tensor(
                    theta_samples["log_w"][sample_idx_int],
                    dtype=torch.float32,
                    device=device,
                )
                log_v_prior = torch.as_tensor(
                    theta_samples["log_v"][sample_idx_int],
                    dtype=torch.float32,
                    device=device,
                )
                _ = visualize_function(
                    step=0,
                    vocabulary=vocabulary,
                    log_w=log_w_prior,
                    log_v=log_v_prior,
                    path=run_dir,
                    simulated_true_name=f"prior_function_{vis_rank:02d}_idx_{sample_idx_int:06d}",
                )
            print(
                "Saved "
                f"{num_prior_function_plots} random prior function visualizations: "
                f"{selected_prior_indices.tolist()}"
            )
    
    # Load theta samples fully onto `device` and batch them
    train_dl, val_dl = load_theta_gpu_splits(
        path=theta_path,
        test_size=num_samples_eval,
        device=device,
    )
    print(f"Loaded theta samples from {theta_path}.")
    print(f"Created train and validation theta splits with {len(train_dl)} and {len(val_dl)} samples respectively, resident on {device}.")

    # Initialize the fitness function, f_theta
    fitness_function = FitnessFunction(
        num_components=N,
        sequence_length=L,
        vocab_size=K,
        glm=NormalGLM(),
    ).to(device)
    summary_statistics_path = os.path.join(run_dir, "prior_function_summary.png")
    fitness_function.generate_summary_statistics(theta_samples=theta_samples, path=summary_statistics_path)

    # Also, make a visualization of the graphical model
    graphical_model_path = summary_statistics_path.replace("summary", "graphical_model")
    if not graphical_model_path.endswith(".png"):
        graphical_model_path = f"{graphical_model_path}.png"
    visualize_graphical_model(
        path=graphical_model_path,
        N=N,
        L=L,
        K=K,
        gamma=weights_dist,
        inverse_link_function=glm.inverse_link_function,
        sigma=sigma,
    )

    # Load true function
    if oracle_type == 'bump':
        assert true_num_components is not None, "true_num_components must be specified when using bump function type"
        assert true_weights_dist is not None, "true_weights_dist must be specified when using bump function type"
        fixed_log_w_true, fixed_log_v_true, fixed_mode_true, true_theta_file, val_dl = _load_or_create_true_theta(
            project_path=project_path,
            val_dl=val_dl,
            theta_index=true_theta_index,
            device=device,
            true_weights_dist=true_weights_dist,
            true_num_components=true_num_components,
            seed=true_theta_seed,
        )
        print(f"Using campaign true theta from: {true_theta_file}")
    elif oracle_type == 'plm':
        assert true_weights_dist is None, "true_weights_dist must be None when using PLM function type"
        assert true_num_components is None, "true_num_components must be None when using PLM function type"
        print('Using PLM-based true theta with no fixed components (oracle)')

    # Initialize the EIG model
    eig_model = EIG(
        variational_dist=variational_dist,
        experiment=experiment,
    ).to(device)

    train_objective = EIGObjective(
        eig_model=eig_model,
        fitness_function=fitness_function,
        deterministic=False,
    ).to(device)
    eval_objective = EIGObjective(
        eig_model=eig_model,
        fitness_function=fitness_function,
        deterministic=True,
    ).to(device)

    synthesis_params = []
    other_params = []
    for name, param in eig_model.named_parameters():
        if name.startswith("experiment."):
            synthesis_params.append(param)
        else:
            other_params.append(param)

    # Possibility for smaller differential learning rates
    param_groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": synthesis_params, "lr": lr * synthesis_lr_scale, "weight_decay": synthesis_weight_decay},
    ]

    # Optimizer
    # print trainable parameters
    for name, param in eig_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    # print how many parameters are in synthesis_params vs other_params
    num_synthesis_params = sum(p.numel() for p in synthesis_params)
    num_other_params = sum(p.numel() for p in other_params)
    print(f"Number of synthesis parameters: {num_synthesis_params}")
    print(f"Number of other parameters (variational distribution): {num_other_params}")

    # Optimizer
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(param_groups, fused=True)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(param_groups, fused=True)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(param_groups, momentum=momentum)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(param_groups, momentum=momentum)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    
    # lr scheduler
    if lr_warmup_steps > 0:
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=lr_warmup_steps)
    else:
        lr_scheduler = None

    # sigma scheduler
    sigma_at_step = make_sigma_schedule(
        sigma=sigma,
        sigma_zero_warmup=sigma_zero_warmup,
        sigma_warmup=sigma_warmup,
        device=device,
    )

    if dtype == 'bf16':
        amp_dtype = torch.bfloat16
    elif dtype == 'fp16':
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use one of: bf16, fp16")

    # gradient scaling for mixed precision training
    scaler = GradScaler(
        enabled = amp_dtype == torch.float16
    )

    start_epoch, start_step, resumed_best_eval_loss, resumed_best_eval_epoch = _maybe_resume_from_checkpoint(
        run_dir=run_dir,
        eig_model=eig_model,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    if start_epoch >= num_epochs:
        print(
            f"Checkpoint already at epoch {start_epoch - 1}, which is >= requested num_epochs={num_epochs}. "
            "Nothing to do."
        )
        return

    best_eval_loss = float(resumed_best_eval_loss)
    best_eval_epoch = int(resumed_best_eval_epoch)
    amp_device_type = _amp_device_type(device)
    train_objective_compiled = torch.compile(train_objective)
    eval_objective_compiled = torch.compile(eval_objective)
    diagnostic_every_n = max(diagnostic_every, 1)
    log_every_n = max(log_every, 1)
    eval_every_n = max(eval_every, 1)
    top_k_every_n = max(top_k_every, 1)
    umap_every_n = max(umap_every, 1)
    visualize_every_n = max(visualize_every, 1)

    profiler = None
    if enable_torch_profiler:
        profiler_wait_steps = max(int(profiler_wait_steps), 0)
        profiler_warmup_steps = max(int(profiler_warmup_steps), 0)
        profiler_active_steps = max(int(profiler_active_steps), 1)
        profiler_repeat = max(int(profiler_repeat), 1)
        profiler_dir = os.path.join(run_dir, "torch_profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if device.startswith("cuda") and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=profiler_wait_steps,
                warmup=profiler_warmup_steps,
                active=profiler_active_steps,
                repeat=profiler_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=profiler_record_shapes,
            profile_memory=profiler_profile_memory,
            with_stack=profiler_with_stack,
        )
        profiler.start()
        print(f"PyTorch profiler enabled. Traces will be written to: {profiler_dir}")

    # Training loop (epoch-based)
    start_time = time.time()
    step = start_step
    for epoch in range(start_epoch, num_epochs):
        eig_model.train()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_loss_count = 0

        for log_w, log_v, mode in train_dl.epoch_batches(batch_size=B):
            step += 1

            diagnostics_due = use_wandb and (step % diagnostic_every_n == 0)

            if experiment.tau_is_frozen and step >= tau_warmup_steps:
                experiment._parameters_syntheses_unscaled.requires_grad_(True)
                experiment.tau_is_frozen = False
                print(f"Step {step}: Unfroze syntheses model parameters.")

            # zerograd
            optimizer.zero_grad(set_to_none=True)

            current_sigma = sigma_at_step(step=step)

            # x_0 = argmax_x f_theta(x)
            x_0 = fitness_function.argmax(log_w=log_w, log_v=log_v, candidates=mode)

            with autocast(device_type=amp_device_type, dtype=amp_dtype):
                loss, y_e = train_objective_compiled(
                    log_w=log_w,
                    log_v=log_v,
                    x_0=x_0,
                    sigma=current_sigma,
                )

            # Backpropagation
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            
            # Apply gradient clipping to prevent exploding gradients
            if clip_grad_value is not None:
                # Use norm-based clipping
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(eig_model.parameters(), max_norm=clip_grad_value)
            elif diagnostics_due:
                # compute gradients for diagnostics
                grads = [p.grad for p in eig_model.parameters() if p.grad is not None]
                grad_norm_tensor = torch.nn.utils.get_total_norm(grads)
            else:
                grad_norm_tensor = None

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if profiler is not None:
                profiler.step()

            # Extract diagnostics
            if diagnostics_due:
                current_lr = optimizer.param_groups[0]['lr']

                # Compute gradient norm
                assert grad_norm_tensor is not None, "Gradient norm tensor is None. Ensure that gradients are computed before diagnostics."
                total_grad_norm = grad_norm_tensor.item()

                # Compute statistics
                # pi_logits = eig_model.experiment.log_weights_experiments.detach()  # [num_experiments]
                pi_weights = eig_model.experiment.weights_experiments.detach()  # [num_experiments]
                # tau_logits = eig_model.experiment.log_probs_syntheses.detach()  # [num_syntheses]
                tau_probs = eig_model.experiment.probs_syntheses.detach()  # [num_syntheses]
                diagnostics_dict = {
                    "loss": loss.item(),
                    # Y statistics
                    "y_e_mean": y_e.mean().item(),
                    "y_e_min": y_e.min().item(),
                    "y_e_max": y_e.max().item(),
                    # Pi probabilities statistics
                    "pi_weights_min": pi_weights.min().item(),
                    "pi_weights_max": pi_weights.max().item(),
                    # Tau probabilities statistics
                    "tau_probs_min": tau_probs.min().item(),
                    "tau_probs_max": tau_probs.max().item(),
                    # Gradient norm
                    "grad_norm": total_grad_norm,
                    # GLM sigma (with warmup)
                    "glm_sigma": current_sigma.item(),
                    # Learning rate
                    "lr": current_lr,
                }
                # Log to wandb
                wandb.log(diagnostics_dict, step=step)

            epoch_loss_sum += loss.detach()
            epoch_loss_count += 1

        # Epoch-based logging
        if epoch % log_every_n == 0:
            mean_epoch_loss = (epoch_loss_sum / max(epoch_loss_count, 1)).item()
            print(f"Epoch {epoch}: mean_train_loss={mean_epoch_loss:.4f}, last_step={step}")

        should_eval = (epoch % eval_every_n == 0) or (epoch == num_epochs - 1)
        if not should_eval:
            continue

        # Evaluate loss every epoch for early stopping; run expensive top-k only on its own cadence.
        print(f"Evaluating at epoch {epoch} (step {step})...")
        eig_model.eval()
        evaluation_tau = experiment.probs_syntheses
        evaluation_pi = experiment.weights_experiments

        do_top_k_eval = (epoch % top_k_every_n == 0) or (epoch == num_epochs - 1)

        eval_log_probs = torch.zeros((), device=device)
        eval_num_examples = 0
        y_actual_samples_eval = []
        y_max_samples_eval = []

        # Iterate over the full held-out split each evaluation pass.
        eval_sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for log_w_eval, log_v_eval, mode_eval in val_dl.eval_batches(batch_size=batch_size_eval):
                argmax_x_eval = fitness_function.argmax(log_w=log_w_eval, log_v=log_v_eval, candidates=mode_eval)
                with autocast(device_type=amp_device_type, dtype=amp_dtype):
                    eval_log_prob_mean, y_e_eval = eval_objective_compiled(
                        log_w=log_w_eval,
                        log_v=log_v_eval,
                        x_0=argmax_x_eval,
                        sigma=eval_sigma,
                    )

                batch_size_eval_actual = int(log_w_eval.shape[0])
                eval_log_probs += eval_log_prob_mean * batch_size_eval_actual
                eval_num_examples += batch_size_eval_actual

                if do_top_k_eval:

                    samples = variational_dist.get_topk(
                        y_e=y_e_eval,
                        k=k_for_top_k,
                        beam_width=beam_width_for_top_k,
                        chunk_size=beam_chunk_size,
                    )

                    # Prepare for top-k fitness histogram
                    y_actual_samples_eval.append(fitness_function.f(log_w=log_w_eval, log_v=log_v_eval, x=samples))  # [B_theta, k]
                    y_max_samples_eval.append(fitness_function.f(log_w=log_w_eval, log_v=log_v_eval, x=argmax_x_eval.unsqueeze(1)))  # [B_theta, 1]

        loss_eval = (eval_log_probs / max(eval_num_examples, 1)).item()  # Average log probability over observed test samples

        # Keep all post-eval actions grouped at epoch scope.
        if should_eval:

            if use_wandb:
                wandb.log({"loss_eval": loss_eval}, step=step)

            improved = float(loss_eval) < (best_eval_loss - float(early_stopping_min_delta))
            if improved:
                best_eval_loss = float(loss_eval)
                best_eval_epoch = epoch
                # Save best.pt
                checkpoint_path = _save_checkpoint(
                    run_dir=run_dir,
                    epoch=epoch,
                    step=step,
                    eig_model=eig_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    best_eval_loss=best_eval_loss,
                    best_eval_epoch=best_eval_epoch,
                    is_best=True,
                )
                print(f"Saved checkpoint (new best): {checkpoint_path}")
            elif early_stopping_patience > 0 and best_eval_epoch >= 0 and (epoch - best_eval_epoch) >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}: loss_eval did not improve for {early_stopping_patience} epochs. "
                    f"Best loss_eval={best_eval_loss:.6f} at epoch {best_eval_epoch}."
                )
                checkpoint_path = _save_checkpoint(
                    run_dir=run_dir,
                    epoch=epoch,
                    step=step,
                    eig_model=eig_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    best_eval_loss=best_eval_loss,
                    best_eval_epoch=best_eval_epoch,
                )
                print(f"Saved checkpoint: {checkpoint_path}")
                if profiler is not None:
                    profiler.stop()
                return

            if do_top_k_eval:
                # aggregate the samples
                y_max = torch.cat(y_max_samples_eval, dim=0)  # [num_samples_eval, 1]
                y_actual = torch.cat(y_actual_samples_eval, dim=0)  # [num_samples_eval, k]
                y_fraction = y_actual / (y_max + 1e-12)  # [num_samples_eval, k] (normalize to [0, 1])

                top_k_histogram_path = top_k_fitness(
                    step=step,
                    top_k_fitness_values=y_fraction,
                    output_dir=run_dir,
                ) # [B_theta]

                # get values for the top-k fitness metrics
                max_top_k_fitness_values = torch.max(y_fraction, dim=-1).values  # [B_theta]
                top_k_mean_over_theta = max_top_k_fitness_values.mean().item()
                top_k_max_over_theta = max_top_k_fitness_values.max().item()
                print(f"Top-k fitness: mean_over_theta={top_k_mean_over_theta:.4f}, max_over_theta={top_k_max_over_theta:.4f}")
                if use_wandb:
                    wandb.log({
                        "top_k_mean_over_theta": top_k_mean_over_theta,
                        "top_k_max_over_theta": top_k_max_over_theta,
                        "top_k_fitness_histogram": wandb.Image(top_k_histogram_path),
                    }, step=step)

            do_visualize = (epoch % visualize_every_n == 0) or (epoch == num_epochs - 1)
            if do_visualize:
                # Save logoplots periodically
                logopaths = visualize_tau_and_pi(
                    tau=evaluation_tau,
                    pi=evaluation_pi,
                    vocabulary=vocabulary,
                    step=step,
                    num_visualized_syntheses=num_visualized_syntheses,
                    output_dir=run_dir
                )
                # Log logoplots to wandb
                logoplot_images = {}
                for i, logopath in enumerate(logopaths):
                    print(f"  → Saved logoplot: {logopath}")
                    if use_wandb:
                        logoplot_images[f"logoplot_exp_{i}"] = wandb.Image(logopath)
                if use_wandb:
                    wandb.log(logoplot_images, step=step)
            # if not improved, save a checkpoint for potential early stopping
            if not improved:
                checkpoint_path = _save_checkpoint(
                    run_dir=run_dir,
                    epoch=epoch,
                    step=step,
                    eig_model=eig_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    best_eval_loss=best_eval_loss,
                    best_eval_epoch=best_eval_epoch,
                )
                print(f"Saved checkpoint: {checkpoint_path}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    print(f"Training of experiment completed in {hours}h {minutes}m")

    if profiler is not None:
        profiler.stop()
        print("PyTorch profiler session completed.")


def thompson_sampling(
    run_seed: int,
    true_theta_seed: int,
    theta_path: Optional[str],
    sequence_length: int,
    vocab_size: int,
    num_samples_train: int,
    num_samples_prior: int,
    num_samples_eval: int,
    num_components: int,
    weights_dist: Union[float, dist_np.Distribution],
    oracle_type: str,
    true_weights_dist: Optional[Union[float, dist_np.Distribution]],
    true_num_components: Optional[int],
    overlap_reg_strength: float,
    num_experiments: int,
    num_syntheses:int,
    sigma: float,
    device: str,
    run_dir: str,
    project_path: str,
    true_theta_index: int,
):
    """
    Perform thompson sampling and save the result as an Experiment object in the checkpoint directory.
    """
    # Shape
    N = num_components
    E = num_experiments
    S = num_syntheses
    L = sequence_length
    K = vocab_size

    if S != E:
        raise NotImplementedError(f"For Thompson sampling, S different from E does not make sense.")

    # randomness
    torch.manual_seed(run_seed)
    key = jax.random.PRNGKey(run_seed)

    # GLM
    glm = NormalGLM()

    # sample theta from the prior/posterior from the previous round
    if theta_path is None:
        key, predictive_key = jax.random.split(key)
        print("No theta path provided. Generating new prior samples...", flush=True)
        theta_path = os.path.join(run_dir, "prior_samples.npz")

        if os.path.exists(theta_path):
            print(f"Theta samples file {theta_path} already exists. Loading existing samples...", flush=True)
            theta_samples = np.load(theta_path)
        else:
            num_theta_samples = num_samples_prior + num_samples_eval
            start_time = time.time()
            print(f"Sampling {num_theta_samples} prior samples...", flush=True)
            theta_samples = predictive(
                key=predictive_key,
                N=N,
                L=L,
                K=K,
                gamma=weights_dist,
                inverse_link_function=glm.inverse_link_function,
                sigma=sigma,
                overlap_reg_strength=overlap_reg_strength,
                path=theta_path,
                num_samples=num_theta_samples
            )
            elapsed = time.time() - start_time
            print(f"Saved theta samples to {theta_path} (took {elapsed:.1f}s)", flush=True)
    theta_samples = np.load(theta_path)

    # Load theta samples onto the GPU
    train_dl, val_dl = load_theta_gpu_splits(
        path=theta_path,
        test_size=num_samples_eval,
        device=device,
    )
    print(f"Loaded theta samples from {theta_path}.")
    print(f"Created train and validation theta splits with {len(train_dl)} and {len(val_dl)} samples respectively, resident on {device}.")

    # Initialize the fitness function
    fitness_function = FitnessFunction(
        num_components=N,
        sequence_length=L,
        vocab_size=K,
        glm=NormalGLM(),
    ).to(device)
    summary_statistics_path = os.path.join(run_dir, "prior_function_summary.png")
    fitness_function.generate_summary_statistics(theta_samples=theta_samples, path=summary_statistics_path)
    # Also, make a nice visualization of the graphical model
    graphical_model_path = summary_statistics_path.replace("summary", "graphical_model")
    if not graphical_model_path.endswith(".png"):
        graphical_model_path = f"{graphical_model_path}.png"
    visualize_graphical_model(
        path=graphical_model_path,
        N=N,
        L=L,
        K=K,
        gamma=weights_dist,
        inverse_link_function=glm.inverse_link_function,
        sigma=sigma,
    )
    # Load true function
    if oracle_type == 'bump':
        assert true_num_components is not None, "true_num_components must be specified when using bump function type"
        assert true_weights_dist is not None, "true_weights_dist must be specified when using bump function type"
        fixed_log_w_true, fixed_log_v_true, fixed_mode_true, true_theta_file, val_dl = _load_or_create_true_theta(
            project_path=project_path,
            val_dl=val_dl,
            theta_index=true_theta_index,
            device=device,
            true_weights_dist=true_weights_dist,
            true_num_components=true_num_components,
            seed=true_theta_seed,
        )
        print(f"Using campaign true theta from: {true_theta_file}")
    elif oracle_type == 'plm':
        assert true_weights_dist is None, "true_weights_dist must be None when using PLM function type"
        assert true_num_components is None, "true_num_components must be None when using PLM function type"
        print('Using PLM-based true theta with no fixed components (oracle)')

    # get the first (shuffled) batch of theta samples for Thompson sampling.
    log_w_thompson, log_v_thompson, mode_thompson = next(train_dl.epoch_batches(batch_size=S))

    # define the experiment as delta masses on the argmax_x of these samples
    thompson_samples = fitness_function.argmax(log_w=log_w_thompson, log_v=log_v_thompson, candidates=mode_thompson)  # [num_experiments, sequence_length]
    # save to file as a npz file
    thompson_samples_path = os.path.join(run_dir, "thompson_samples.npz")
    experiment = thompson_samples_to_experiment(
        thompson_samples=thompson_samples,
        sequence_length=L,
        vocab_size=K,
        num_experiments=E,
    )
    torch.save(experiment.state_dict(), os.path.join(run_dir, "thompson_experiment.pt"))
    np.savez(thompson_samples_path, thompson_samples=thompson_samples.cpu().numpy())
    print(f"Saved Thompson samples to {thompson_samples_path} and Thompson experiment state dict to {os.path.join(run_dir, 'thompson_experiment.pt')}")


def _main(cfg: DictConfig) -> None:

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config_dict, dict) 
    try:
        run_dir = str(Path(HydraConfig.get().runtime.output_dir).resolve())
    except Exception:
        # Fallback if Hydra is not initialized (e.g., when called from threads)
        run_dir = os.path.join(str(cfg.project_path), f"round_{cfg.round}_train")
        run_dir = str(Path(run_dir).resolve())

    os.makedirs(run_dir, exist_ok=True)
    print(f"Hydra run directory: {run_dir}")

    round_idx = int(cfg.round)
    if round_idx < 1:
        raise ValueError(f"round must be >= 1, got {round_idx}")

    base_seed = int(cfg.seed)
    duplicate_idx = _resolve_duplicate_index(cfg)
    run_seed = _compose_round_seed(base_seed, duplicate_idx, round_idx)
    true_theta_seed = _compose_true_theta_seed(base_seed, duplicate_idx)

    acquisition_function = str(cfg.acquisition_function)
    assert acquisition_function in ("lifted", "thompson", "argmax_lifted", "gp_acqf"), (
        f"Unsupported acquisition_function: {acquisition_function!r}. "
        "Expected one of: lifted, thompson, argmax_lifted, gp_acqf."
    )

    if acquisition_function == "thompson":
        cfg.use_wandb = False  # Disable wandb for Thompson sampling
        if _is_thompson_complete(run_dir=run_dir, round_idx=round_idx):
            print(
                "Thompson sampling already complete for this run directory; "
                "skipping main.py without starting wandb."
            )
            return
    elif acquisition_function == "gp_acqf":
        cfg.use_wandb = False  # No training step to log for this acquisition function
    else:
        if "num_epochs" not in cfg:
            raise ValueError("Missing required config key: num_epochs")
        num_epochs = int(cfg.num_epochs)
        if _is_training_complete(run_dir=run_dir, num_epochs=num_epochs):
            print(
                "Training already complete for this run directory; "
                "skipping main.py without starting wandb."
            )
            return

    # Build weights distribution for the fitness function
    weight_dist = _build_weights_dist(
        mode=str(cfg.weights.mode),
        value=float(cfg.weights.value),
        gamma_concentration=float(cfg.weights.gamma_concentration),
        gamma_rate=float(cfg.weights.gamma_rate),
    )

    # build the 'true' fitness function
    if cfg.oracle.type == "bump":
        true_weights_cfg = OmegaConf.select(cfg, "oracle.weights")
        if true_weights_cfg is None:
            print("No separate oracle.weights config found; using the same weights distribution for the true function as for the fitness function.")
            true_weights_cfg = cfg.weights
        true_weight_dist = _build_weights_dist(
            mode=str(true_weights_cfg.mode),
            value=float(true_weights_cfg.value),
            gamma_concentration=float(true_weights_cfg.gamma_concentration),
            gamma_rate=float(true_weights_cfg.gamma_rate),
        )
        true_num_components = OmegaConf.select(cfg, "oracle.num_components")
        if true_num_components is None:
            print("No separate oracle.num_components config found; using the same num_components for the true function as for the fitness function.")
            true_num_components = int(cfg.num_components)
        else:
            true_num_components = int(true_num_components)
        if true_num_components < 1:
            raise ValueError(f"oracle.num_components must be >= 1, got {true_num_components}")
    elif cfg.oracle.type == "plm":
        true_weight_dist = None
        true_num_components = None
    else:
        raise ValueError(f"Unsupported oracle.type: {cfg.oracle.type}")

    if acquisition_function == "gp_acqf":
        # No training step at all
        if cfg.oracle.type == "bump":
            assert true_weight_dist is not None and true_num_components is not None
            _ensure_true_theta_exists(
                project_path=str(cfg.project_path),
                run_dir=run_dir,
                run_seed=run_seed,
                true_theta_seed=true_theta_seed,
                sequence_length=int(cfg.sequence_length),
                vocab_size=len(tuple(cfg.vocabulary)),
                num_samples_prior=int(cfg.num_samples_prior),
                num_samples_eval=int(cfg.num_samples_eval),
                num_components=int(cfg.num_components),
                weights_dist=weight_dist,
                sigma=float(cfg.sigma),
                overlap_reg_strength=float(cfg.overlap_reg_strength),
                true_weights_dist=true_weight_dist,
                true_num_components=true_num_components,
                true_theta_index=int(cfg.simulation.theta_index),
                device=str(cfg.device),
            )
        print(
            "acquisition_function=gp_acqf: main.py has no training step for this acquisition "
            "function; candidate selection happens entirely in simulate_observation.py."
        )
        return

    if cfg.use_wandb:
        secret_path = hydra.utils.to_absolute_path(str(cfg.wandb_secret_file))
        with open(secret_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

        wandb_resume_id = _resolve_wandb_resume_id(run_dir)
        init_kwargs = {
            "entity": None if cfg.wandb_entity is None else str(cfg.wandb_entity),
            "project": str(cfg.wandb_project),
            "dir": run_dir,
            "config": config_dict,
        }
        if wandb_resume_id is not None:
            init_kwargs["id"] = wandb_resume_id
            init_kwargs["resume"] = "must"
            print(f"Resuming wandb run id: {wandb_resume_id}")

        wandb.init(**init_kwargs)  # type: ignore[arg-type]
        _record_wandb_run_metadata(run_dir=run_dir)

    theta_path = str(cfg.theta_path) if cfg.theta_path is not None else None
    if theta_path is None and round_idx > 1:
        theta_path = _prev_round_theta_path(str(cfg.project_path), round_idx)
        if not os.path.exists(theta_path):
            raise FileNotFoundError(
                f"Expected posterior theta samples for round {round_idx - 1} at {theta_path}, but it does not exist. "
                "Run simulate_observation.py for the previous round first."
            )

    if cfg.acquisition_function == "thompson":
        thompson_sampling(
            run_seed=run_seed,
            true_theta_seed=true_theta_seed,
            theta_path=theta_path,
            sequence_length=int(cfg.sequence_length),
            vocab_size=len(tuple(cfg.vocabulary)),
            num_samples_train=int(cfg.num_samples_train),
            num_samples_prior=int(cfg.num_samples_prior),
            num_samples_eval=int(cfg.num_samples_eval),
            num_components=int(cfg.num_components),
            weights_dist=weight_dist,
            oracle_type=str(cfg.oracle.type),
            true_weights_dist=true_weight_dist,
            true_num_components=true_num_components,
            overlap_reg_strength=float(cfg.overlap_reg_strength),
            num_experiments=int(cfg.num_experiments),
            num_syntheses=int(cfg.num_syntheses),
            sigma=float(cfg.sigma),
            device=str(cfg.device),
            run_dir=run_dir,
            project_path=str(cfg.project_path),
            true_theta_index=int(cfg.simulation.theta_index),
        )
    else:
        train(
            run_seed=run_seed,
            true_theta_seed=true_theta_seed,
            round_idx=round_idx,
            theta_path=theta_path,
            batch_size=int(cfg.batch_size),
            sequence_length=int(cfg.sequence_length),
            vocab_size=len(tuple(cfg.vocabulary)),
            vocabulary=tuple(cfg.vocabulary),
            num_samples_train=int(cfg.num_samples_train),
            num_samples_prior=int(cfg.num_samples_prior),
            num_components=int(cfg.num_components),
            weights_dist=weight_dist,
            oracle_type=str(cfg.oracle.type),
            true_weights_dist=true_weight_dist,
            true_num_components=true_num_components,
            overlap_reg_strength=float(cfg.overlap_reg_strength),
            num_experiments=int(cfg.num_experiments),
            num_syntheses=int(cfg.num_syntheses),
            initialize_syntheses_per_experiment=bool(cfg.initialize_syntheses_per_experiment),
            freeze_experiment_weights=bool(cfg.freeze_experiment_weights),
            tau_warmup_steps=int(cfg.tau_warmup_steps),
            model_type=str(cfg.model_type),
            model_hidden_dim=int(cfg.model_hidden_dim),
            num_heads=int(cfg.num_heads),
            num_layers=int(cfg.num_layers),
            mlp_ratio=int(cfg.mlp_ratio),
            dropout=float(cfg.dropout),
            use_rope=bool(cfg.use_rope),
            use_pos_embed=bool(cfg.use_pos_embed),
            num_epochs=num_epochs,
            sigma=float(cfg.sigma),
            sigma_zero_warmup=int(cfg.sigma_zero_warmup),
            sigma_warmup=int(cfg.sigma_warmup),
            diagnostic_every=int(cfg.diagnostic_every),
            log_every=int(cfg.log_every),
            eval_every=int(cfg.eval_every),
            top_k_every=int(cfg.get("top_k_every", cfg.eval_every)),
            umap_every=int(cfg.get("umap_every", cfg.eval_every)),
            visualize_every=int(cfg.get("visualize_every", cfg.eval_every)),
            num_visualized_syntheses=int(cfg.get("num_visualized_syntheses", 1)),
            k_for_top_k=int(cfg.k_for_top_k),
            beam_width_for_top_k=int(cfg.beam_width_for_top_k),
            beam_chunk_size=int(cfg.beam_chunk_size),
            num_samples_eval=int(cfg.num_samples_eval),
            batch_size_eval=int(cfg.batch_size_eval),
            use_umap_eval=bool(cfg.use_umap_eval),
            recompute_umap=bool(cfg.recompute_umap),
            umap_together=bool(cfg.umap_together),
            # umap_every=int(cfg.umap_every),
            umap_num_x_0_samples=int(cfg.umap_num_x_0_samples),
            umap_num_q_samples_per_experiment=int(cfg.umap_num_q_samples_per_experiment),
            umap_n_neighbors=int(cfg.umap_n_neighbors),
            umap_min_dist=float(cfg.umap_min_dist),
            umap_simulated_true=bool(cfg.umap_simulated_true),
            num_samples_simulation=int(cfg.num_samples_simulation),
            num_warmup_mcmc=int(cfg.num_warmup_mcmc),
            num_chains_mcmc=int(cfg.get("num_chains_mcmc", 1)),
            mcmc_chain_method=str(cfg.get("mcmc_chain_method", "auto")),
            enable_torch_profiler=bool(cfg.get("enable_torch_profiler", False)),
            profiler_wait_steps=int(cfg.get("profiler_wait_steps", 1)),
            profiler_warmup_steps=int(cfg.get("profiler_warmup_steps", 1)),
            profiler_active_steps=int(cfg.get("profiler_active_steps", 6)),
            profiler_repeat=int(cfg.get("profiler_repeat", 1)),
            profiler_record_shapes=bool(cfg.get("profiler_record_shapes", True)),
            profiler_profile_memory=bool(cfg.get("profiler_profile_memory", True)),
            profiler_with_stack=bool(cfg.get("profiler_with_stack", False)),
            device=str(cfg.device),
            optimizer_name=str(cfg.optimizer_name),
            momentum=float(cfg.momentum),
            lr=float(cfg.learning_rate),
            synthesis_lr_scale=float(cfg.synthesis_lr_scale),
            lr_warmup_steps=int(cfg.lr_warmup_steps),
            weight_decay=float(cfg.weight_decay),
            synthesis_weight_decay=float(cfg.synthesis_weight_decay),
            dtype=str(cfg.dtype),
            use_wandb=bool(cfg.use_wandb),
            clip_grad_value=None if cfg.clip_grad_value is None else float(cfg.clip_grad_value),
            run_dir=run_dir,
            project_path=str(cfg.project_path),
            true_theta_index=int(cfg.simulation.theta_index),
            early_stopping_patience=int(cfg.get("early_stopping_patience", 0)),
            early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", 0.0)),
        )

    if cfg.use_wandb:
        wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    return _main(cfg)

if __name__ == "__main__":
    main()  # type: ignore[call-arg]