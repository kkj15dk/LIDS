import copy
import os
from typing import Union, Tuple, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import numpy as np
import torch
import jax.numpy as jnp
from torch.amp.autocast_mode import autocast
from omegaconf import DictConfig

import logomaker
import matplotlib.pyplot as plt
import pandas as pd

from MCMC import (
    ObservationBlock,
    save_observation_block,
    load_observation_blocks,
    construct_posterior_samples,
    construct_posterior_samples_svi,
    _build_weights_dist,
    MCMCConvergenceError,
)
from eval import (
    compute_top_k_true_metrics,
    compute_map_metrics_from_posterior,
    visualize_tau_and_pi,
    visualize_function,
    visualize_umap_discrete,
    _experiment_from_observation_block,
)
from experiment import (
    Experiment,
    load_experiment_from_eig_checkpoint,
    load_experiment_from_thompson_samples,
    thompson_samples_to_experiment,
)
from function import NormalGLM
from oracle import Oracle, OraclePLM, OracleBump
from main import FitnessFunction
from model import ExperimentCategorical
from timing import print_time
from GP import select_candidates as gp_select_candidates, compute_map_metrics as gp_compute_map_metrics


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


def _read_map_ratio_from_metrics(out_dir: str) -> float | None:
    map_metrics_path = os.path.join(out_dir, "map_metrics.txt")
    if not os.path.exists(map_metrics_path):
        return None
    with open(map_metrics_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() == "MAP_ratio":
                try:
                    return float(value.strip())
                except ValueError:
                    return None
    return None


def _compute_importance_weight_ess(log_ws: list[torch.Tensor]) -> dict[str, np.ndarray]:
    """Effective sample size (ESS) of the importance weights w(x) = q(x)/p(x) used by OraclePLM's
    Monte Carlo estimate of <f, q>."""
    log_w_all = torch.cat(log_ws, dim=0)  # [num_samples, num_experiments]
    num_samples = log_w_all.shape[0]
    log_sum_w = torch.logsumexp(log_w_all, dim=0)  # [num_experiments]
    log_sum_w2 = torch.logsumexp(2.0 * log_w_all, dim=0)  # [num_experiments]
    ess = torch.exp(2.0 * log_sum_w - log_sum_w2)  # [num_experiments]
    return {
        "num_samples": np.asarray(num_samples),
        "ess": ess.numpy(),
        "ess_fraction": (ess / num_samples).numpy(),
    }


def _perturb_experiment_synthesis(experiment: Experiment, rho: float, seed: int) -> Experiment:
    """
    Deep copy of `experiment` with beta corrupted to:
    beta_true = (1 - rho) * beta + rho * omega, with omega[s, l, :] ~ Dirichlet(1_K).
    """
    pert = copy.deepcopy(experiment)
    with torch.no_grad():
        beta = pert.probs_syntheses.detach()  # [S, L, K], already softmaxed
        S, L, K = beta.shape
        g = torch.Generator(device=beta.device).manual_seed(int(seed))
        e = -torch.rand((S, L, K), generator=g, device=beta.device).clamp_min(1e-12).log()
        omega = e / e.sum(dim=-1, keepdim=True)  # Dirichlet(1_K) exactly
        beta_true = (1.0 - rho) * beta + rho * omega
        beta_true = beta_true / beta_true.sum(dim=-1, keepdim=True)
        pert._parameters_syntheses_unscaled.data.copy_(beta_true.clamp_min(1e-12).log())
    pert.eval()
    return pert


def simulate_observation(
    experiment: Experiment,
    oracle: Oracle,
    sigma: float,
    return_log_ws: bool = False,
    q_misspec_rho: float = 0.0,
    q_misspec_seed: Optional[int] = None,
) -> Tuple[ObservationBlock, Optional[dict[str, np.ndarray]]]:
    """Simulate one observation block from a trained experiment and a plm as an oracle."""
    log_tau = jnp.asarray(experiment.log_probs_syntheses.detach().cpu().numpy())
    log_pi = jnp.asarray(experiment.log_weights_experiments.detach().cpu().numpy())

    oracle_experiment = experiment
    if q_misspec_rho > 0.0:
        assert q_misspec_seed is not None, "q_misspec_seed must be provided when q_misspec_rho > 0.0"
        oracle_experiment = _perturb_experiment_synthesis(experiment, q_misspec_rho, q_misspec_seed)

    y_e, log_ws = oracle.rsample_y(
        experiment=oracle_experiment,
        sigma=sigma,
        return_diagnostics=return_log_ws,
    )
    y_e = jnp.asarray(y_e.detach().cpu().numpy())

    importance_weight_ess = None
    if return_log_ws and log_ws is not None:
        importance_weight_ess = _compute_importance_weight_ess(log_ws)
        ess_str = ", ".join(
            f"{ess:.1f} ({frac * 100:.1f}%)"
            for ess, frac in zip(importance_weight_ess["ess"], importance_weight_ess["ess_fraction"])
        )
        print(
            f"Importance sampling ESS per experiment (out of {int(importance_weight_ess['num_samples'])} "
            f"samples): {ess_str}"
        )

    return ObservationBlock(
        log_tau=log_tau,
        log_pi=log_pi,
        y=y_e,
    ), importance_weight_ess


def _load_fixed_true_theta(
    project_path: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load campaign-level fixed true theta (created by main.py initialization)."""
    true_theta_path = os.path.join(project_path, "true_theta.npz")

    if not os.path.exists(true_theta_path):
        raise FileNotFoundError(
            f"Missing fixed true theta file: {true_theta_path}. "
            "Run main.py initialization first to create it."
        )

    fixed = np.load(true_theta_path)
    log_w = torch.as_tensor(fixed["log_w_true"], dtype=torch.float32, device=device)
    log_v = torch.as_tensor(fixed["log_v_true"], dtype=torch.float32, device=device)
    return log_w, log_v, true_theta_path


def _load_previous_observation_blocks(
    project_path: str,
    round_idx: int,
) -> tuple[list[str], tuple[ObservationBlock, ...]]:
    """Load and validate all previous rounds' observation blocks in order."""
    if round_idx <= 1:
        return [], tuple()

    paths: list[str] = []
    for prev_round in range(1, round_idx):
        path = os.path.join(project_path, f"round_{prev_round}_obs", "observation_block.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing observation block for a previous round. "
                f"Expected file for round {prev_round} at {path}."
            )
        paths.append(path)

    blocks = load_observation_blocks(paths)
    for idx, block in enumerate(blocks, start=1):
        if block.y is None:
            raise ValueError(
                f"Previous observation block for round {idx} has y=None at {paths[idx - 1]}"
            )

    return paths, blocks


def _load_theta_file_for_visualization(
    source_run_dir: str,
    project_path: str,
    round_idx: int,
) -> str:
    # I'll make this just point to the prior samples, as it makes it easier for me to debug
    # the HMC smaple quality when I always compare to round 1 prior samples.
    path = os.path.join(project_path, "round_1_train", "prior_samples.npz")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Missing theta samples for visualization at {path}"
    )


def _load_variational_from_checkpoint(ckpt_path: str, cfg: DictConfig, device: str) -> ExperimentCategorical:
    if cfg.model_type == "categorical":
        variational = ExperimentCategorical(
            vocab_size=len(tuple(cfg.vocabulary)),
            sequence_length=int(cfg.sequence_length),
            num_experiments=int(cfg.num_experiments),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type={cfg.model_type}. Use one of: categorical")
    checkpoint = torch.load(ckpt_path, map_location=device)
    eig_state = checkpoint["eig_model"]
    var_state = {k[len("variational_dist."):]: v for k, v in eig_state.items() if k.startswith("variational_dist.")}
    variational.load_state_dict(var_state, strict=True)
    variational.eval()
    return variational


def _resolve_mcmc_chain_method(device: str, num_chains: int, requested_method: str) -> str:
    """Resolve MCMC chain strategy; auto defaults to vectorized."""
    method = requested_method.lower().strip()
    if method == "auto":
        return "vectorized"
    if method not in ("parallel", "vectorized", "sequential"):
        raise ValueError(
            f"Unsupported mcmc_chain_method={requested_method}. Use one of: auto, parallel, vectorized, sequential"
        )
    return method


def build_plm_oracle(cfg: DictConfig, device: str) -> OraclePLM:
    """Construct the PLM oracle from config. Expensive (loads the PLM checkpoint from disk onto
    the GPU) and fully determined by cfg.oracle.*/cfg.sequence_length/device, none of which vary
    across rounds (or duplicates) within a run - so callers looping over rounds should build this
    once and reuse it, rather than letting _main construct a fresh one every round."""
    return OraclePLM(
        wt_str=cfg.oracle.wt_str,
        continuation_L=int(cfg.sequence_length),
        beta=float(cfg.oracle.beta),
        glm=NormalGLM(),
        device=torch.device(device),
        num_samples=int(cfg.oracle.num_samples),
        batch_size=int(cfg.oracle.batch_size),
        defensive_alpha=float(cfg.oracle.defensive_alpha),
    )


def visualize_umap_for_round(cfg: DictConfig, oracle: Optional[Oracle] = None, reference_only: bool = False) -> None:
    """
    Render the UMAP diagnostic plot (+ a few function plots) for an already-completed round.
    """
    round_idx = int(cfg.round)
    step = round_idx
    project_path = str(cfg.project_path)
    source_run_dir = os.path.join(project_path, f"round_{round_idx}_train")
    out_dir = os.path.join(project_path, f"round_{round_idx}_obs")
    posterior_theta_path = os.path.join(out_dir, "posterior_samples_conditioned.npz")
    if not os.path.exists(posterior_theta_path):
        raise FileNotFoundError(
            f"Missing {posterior_theta_path}; run simulate_observation for round {round_idx} first."
        )

    device = str(cfg.device)
    round_seed = _compose_round_seed(int(cfg.seed), _resolve_duplicate_index(cfg), round_idx)

    acquisition_function = str(cfg.acquisition_function)
    assert acquisition_function in ("lifted", "thompson", "argmax_lifted", "gp_acqf"), (
        f"Unsupported acquisition_function={acquisition_function!r}. Use one of: lifted, thompson, argmax_lifted, gp_acqf"
    )
    # thompson, argmax_lifted and gp_acqf are all downstream-deterministic (one delta-mass sequence
    # per experiment, no variational posterior) - they only differ in how that Experiment gets built.
    is_discretized = acquisition_function != "lifted"

    if acquisition_function == "thompson":
        experiment = load_experiment_from_thompson_samples(source_run_dir=source_run_dir, device=device)
        variational = None
    elif acquisition_function == "gp_acqf":
        raise NotImplementedError(
            "visualize_umap_for_round does not support acquisition_function=gp_acqf yet."
        )
    else:
        best_ckpt_path = os.path.join(source_run_dir, "checkpoints", "best.pt")
        latest_ckpt_path = os.path.join(source_run_dir, "checkpoints", "latest.pt")
        if cfg.early_stopping_patience == 0:
            best_ckpt_path = latest_ckpt_path
        ckpt_path = best_ckpt_path if os.path.exists(best_ckpt_path) else latest_ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found. Tried best={best_ckpt_path} and latest={latest_ckpt_path}"
            )
        experiment = load_experiment_from_eig_checkpoint(ckpt_path=ckpt_path, device=device)
        if acquisition_function == "argmax_lifted":
            experiment = thompson_samples_to_experiment(
                thompson_samples=experiment.argmax(),
                sequence_length=experiment.sequence_length,
                vocab_size=experiment.vocab_size,
                num_experiments=experiment.num_experiments,
            )
            variational = None
        else:
            variational = _load_variational_from_checkpoint(ckpt_path, cfg=cfg, device=device)

    glm = NormalGLM()
    oracle_type = cfg.oracle.type
    assert oracle_type == "plm" or oracle_type == "bump", f"Unsupported oracle.type={oracle_type}. Use one of: plm, bump"

    fitness_function = FitnessFunction(
        num_components=int(cfg.num_components),
        sequence_length=int(cfg.sequence_length),
        vocab_size=len(tuple(cfg.vocabulary)),
        glm=glm,
    ).to(device)

    if oracle_type == "bump":
        log_w_true, log_v_true, _ = _load_fixed_true_theta(project_path=project_path, device=device)
        if oracle is None:
            oracle = OracleBump(
                num_components=int(cfg.oracle.num_components),
                sequence_length=int(cfg.sequence_length),
                vocab_size=len(tuple(cfg.vocabulary)),
                log_w_true=log_w_true,
                log_v_true=log_v_true,
                glm=glm,
            ).to(device)
    elif oracle_type == "plm":
        if oracle is None:
            oracle = build_plm_oracle(cfg, device=device)
    assert oracle is not None, "Oracle should be initialized."

    if reference_only:
        # Every observation block accumulated through this round
        _, conditioning_blocks = _load_previous_observation_blocks(project_path=project_path, round_idx=round_idx + 1)
        umap_together = True
        recompute_umap = True
        umap_output_dir = os.path.join(project_path, "umap_reference")
        os.makedirs(umap_output_dir, exist_ok=True)
    else:
        # Just this round's own observation block
        current_obs_path = os.path.join(out_dir, "observation_block.npz")
        conditioning_blocks = load_observation_blocks([current_obs_path])
        umap_together = bool(cfg.umap_together)
        recompute_umap = bool(cfg.recompute_umap)
        umap_output_dir = out_dir

    x_0_true = oracle.x_0
    f_of_x_0_true = oracle.f(x_0_true.unsqueeze(0)).squeeze(0).item()

    if str(cfg.dtype) == "bf16":
        amp_dtype = torch.bfloat16
    elif str(cfg.dtype) == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Use one of: bf16, fp16")

    posterior_samples_variational = None
    if not is_discretized:
        assert variational is not None, "Variational model should be loaded for non-discretized (lifted) experiments."
        current_block = conditioning_blocks[-1]
        if current_block.y is None:
            raise ValueError(f"Observation block for round {round_idx} has no y")
        y_e_true = torch.as_tensor(
            np.array(current_block.y, dtype=np.float32, copy=True), dtype=torch.float32, device=device
        )
        with autocast(device_type=device, dtype=amp_dtype):
            posterior_samples_variational = variational.sample(
                y_e=y_e_true.unsqueeze(0),
                num_samples=int(cfg.num_samples_simulation),
                chunk_size=256,
            ).squeeze(0)
        posterior_samples_variational = posterior_samples_variational.detach().cpu()

    theta_vis_path = _load_theta_file_for_visualization(
        source_run_dir=source_run_dir,
        project_path=project_path,
        round_idx=round_idx,
    )
    print(f"Using theta visualization source: {theta_vis_path}")
    theta_vis = np.load(theta_vis_path)
    num_x0 = min(int(cfg.umap_num_x_0_samples), int(theta_vis["log_w"].shape[0]))
    log_w_prior = torch.as_tensor(theta_vis["log_w"][:num_x0], dtype=torch.float32, device=device)
    log_v_prior = torch.as_tensor(theta_vis["log_v"][:num_x0], dtype=torch.float32, device=device)
    mode_prior = torch.as_tensor(theta_vis["mode"][:num_x0], dtype=torch.long, device=device)
    x_0_samples = fitness_function.argmax(log_w=log_w_prior, log_v=log_v_prior, candidates=mode_prior)

    # Compute sampled functions for visualization
    f_samples = oracle.sample_f_as_dist(num_samples=int(cfg.num_samples_simulation))

    posterior_np = np.load(posterior_theta_path)

    # Visualize a few random HMC posterior samples as standalone functions.
    num_hmc_function_plots = min(5, int(posterior_np["log_w"].shape[0]))
    if num_hmc_function_plots > 0:
        rng = np.random.default_rng(int(round_seed))
        selected_hmc_indices = rng.choice(
            int(posterior_np["log_w"].shape[0]),
            size=num_hmc_function_plots,
            replace=False,
        )
        for vis_rank, sample_idx in enumerate(selected_hmc_indices):
            sample_idx_int = int(sample_idx)
            log_w_hmc = torch.as_tensor(posterior_np["log_w"][sample_idx_int], dtype=torch.float32, device=device)
            log_v_hmc = torch.as_tensor(posterior_np["log_v"][sample_idx_int], dtype=torch.float32, device=device)
            _ = visualize_function(
                step=step,
                vocabulary=tuple(cfg.vocabulary),
                log_w=log_w_hmc,
                log_v=log_v_hmc,
                path=out_dir,
                simulated_true_name=f"hmc_function_{vis_rank:02d}_idx_{sample_idx_int:06d}",
            )
        print(f"Saved {num_hmc_function_plots} random HMC function visualizations: {selected_hmc_indices.tolist()}")

    num_vis_posterior = min(int(cfg.num_samples_simulation), int(posterior_np["log_w"].shape[0]))
    posterior_log_w = torch.as_tensor(posterior_np["log_w"][:num_vis_posterior], dtype=torch.float32, device=device)
    posterior_log_v = torch.as_tensor(posterior_np["log_v"][:num_vis_posterior], dtype=torch.float32, device=device)
    posterior_mode = torch.as_tensor(posterior_np["mode"][:num_vis_posterior], dtype=torch.long, device=device)
    posterior_x0_samples_mcmc = fitness_function.argmax(
        log_w=posterior_log_w,
        log_v=posterior_log_v,
        candidates=posterior_mode,
    )
    posterior_x0_samples_mcmc_divergent = None
    if "diverging" in posterior_np.files:
        diverging_mask = np.asarray(posterior_np["diverging"], dtype=np.bool_)
        if diverging_mask.ndim >= 2:
            diverging_mask = diverging_mask.reshape(-1)
        diverging_mask = diverging_mask[:num_vis_posterior]
        if np.any(diverging_mask):
            diverging_mask_t = torch.as_tensor(
                diverging_mask,
                dtype=torch.bool,
                device=posterior_x0_samples_mcmc.device,
            )
            posterior_x0_samples_mcmc_divergent = posterior_x0_samples_mcmc[diverging_mask_t]

    _ = visualize_umap_discrete(
        vocabulary=tuple(cfg.vocabulary),
        observation_blocks=conditioning_blocks,
        step=step,
        x_0_samples=x_0_samples,
        num_q_samples_per_experiment=int(cfg.umap_num_q_samples_per_experiment),
        n_neighbors=int(cfg.umap_n_neighbors),
        min_dist=float(cfg.umap_min_dist),
        recompute=recompute_umap,
        umap_together=umap_together,
        output_dir=umap_output_dir,
        umap_cache_dir=project_path,
        x_0_true=x_0_true,
        f_of_x_0_true=f_of_x_0_true,
        f_samples=f_samples,
        posterior_samples_variational=posterior_samples_variational,
        posterior_samples_mcmc=posterior_x0_samples_mcmc,
        posterior_samples_mcmc_divergent=posterior_x0_samples_mcmc_divergent,
        thompson_sampling=is_discretized,
        deduplicate_samples_for_umap=bool(cfg.deduplicate_samples_for_umap),
        umap_color_divergences=bool(cfg.umap_color_divergences),
    )
    print(f"Saved UMAP visualization for round {round_idx} to: {umap_output_dir}")


def _main(cfg: DictConfig, oracle: Optional[Oracle] = None) -> float | None:
    do_visualization = bool(cfg.do_visualization)
    round_idx = int(cfg.round)
    step = int(cfg.round)

    if round_idx < 1:
        raise ValueError(f"round must be >= 1, got {round_idx}")

    base_seed = int(cfg.seed)
    duplicate_idx = _resolve_duplicate_index(cfg)
    round_seed = _compose_round_seed(base_seed, duplicate_idx, round_idx)
    torch.manual_seed(round_seed)

    project_path = str(cfg.project_path)
    source_run_dir = os.path.join(project_path, f"round_{round_idx}_train")
    out_dir = os.path.join(project_path, f"round_{round_idx}_obs")
    out_name = str(cfg.simulation.observation_output_name)
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        print(
            f"Observation already exists at {out_path}. "
            f"Skipping simulate_observation for round {round_idx}."
        )
        return _read_map_ratio_from_metrics(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    diagnostics_csv_path = os.path.join(out_dir, "mcmc_diagnostics.csv")
    if os.path.exists(diagnostics_csv_path):
        os.remove(diagnostics_csv_path)

    device = str(cfg.device)
    num_chains_mcmc = int(cfg.get("num_chains_mcmc", 2))
    if num_chains_mcmc < 1:
        raise ValueError(f"num_chains_mcmc must be >= 1, got {num_chains_mcmc}")
    chain_method_mcmc = _resolve_mcmc_chain_method(
        device=device,
        num_chains=num_chains_mcmc,
        requested_method=str(cfg.get("mcmc_chain_method", "auto")),
    )
    mcmc_target_accept_prob = float(cfg.get("mcmc_target_accept_prob", 0.8))
    mcmc_max_tree_depth = int(cfg.get("mcmc_max_tree_depth", 10))
    mcmc_dense_mass = bool(cfg.get("mcmc_dense_mass", False))
    mcmc_write_diagnostics = bool(cfg.get("mcmc_write_diagnostics", True))

    acquisition_function = str(cfg.acquisition_function)
    assert acquisition_function in ("lifted", "thompson", "argmax_lifted", "gp_acqf"), (
        f"Unsupported acquisition_function={acquisition_function!r}. Use one of: lifted, thompson, argmax_lifted, gp_acqf"
    )

    # Flags for later
    is_discretized = acquisition_function != "lifted"
    needs_posterior_sampling = acquisition_function != "gp_acqf"

    if acquisition_function == "thompson":
        experiment = load_experiment_from_thompson_samples(source_run_dir=source_run_dir, device=device)
        variational = None

    elif acquisition_function == "gp_acqf":
        _, gp_previous_blocks = _load_previous_observation_blocks(
            project_path=project_path,
            round_idx=round_idx,
        )
        gp_candidates = gp_select_candidates(
            previous_blocks=gp_previous_blocks,
            sequence_length=int(cfg.sequence_length),
            vocab_size=len(tuple(cfg.vocabulary)),
            num_experiments=int(cfg.num_experiments),
            sigma=float(cfg.sigma),
            kernel_name=str(cfg.gp.kernel),
            acquisition_name=str(cfg.gp.acquisition),
            num_restarts=int(cfg.gp.num_restarts),
            raw_samples=int(cfg.gp.raw_samples),
            max_batch_size=int(cfg.gp.max_batch_size),
            max_train_points=int(cfg.gp.max_train_points),
            device=device,
        )
        experiment = thompson_samples_to_experiment(
            thompson_samples=gp_candidates,
            sequence_length=int(cfg.sequence_length),
            vocab_size=len(tuple(cfg.vocabulary)),
            num_experiments=int(cfg.num_experiments),
        )
        variational = None

        # Free up cache
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        best_ckpt_path = os.path.join(source_run_dir, "checkpoints", "best.pt")
        latest_ckpt_path = os.path.join(source_run_dir, "checkpoints", "latest.pt")
        if cfg.early_stopping_patience == 0:
            print("Early stopping disabled (early_stopping_patience=0), defaulting to latest.pt for checkpoint.")
            best_ckpt_path = latest_ckpt_path
        ckpt_path = best_ckpt_path if os.path.exists(best_ckpt_path) else latest_ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found. Tried best={best_ckpt_path} and latest={latest_ckpt_path}"
            )
        experiment = load_experiment_from_eig_checkpoint(ckpt_path=ckpt_path, device=device)
        if acquisition_function == "argmax_lifted":
            experiment = thompson_samples_to_experiment(
                thompson_samples=experiment.argmax(),
                sequence_length=experiment.sequence_length,
                vocab_size=experiment.vocab_size,
                num_experiments=experiment.num_experiments,
            )
            variational = None
        else:
            variational = _load_variational_from_checkpoint(ckpt_path, cfg=cfg, device=device)

    glm = NormalGLM()

    oracle_type = cfg.oracle.type
    assert oracle_type == "plm" or oracle_type == "bump", f"Unsupported oracle.type={oracle_type}. Use one of: plm, bump"

    # Base fitness function - used to approximate an oracle
    fitness_function = FitnessFunction(
        num_components=int(cfg.num_components),
        sequence_length=int(cfg.sequence_length),
        vocab_size=len(tuple(cfg.vocabulary)),
        glm=glm,
    ).to(device)
    
    weight_dist = _build_weights_dist(
        mode=str(cfg.weights.mode),
        value=float(cfg.weights.value),
        gamma_concentration=float(cfg.weights.gamma_concentration),
        gamma_rate=float(cfg.weights.gamma_rate),
    )

    # BUMP ORACLE
    if oracle_type == "bump":
        log_w_true, log_v_true, true_theta_path = _load_fixed_true_theta(
            project_path=project_path,
            device=device,
        )
        oracle = OracleBump(
            num_components=int(cfg.oracle.num_components),
            sequence_length=int(cfg.sequence_length),
            vocab_size=len(tuple(cfg.vocabulary)),
            log_w_true=log_w_true,
            log_v_true=log_v_true,
            glm=glm,
        ).to(device)

    # PLM ORACLE
    elif oracle_type == "plm":
        if oracle is None:
            oracle = build_plm_oracle(cfg, device=device)
        log_w_true = None
        log_v_true = None
        true_theta_path = None
    else:
        raise ValueError(f"Unsupported oracle type: {oracle_type}")

    q_misspec_rho = float(cfg.get("q_misspec_rho", 0.0))
    with print_time(f"round={round_idx} oracle query (simulate_observation)"):
        block_current, importance_weight_ess = simulate_observation(
            experiment=experiment,
            oracle=oracle,
            sigma=float(cfg.sigma),
            return_log_ws=bool(cfg.oracle.return_diagnostics),
            q_misspec_rho=q_misspec_rho,
            q_misspec_seed=round_seed if q_misspec_rho > 0.0 else None,
        )

    # Initialize visualization-related variables
    probs_np = None

    if block_current.y is None:
        raise ValueError("Simulated observation unexpectedly has y=None")
    previous_obs_paths, previous_blocks = _load_previous_observation_blocks(
        project_path=project_path,
        round_idx=round_idx,
    )
    conditioning_blocks = (*previous_blocks, block_current)
    save_observation_block(block_current, out_path)

    # Early visualizations
    if do_visualization:
        with print_time(f"round={round_idx} early visualization (tau/pi + true function)"):
            vis_experiment = _experiment_from_observation_block(block_current, device=device)
            _ = visualize_tau_and_pi(
                tau=vis_experiment.probs_syntheses,
                pi=vis_experiment.weights_experiments,
                vocabulary=tuple(cfg.vocabulary),
                step=step,
                num_visualized_syntheses=int(cfg.get("num_visualized_syntheses", 1)),
                output_dir=out_dir,
            )
            if isinstance(oracle, OracleBump):
                assert isinstance(log_w_true, torch.Tensor) and isinstance(log_v_true, torch.Tensor), "Bump oracle should have log_w_true and log_v_true tensors loaded."
                _ = visualize_function(
                    step=step,
                    vocabulary=tuple(cfg.vocabulary),
                    log_w=log_w_true,
                    log_v=log_v_true,
                    path=out_dir,
                )

    if importance_weight_ess is not None:
        importance_weight_ess_path = os.path.join(out_dir, "importance_weight_ess.txt")
        with open(importance_weight_ess_path, "w", encoding="utf-8") as handle:
            handle.write(f"num_samples: {int(importance_weight_ess['num_samples'])}\n")
            handle.write("ess: " + ",".join(f"{v:.8f}" for v in importance_weight_ess["ess"]) + "\n")
            handle.write(
                "ess_fraction: " + ",".join(f"{v:.8f}" for v in importance_weight_ess["ess_fraction"]) + "\n"
            )
        print(f"Saved importance weight ESS diagnostics to: {importance_weight_ess_path}")

    if block_current.y is None:
        raise ValueError("Current observation block has no y")
    y_e_true_np = np.array(block_current.y, dtype=np.float32, copy=True)
    y_e_true = torch.as_tensor(y_e_true_np, dtype=torch.float32, device=device)

    # Defer sampling heavy visualization tensors until later (only if enabled)
    if str(cfg.dtype) == "bf16":
        amp_dtype = torch.bfloat16
    elif str(cfg.dtype) == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Use one of: bf16, fp16")

    ### only for non-discretized (lifted) experiments ###
    if not is_discretized:
        assert variational is not None, "Variational model should be loaded for non-discretized (lifted) experiments."
        # Top-k summary metrics are useful regardless of visualization setting.
        top_k_summary_path = os.path.join(out_dir, "top_k_x0_true.txt")
        with print_time(f"round={round_idx} top-k metrics (beam search)"):
            top_k_metrics = compute_top_k_true_metrics(
                variational_dist=variational,
                y_e=y_e_true,
                oracle=oracle,
                k_for_top_k=int(cfg.k_for_top_k),
                beam_width_for_top_k=int(cfg.beam_width_for_top_k),
                beam_chunk_size=int(cfg.beam_chunk_size),
                device=device,
                amp_dtype=amp_dtype,
            )
        with open(top_k_summary_path, "w", encoding="utf-8") as handle:
            handle.write("context: observed_y\n")
            for key, value in top_k_metrics.items():
                if isinstance(value, list):
                    handle.write(f"{key}: {','.join(f'{v:.8f}' for v in value)}\n")
                else:
                    handle.write(f"{key}: {value:.8f}\n")

        # Heavy visualization-related sampling and plotting (only if requested)
        if do_visualization:
            with print_time(f"round={round_idx} variational visualization (sampling + logoplot)"):
                # Show a probability logo over sequence positions for the variational distribution.
                with autocast(device_type=device, dtype=amp_dtype):
                    probs = variational._dist(y_e=y_e_true.unsqueeze(0)).base_dist.probs
                probs_np = probs.squeeze(0).detach().cpu().numpy()
                logo_df = pd.DataFrame(
                    {token: probs_np[:, i] for i, token in enumerate(cfg.vocabulary)},
                    index=np.arange(1, probs_np.shape[0] + 1),
                )

                fig, ax = plt.subplots(figsize=(10, 4.5))
                logomaker.Logo(logo_df, ax=ax)
                ax.set_title("Variational Categorical Distribution")
                ax.set_ylabel("Token probabilities")
                ax.set_xlabel("Sequence position")
                ax.set_xticks(np.arange(1, variational.sequence_length + 1))
                ax.set_ylim(0.0, 1.0)
                fig.tight_layout()

                logo_path = os.path.join(out_dir, "variational_categorical_logoplot.png")
                fig.savefig(logo_path, dpi=180)
                plt.close(fig)
                print(f"Saved variational categorical logoplot visualization to: {logo_path}")
    ### Done with non-Thompson sampling specific visualizations and metrics. ###

    # Free GPU memory held by the training-time model before JAX MCMC runs.
    if device.startswith("cuda") and torch.cuda.is_available():
        # Variational model is no longer needed after top-k summary is written.
        del variational
        torch.cuda.empty_cache()

    if needs_posterior_sampling:
        #### POSTERIOR SAMPLING
        # Posterior samples used as theta dataset for the next training round.
        posterior_theta_path = os.path.join(out_dir, "posterior_samples_conditioned.npz")
        theta_samples_total = int(cfg.num_samples_train) + int(cfg.num_samples_eval)
        theta_samples_per_chain = max(1, (theta_samples_total + num_chains_mcmc - 1) // num_chains_mcmc)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Once a duplicate's MCMC chains fail (R-hat too high), stay on SVI for every later round
        svi_fallback_marker_path = os.path.join(project_path, "svi_fallback_triggered.txt")
        effective_approximation = str(cfg.posterior_approximation)
        if effective_approximation == "mcmc" and os.path.exists(svi_fallback_marker_path):
            print(
                f"SVI fallback previously triggered for this duplicate (see {svi_fallback_marker_path}); "
                f"using variational posterior instead of mcmc for round {round_idx}."
            )
            effective_approximation = "variational"

        def _run_svi() -> None:
            _ = construct_posterior_samples_svi(
                key=jnp.array([int(round_seed), 0], dtype=jnp.uint32),
                N=int(cfg.num_components),
                L=int(cfg.sequence_length),
                K=len(tuple(cfg.vocabulary)),
                gamma=weight_dist,
                inverse_link_function=glm.inverse_link_function,
                sigma=float(cfg.sigma),
                overlap_reg_strength=float(cfg.overlap_reg_strength),
                num_samples=theta_samples_total,
                observations=conditioning_blocks,
                path=posterior_theta_path,
                num_svi_steps=int(cfg.num_svi_steps),
                svi_learning_rate=float(cfg.svi_learning_rate),
                svi_lr_min_fraction=float(cfg.get("svi_lr_min_fraction", 0.0)),
                svi_warmup_frac=float(cfg.get("svi_warmup_frac", 0.02)),
                svi_num_particles=int(cfg.svi_num_particles),
                svi_grad_clip_ratio=float(cfg.get("svi_grad_clip_ratio", 0.1)),
                svi_adam_eps=float(cfg.get("svi_adam_eps", 1e-6)),
                svi_adam_update_clip=float(cfg.get("svi_adam_update_clip", 3.0)),
                svi_guide_init_scale=float(cfg.get("svi_guide_init_scale", 0.01)),
                svi_check_every=int(cfg.get("svi_check_every", 1000)),
                svi_patience=int(cfg.get("svi_patience", 0)),
                svi_min_delta=float(cfg.get("svi_min_delta", 0.0)),
                diagnostics_csv_path=diagnostics_csv_path,
                init_seed=int(round_seed),
            )

        with print_time(f"round={round_idx} posterior sampling ({effective_approximation})"):
            if effective_approximation == "mcmc":
                try:
                    _ = construct_posterior_samples(
                        key=jnp.array([int(round_seed), 0], dtype=jnp.uint32),
                        N=int(cfg.num_components),
                        L=int(cfg.sequence_length),
                        K=len(tuple(cfg.vocabulary)),
                        gamma=weight_dist,
                        inverse_link_function=glm.inverse_link_function,
                        sigma=float(cfg.sigma),
                        overlap_reg_strength=float(cfg.overlap_reg_strength),
                        num_warmup=int(cfg.num_warmup_mcmc),
                        num_samples=theta_samples_per_chain,
                        observations=conditioning_blocks,
                        path=posterior_theta_path,
                        num_chains=num_chains_mcmc,
                        chain_method=chain_method_mcmc,
                        target_accept_prob=mcmc_target_accept_prob,
                        max_tree_depth=mcmc_max_tree_depth,
                        dense_mass=mcmc_dense_mass,
                        diagnostics_csv_path=diagnostics_csv_path if mcmc_write_diagnostics else None,
                        diagnostics_run_label="theta_dataset_posterior",
                        max_flat_samples=theta_samples_total,
                        diagnostics_include_ess=bool(cfg.get("mcmc_diagnostics_include_ess", True)),
                        init_seed=int(round_seed),
                        rhat_check_site="log_v",
                        rhat_threshold=float(cfg.get("mcmc_svi_fallback_rhat_threshold", 1.1)),
                    )
                except MCMCConvergenceError as exc:
                    print(
                        f"round={round_idx}: MCMC failed to converge ({exc}). Switching to SVI for "
                        "this and every later round of this duplicate."
                    )
                    with open(svi_fallback_marker_path, "w", encoding="utf-8") as handle:
                        handle.write(f"triggered_at_round: {round_idx}\n")
                        handle.write(f"log_v_median_rhat: {exc.rhat:.6f}\n")
                    _run_svi()
            elif effective_approximation == "variational":
                _run_svi()

        posterior_np = np.load(posterior_theta_path)

        ### MAP METRICS
        MAP_metrics_path = os.path.join(out_dir, "map_metrics.txt")
        with print_time(f"round={round_idx} MAP metrics (mode of x* from posterior samples)"):
            posterior_log_w_full = torch.as_tensor(posterior_np["log_w"], dtype=torch.float32, device=device)
            posterior_log_v_full = torch.as_tensor(posterior_np["log_v"], dtype=torch.float32, device=device)
            posterior_mode_full = torch.as_tensor(posterior_np["mode"], dtype=torch.long, device=device)
            MAP_metrics = compute_map_metrics_from_posterior(
                fitness_function=fitness_function,
                oracle=oracle,
                posterior_log_w=posterior_log_w_full,
                posterior_log_v=posterior_log_v_full,
                posterior_mode=posterior_mode_full,
            )
        with open(MAP_metrics_path, "w", encoding="utf-8") as handle:
            for key, value in MAP_metrics.items():
                if np.isscalar(value):
                    handle.write(f"{key}: {value:.8f}\n")
                else:
                    handle.write(f"{key}_shape: {tuple(np.asarray(value).shape)}\n")
            handle.write(f"q_misspec_rho: {q_misspec_rho}\n")

        map_ratio = None
        if "MAP_ratio" in MAP_metrics:
            try:
                map_ratio = float(MAP_metrics["MAP_ratio"])
            except (TypeError, ValueError):
                map_ratio = None
    else:
        # gp_acqf:
        posterior_theta_path = ""
        MAP_metrics_path = os.path.join(out_dir, "map_metrics.txt")
        with print_time(f"round={round_idx} MAP metrics (argmax of GP posterior mean)"):
            MAP_metrics = gp_compute_map_metrics(
                conditioning_blocks=conditioning_blocks,
                oracle=oracle,
                sigma=float(cfg.sigma),
                kernel_name=str(cfg.gp.kernel),
                num_restarts=int(cfg.gp.num_restarts),
                raw_samples=int(cfg.gp.raw_samples),
                max_batch_size=int(cfg.gp.max_batch_size),
                max_train_points=int(cfg.gp.max_train_points),
                device=device,
            )
        with open(MAP_metrics_path, "w", encoding="utf-8") as handle:
            for key, value in MAP_metrics.items():
                if np.isscalar(value):
                    handle.write(f"{key}: {value:.8f}\n")
                else:
                    handle.write(f"{key}_shape: {tuple(np.asarray(value).shape)}\n")
            handle.write(f"q_misspec_rho: {q_misspec_rho}\n")

        map_ratio = None
        if "MAP_ratio" in MAP_metrics:
            try:
                map_ratio = float(MAP_metrics["MAP_ratio"])
            except (TypeError, ValueError):
                map_ratio = None

    ### SAVE METADATA
    np.savez(
        os.path.join(out_dir, "simulation_metadata.npz"),
        source_run_dir=np.asarray(source_run_dir),
        true_theta_path=np.asarray(true_theta_path),
        posterior_theta_path=np.asarray(posterior_theta_path),
        top_k_summary_path=np.asarray(top_k_summary_path) if not is_discretized else np.asarray(""),
        conditioning_observation_paths=np.asarray(previous_obs_paths + [out_path]),
    )
    if needs_posterior_sampling:
        if mcmc_write_diagnostics:
            print(f"Saved MCMC diagnostics to: {diagnostics_csv_path}")
        else:
            print("Skipped MCMC diagnostics CSV (mcmc_write_diagnostics=false)")
    print(f"Saved observation block to: {out_path}")
    return map_ratio


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float | None:
    return _main(cfg)

if __name__ == "__main__":
    main()  # type: ignore[call-arg]
