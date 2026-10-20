from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union
import csv
import os

import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoBNAFNormal, AutoIAFNormal
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from numpyro.distributions.transforms import OrderedTransform

import jax
import jax.random as random
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch

from function import FitnessFunction, compute_log_linear_predictor
from experiment import Experiment
from oracle import Oracle


def inverse_link_function(x):
    return x

def _build_weights_dist(mode: str, value: float, gamma_concentration: float, gamma_rate: float) -> Union[float, dist.Distribution]:
    if mode == "fixed":
        return value
    elif mode == "gamma":
        return dist.Gamma(concentration=gamma_concentration, rate=gamma_rate)
    elif mode == "shifted_gamma":
        # p(w) = value + Gamma(gamma_concentration, gamma_rate).
        return dist.TransformedDistribution(
            dist.Gamma(concentration=gamma_concentration, rate=gamma_rate),
            dist.transforms.AffineTransform(loc=value, scale=1.0, domain=dist.constraints.positive),
        )
    else:
        raise ValueError(f"Invalid weights mode: {mode}. Use one of: fixed, gamma, shifted_gamma")


def pairwise_overlap_penalty(log_v) -> jnp.ndarray:
    """Compute log(sum_{n<m} prod_l <v_nl, v_ml>) for v=exp(log_v), log_v shape [N, L, K]."""
    n_components = log_v.shape[0]
    if n_components < 2:
        return jnp.asarray(0.0)

    # inner product <v_nl, v_ml>
    log_pairwise_inner = jax.nn.logsumexp(log_v[:, None, :, :] + log_v[None, :, :, :], axis=-1)  # [N, N, L]
    # product over l: prod_l <v_nl, v_ml>
    log_pairwise_product = jnp.sum(log_pairwise_inner, axis=-1)  # [N, N]
    # strict upper triangle only (n < m)
    upper_i, upper_j = jnp.triu_indices(n_components, k=1)
    return jax.nn.logsumexp(log_pairwise_product[upper_i, upper_j])

class ObservationBlock(NamedTuple):
    """observation block.

    Shapes:
    - log_tau: [S, L, K]
    - log_pi: [E, S]
    - y: [E] or None
    """

    log_tau: jnp.ndarray
    log_pi: jnp.ndarray
    y: Optional[jnp.ndarray] = None


VectorizedObservationData = Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]


def save_observation_block(block: ObservationBlock, path: str) -> str:
    """Save a single observation block to .npz for reuse in later rounds."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    y_np = np.asarray(jax.device_get(block.y)) if block.y is not None else np.asarray([], dtype=np.float32)
    np.savez(
        path,
        log_tau=np.asarray(jax.device_get(block.log_tau)),
        log_pi=np.asarray(jax.device_get(block.log_pi)),
        y=y_np,
        has_y=np.asarray(block.y is not None, dtype=np.bool_),
    )
    return path


def load_observation_block(path: str) -> ObservationBlock:
    """Load one observation block saved by save_observation_block."""
    data = np.load(path)
    log_tau = jnp.asarray(data["log_tau"])
    log_pi = jnp.asarray(data["log_pi"])
    has_y = bool(np.asarray(data["has_y"]).item()) if "has_y" in data else True
    y = jnp.asarray(data["y"]) if has_y else None
    return ObservationBlock(log_tau=log_tau, log_pi=log_pi, y=y)


def load_observation_blocks(paths: Sequence[str]) -> tuple[ObservationBlock, ...]:
    """Load multiple observation block files in order."""
    return tuple(load_observation_block(p) for p in paths)


def _pack_samples_for_disk(samples: dict[str, jnp.ndarray], K: int) -> dict[str, np.ndarray]:
    """Convert JAX sample arrays to compact NumPy arrays for disk storage."""
    packed: dict[str, np.ndarray] = {}
    for name, value in samples.items():
        arr = np.asarray(jax.device_get(value))
        if name == "mode":
            arr = arr.astype(np.uint8 if K <= 256 else np.uint16, copy=False)
        elif arr.dtype.kind == "f" and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        packed[name] = arr
    return packed


def _normalize_mcmc_key(key: jnp.ndarray, num_chains: int) -> jnp.ndarray:
    """Normalize rng key shape for single-chain or multi-chain MCMC runs."""
    key = jnp.asarray(key, dtype=jnp.uint32)
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {num_chains}")

    if num_chains == 1:
        if key.shape == (2,):
            return key
        if key.ndim == 2 and key.shape[1] == 2:
            return key[0]
        raise ValueError(
            f"Expected PRNG key shape (2,) for a single chain, got {tuple(key.shape)}"
        )

    if key.shape == (2,):
        return jax.random.split(key, num_chains)
    if key.ndim == 2 and key.shape == (num_chains, 2):
        return key
    raise ValueError(
        "Expected PRNG key shape (2,) or (num_chains, 2) for multi-chain MCMC, "
        f"got {tuple(key.shape)} with num_chains={num_chains}"
    )


class NoUsablePosteriorSamplesError(RuntimeError):
    """
    Raised when numerical instability
    """


def _init_params_from_posterior_samples(
    posterior_npz_path: str,
    num_chains: int,
    gamma: float | dist.Distribution,
    seed: int,
) -> dict[str, jnp.ndarray]:
    """
    Build NUTS init_params from a previous round's saved posterior samples
    """
    data = np.load(posterior_npz_path)
    log_w_all = np.asarray(data["log_w"], dtype=np.float32)  # [M, N]
    log_v_all = np.asarray(data["log_v"], dtype=np.float32)  # [M, N, L, K]
    num_available = log_w_all.shape[0]

    log_w0_all = np.max(log_w_all, axis=-1)  # [M]
    usable = np.isfinite(log_w_all).all(axis=-1) & np.isfinite(log_v_all).all(axis=(1, 2, 3))
    if isinstance(gamma, dist.Distribution):
        usable &= np.isfinite(np.exp(log_w0_all))
    usable_idx = np.flatnonzero(usable)
    if usable_idx.size == 0:
        raise NoUsablePosteriorSamplesError(
            f"No usable posterior samples to initialize from at {posterior_npz_path}: all "
            f"{num_available} saved samples are non-finite or overflow when reconstructing an init."
        )
    if usable_idx.size < num_available:
        print(
            f"{num_available - usable_idx.size}/{num_available} saved posterior samples at "
            f"{posterior_npz_path} are non-finite or overflow when reconstructing an init; "
            "excluding them from initialization."
        )

    rng = np.random.default_rng(seed)
    replace = usable_idx.size < num_chains
    if replace:
        print(
            f"Only {usable_idx.size} usable saved posterior samples available at "
            f"{posterior_npz_path}, need num_chains={num_chains}; sampling with replacement."
        )
    idx = rng.choice(usable_idx, size=num_chains, replace=replace)
    log_w = log_w_all[idx]  # [num_chains, N]
    log_v = log_v_all[idx]  # [num_chains, N, L, K]

    log_w0 = log_w0_all[idx]  # [num_chains]
    alpha_init = log_w - log_w0[:, None]  # [num_chains, N], ordered ascending, max 0

    eps = 1e-2
    logits_init = (log_v - np.min(log_v, axis=-1, keepdims=True)) + eps  # [num_chains, N, L, K], > 0

    init_params: dict[str, jnp.ndarray] = {
        "alpha": jnp.asarray(alpha_init, dtype=jnp.float32),
        "logits": jnp.asarray(logits_init, dtype=jnp.float32),
    }
    if isinstance(gamma, dist.Distribution):
        init_params["w_0"] = jnp.asarray(np.exp(log_w0), dtype=jnp.float32)
    return init_params


def _run_nuts_mcmc(
    key,
    N: int,
    L: int,
    K: int,
    gamma: float | dist.Distribution,
    inverse_link_function,
    sigma: float,
    overlap_reg_strength: float,
    observation_data,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    chain_method: str,
    target_accept_prob: float,
    max_tree_depth: int,
    dense_mass: bool,
) -> MCMC:
    """
    Construct a NUTS kernel/MCMC object and run it once.
    """
    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(
        key,
        N=N,
        L=L,
        K=K,
        gamma=gamma,
        inverse_link_function=inverse_link_function,
        sigma=sigma,
        overlap_reg_strength=overlap_reg_strength,
        observation_data=observation_data,
        extra_fields=("diverging", "energy", "num_steps", "accept_prob"),
    )
    return mcmc


def _iter_numeric_sites(samples_by_chain: dict[str, jnp.ndarray]) -> list[tuple[str, np.ndarray]]:
    sites: list[tuple[str, np.ndarray]] = []
    for site, value in samples_by_chain.items():
        arr = jnp.asarray(value)
        if arr.ndim < 2:
            continue
        if not jnp.issubdtype(arr.dtype, jnp.number):
            continue
        arr_float = arr.astype(jnp.float32)
        arr_np = np.asarray(jax.device_get(arr_float))
        sites.append((site, arr_np))
    return sites


def _nan_stats(values: np.ndarray) -> tuple[float, float, float]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(valid)), float(np.min(valid)), float(np.max(valid))


def _collect_mcmc_diagnostics_full_rows(
    samples_by_chain: dict[str, jnp.ndarray],
    run_label: str,
    num_divergences: int,
    include_ess: bool = True,
) -> list[dict[str, Any]]:
    """
    Collect per-element ESS/R-hat diagnostics for every site.
    """
    rows: list[dict[str, Any]] = []
    for site, arr_np in _iter_numeric_sites(samples_by_chain):
        if arr_np.shape[0] > 1:
            rhat = np.asarray(gelman_rubin(arr_np), dtype=np.float64)
        else:
            rhat = None
        if include_ess:
            ess = np.asarray(effective_sample_size(arr_np), dtype=np.float64)
        else:
            ess = np.full(rhat.shape if rhat is not None else arr_np.shape[2:], np.nan, dtype=np.float64)
        if rhat is None:
            rhat = np.full(ess.shape, np.nan, dtype=np.float64)

        for idx in np.ndindex(ess.shape):
            idx_str = "" if len(idx) == 0 else "[{}]".format(",".join(map(str, idx)))
            rows.append(
                {
                    "run_label": run_label,
                    "site": f"{site}{idx_str}",
                    "ess": float(ess[idx]),
                    "r_hat": float(rhat[idx]),
                    "num_divergences": int(num_divergences),
                }
            )
    return rows


def _collect_mcmc_diagnostics_simple_rows(
    samples_by_chain: dict[str, jnp.ndarray],
    run_label: str,
    num_divergences: int,
    include_ess: bool = True,
) -> list[dict[str, Any]]:
    """
    Collect aggregated ESS/R-hat diagnostics per base site.
    """
    rows: list[dict[str, Any]] = []
    for site, arr_np in _iter_numeric_sites(samples_by_chain):
        if arr_np.shape[0] > 1:
            rhat = np.asarray(gelman_rubin(arr_np), dtype=np.float64).reshape(-1)
        else:
            rhat = None
        if include_ess:
            ess = np.asarray(effective_sample_size(arr_np), dtype=np.float64).reshape(-1)
        else:
            flat_size = int(np.prod(arr_np.shape[2:])) if arr_np.ndim > 2 else 1
            ess = np.full(rhat.shape if rhat is not None else (flat_size,), np.nan, dtype=np.float64)
        if rhat is None:
            rhat = np.full(ess.shape, np.nan, dtype=np.float64)

        ess_mean, ess_min, ess_max = _nan_stats(ess)
        rhat_mean, rhat_min, rhat_max = _nan_stats(rhat)
        rows.append(
            {
                "run_label": run_label,
                "site": site,
                "ess_mean": ess_mean,
                "ess_min": ess_min,
                "ess_max": ess_max,
                "r_hat_mean": rhat_mean,
                "r_hat_min": rhat_min,
                "r_hat_max": rhat_max,
                "num_divergences": int(num_divergences),
            }
        )
    return rows


def append_mcmc_diagnostics_csv(path: str, rows: Sequence[dict[str, Any]]) -> str:
    """Append diagnostics rows to CSV."""
    if len(rows) == 0:
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


def write_divergence_interrogation_report(
    summary_path: str,
    samples_by_chain: dict[str, jnp.ndarray],
    extra_fields: dict[str, Any],
    max_rows: int = 5000,
) -> tuple[str, str]:
    """Write divergence localization summary and a table of divergent draws."""
    diverging = np.asarray(jax.device_get(extra_fields.get("diverging", np.asarray([], dtype=np.bool_))))
    if diverging.ndim == 1:
        diverging = diverging[None, :]

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    table_path = f"{os.path.splitext(summary_path)[0]}_divergent_draws.csv"

    if diverging.size == 0:
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("No diverging field available in extra_fields.\n")
        return summary_path, table_path

    features: dict[str, np.ndarray] = {}

    def add_feature(name: str, arr: Any, reducer: str = "mean") -> None:
        arr_np = np.asarray(jax.device_get(arr), dtype=np.float64)
        if arr_np.ndim < 2:
            return
        if arr_np.ndim == 2:
            features[name] = arr_np
            return
        flat = arr_np.reshape(arr_np.shape[0], arr_np.shape[1], -1)
        if reducer == "max":
            features[name] = np.max(flat, axis=-1)
        elif reducer == "min":
            features[name] = np.min(flat, axis=-1)
        elif reducer == "std":
            features[name] = np.std(flat, axis=-1)
        else:
            features[name] = np.mean(flat, axis=-1)

    if "log_w" in samples_by_chain:
        add_feature("log_w_mean", samples_by_chain["log_w"], reducer="mean")
    if "beta" in samples_by_chain:
        add_feature("beta_mean", samples_by_chain["beta"], reducer="mean")
        add_feature("beta_max", samples_by_chain["beta"], reducer="max")
    if "overlap_penalty" in samples_by_chain:
        add_feature("overlap_penalty", samples_by_chain["overlap_penalty"])
    if "log_v" in samples_by_chain:
        add_feature("log_v_std", samples_by_chain["log_v"], reducer="std")

    if "energy" in extra_fields:
        add_feature("energy", extra_fields["energy"])
    if "num_steps" in extra_fields:
        add_feature("num_steps", extra_fields["num_steps"])
    if "accept_prob" in extra_fields:
        add_feature("accept_prob", extra_fields["accept_prob"])

    nondiverging = ~diverging
    div_idx = np.argwhere(diverging)

    def qstats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        selected = values[mask]
        if selected.size == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.quantile(selected, 0.05)),
            float(np.quantile(selected, 0.50)),
            float(np.quantile(selected, 0.95)),
        )

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"num_divergent_draws: {int(div_idx.shape[0])}\n")
        handle.write(f"total_draws: {int(diverging.size)}\n")
        handle.write(f"divergence_rate: {float(np.mean(diverging.astype(np.float64))):.6f}\n")
        for name, arr in features.items():
            d05, d50, d95 = qstats(arr, diverging)
            n05, n50, n95 = qstats(arr, nondiverging)
            handle.write(
                f"{name}: div(q05={d05:.6g}, q50={d50:.6g}, q95={d95:.6g}) "
                f"nondiv(q05={n05:.6g}, q50={n50:.6g}, q95={n95:.6g})\n"
            )

    fieldnames = ["chain", "draw", *features.keys()]
    with open(table_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chain_i, draw_i in div_idx[:max_rows]:
            row: dict[str, Any] = {"chain": int(chain_i), "draw": int(draw_i)}
            for name, arr in features.items():
                row[name] = float(arr[chain_i, draw_i])
            writer.writerow(row)

    return summary_path, table_path


def _validate_observations(
    observations: Sequence[ObservationBlock],
    L: int,
    K: int,
) -> tuple[ObservationBlock, ...]:
    """Validate and match a sequence of observation blocks."""

    blocks = tuple(observations)
    for idx, observation in enumerate(blocks):
        assert observation.log_tau.ndim == 3, (
            f"observations[{idx}].log_tau must have shape (S, L, K), got {observation.log_tau.shape}"
        )
        assert observation.log_pi.ndim == 2, (
            f"observations[{idx}].log_pi must have shape (E, S), got {observation.log_pi.shape}"
        )

        S_i, L_tau, K_tau = observation.log_tau.shape
        E_i, S_pi = observation.log_pi.shape
        assert S_i == S_pi, (
            f"S mismatch in observations[{idx}]: log_tau has {S_i}, log_pi has {S_pi}"
        )
        assert L_tau == L, (
            f"Sequence length mismatch in observations[{idx}]: log_tau has {L_tau}, expected {L}"
        )
        assert K_tau == K, (
            f"Vocab size mismatch in observations[{idx}]: log_tau has {K_tau}, expected {K}"
        )
        if observation.y is not None:
            assert observation.y.shape == (E_i,), (
                f"y shape mismatch in observations[{idx}]: got {observation.y.shape}, expected {(E_i,)}"
            )

    return blocks

def _vectorize_blocks(blocks: Sequence[ObservationBlock]) -> VectorizedObservationData:
    """
    Vectorize a sequence of observation blocks. Assume that the blocks have the same shape pi, tau, and y.
    """
    assert len(blocks) > 0, "No observation blocks to vectorize"
    assert all(block.log_pi.shape == blocks[0].log_pi.shape for block in blocks), "All blocks must have the same log_pi shape"
    assert all(block.log_tau.shape == blocks[0].log_tau.shape for block in blocks), "All blocks must have the same log_tau shape"

    log_pi = jnp.stack([block.log_pi for block in blocks], axis=0)  # [num_rounds, E, S]
    log_tau = jnp.stack([block.log_tau for block in blocks], axis=0)  # [num_rounds, S, L, K]

    has_y = [block.y is not None for block in blocks]
    if any(has_y) and not all(has_y):
        raise ValueError("All observation blocks must either include y or all have y=None")
    if all(has_y):
        y = jnp.stack([block.y for block in blocks], axis=0)  # [num_rounds, E] # type: ignore[list-item]
    else:
        y = None

    return log_pi, log_tau, y


def _prepare_observation_data(
    observations: Optional[Sequence[ObservationBlock]],
    L: int,
    K: int,
    require_y: bool = False,
) -> Optional[VectorizedObservationData]:
    """Validate and vectorize observation blocks before calling NumPyro model code."""
    if observations is None:
        return None

    blocks = _validate_observations(observations=observations, L=L, K=K)
    if len(blocks) == 0:
        raise ValueError("Provide at least one observation block")
    if require_y:
        for idx, block in enumerate(blocks):
            if block.y is None:
                raise ValueError(
                    f"observations[{idx}] has y=None. Posterior conditioning requires observed y values."
                )

    return _vectorize_blocks(blocks)


def model(N: int, 
          L: int, 
          K: int, 
          gamma: float | dist.Distribution, 
          inverse_link_function, 
          sigma: float, 
          overlap_reg_strength: float = 0.0,
          observation_data: Optional[VectorizedObservationData] = None,
):
    '''
     N: number of bump functions
     L: sequence length
     K: number of tokens in the vocabulary
     gamma: either a float or a numpyro Distribution instance. The prior on w.
     inverse_link_function: the inverse link function to use for the GLM likelihood.
     sigma: the standard deviation for the likelihood.
     observation_data: optional tuple with shapes
          log_pi: (R, E, S),
          log_tau: (R, S, L, K),
          y: (R, E) or None.
    '''
    # Priors
    if isinstance(gamma, (float, int)):
        w_0 = gamma
    elif isinstance(gamma, dist.Distribution):
        w_0 = numpyro.sample('w_0', gamma)
    else:
        raise ValueError(f"gamma must be a float, int, or numpyro Distribution instance, got {type(gamma)}")

    # log_u sampling
    # Ordered for label-switching of the N! relabelings.
    log_u = numpyro.sample(
        'alpha',
        dist.TransformedDistribution(
            dist.Normal(0, 1).expand((N,)).to_event(1),
            OrderedTransform(),
        )
    )
    # For numerical stability
    log_u = jnp.clip(log_u, a_max=60.0)
    
    log_u = log_u - jnp.max(log_u) # Max is 0

    # log_w sampling
    log_w0 = jnp.log(w_0)
    log_w = numpyro.deterministic('log_w', log_w0 + log_u)  # shape (N,)

    # Exponential for sparse regression
    logits = numpyro.sample(
        'logits',
        # dist.HalfCauchy(2.0).expand((N, L, K))
        dist.Exponential(0.5).expand((N, L, K))
    )
    # For numerical stability
    logits = jnp.clip(logits, a_max=60.0)

    log_v = numpyro.deterministic('log_v', logits - jnp.max(logits, axis=-1, keepdims=True))  # shape (N, L, K)
    mode = numpyro.deterministic('mode', jnp.argmax(log_v, axis=-1))  # shape (N, L).

    # Add regularization with product over positions.
    if overlap_reg_strength > 0.0 and N > 1:
        log_overlap_penalty = pairwise_overlap_penalty(log_v)
        log_overlap_penalty_normalized = (
            log_overlap_penalty
            - (L * jnp.log(jnp.asarray(float(K))))
        )
        overlap_penalty = numpyro.deterministic('overlap_penalty', jnp.exp(log_overlap_penalty_normalized))
        numpyro.factor('overlap_regularizer', -overlap_reg_strength * overlap_penalty)

    # If prepared observation tensors are provided, add one likelihood term per block.
    if observation_data is not None:
        log_pi, log_tau, y = observation_data
        # shapes:
        # log_w: [N]
        # log_v: [N, L, K]
        # log_pi: [R, E, S]
        # log_tau: [R, S, L, K]
        # y: [R, E] or None
        R, E = log_pi.shape[0], log_pi.shape[1]

        log_linear_predictor = jax.vmap(
            lambda log_pi_i, log_tau_i: compute_log_linear_predictor(
                log_w,
                log_v,
                log_pi_i,
                log_tau_i,
                inverse_link_function=inverse_link_function,
            )
        )(log_pi, log_tau)  # shape (num_rounds, E)
        
        # For numerical stability
        log_linear_predictor = jnp.clip(log_linear_predictor, a_max=30.0)

        linear_predictor = jnp.exp(log_linear_predictor)  # [E]
        mu = inverse_link_function(linear_predictor)  # [E]

        with numpyro.plate("rounds", R, dim=-2):
            with numpyro.plate("experiments", E, dim=-1):
                if y is None:
                    numpyro.sample("obs", dist.Normal(mu, sigma))
                else:
                    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def predictive(key, 
               N: int, 
               L: int, 
               K: int, 
               gamma: float | dist.Distribution, 
               inverse_link_function, 
               sigma: float,
               overlap_reg_strength: float,
               num_samples: int,
               observations: Optional[Sequence[ObservationBlock]] = None,
               posterior_samples: Optional[dict[str, jnp.ndarray]] = None,
               return_sites: Optional[Sequence[str]] = ("log_w", "log_v", "mode"),
               path: Optional[str] = None,
) -> dict[str, jnp.ndarray]:
    """Helper function to construct predictive."""
    observation_data = _prepare_observation_data(
        observations=observations,
        L=L,
        K=K,
        require_y=False,
    )

    predictive = Predictive(
        model, 
        posterior_samples=posterior_samples,
        num_samples=num_samples,
        return_sites=None if return_sites is None else tuple(return_sites),
    )
    predictive_samples = predictive(
        key,
        N=N,
        L=L,
        K=K,
        gamma=gamma,
        inverse_link_function=inverse_link_function,
        sigma=sigma,
        overlap_reg_strength=overlap_reg_strength,
        observation_data=observation_data,
    )
    if path is not None:
        np.savez(path, **_pack_samples_for_disk(predictive_samples, K=K))  # type: ignore[arg-type]

    return predictive_samples

class MCMCConvergenceError(RuntimeError):
    """
    Raised by construct_posterior_samples when rhat_check_site's R-hat exceeds rhat_threshold.
    Indicates that NUTS has catastrophic failure.
    Catch this and fall back to SVI.
    """

    def __init__(self, message: str, rhat: float):
        super().__init__(message)
        self.rhat = rhat


def _median_rhat_for_site(mcmc: MCMC, site: str) -> Optional[float]:
    samples_grouped = mcmc.get_samples(group_by_chain=True)
    if site not in samples_grouped:
        raise KeyError(f"Site '{site}' not found in MCMC samples; available sites: {list(samples_grouped.keys())}")
    arr = np.asarray(jax.device_get(samples_grouped[site]), dtype=np.float64)
    if arr.shape[0] < 2:
        return None
    rhat = np.asarray(gelman_rubin(arr), dtype=np.float64)
    return float(np.nanmedian(rhat))


def construct_observationblock_from_experiment(experiment: Experiment, y: jnp.ndarray) -> ObservationBlock:
    """Helper function to construct observation blocks from an experiment."""
    exp_log_tau = jnp.asarray(experiment.log_probs_syntheses.detach().cpu().numpy()) # shape (S, L, K)
    exp_log_pi = jnp.asarray(experiment.log_weights_experiments.detach().cpu().numpy()) # shape (E, S)
    block = ObservationBlock(log_tau=exp_log_tau, log_pi=exp_log_pi, y=y)
    return block

def construct_posterior_samples(key, 
                                N: int, 
                                L: int, 
                                K: int, 
                                gamma: float | dist.Distribution, 
                                inverse_link_function, 
                                sigma: float,
                                overlap_reg_strength: float,
                                num_warmup: int,
                                num_samples: int,
                                observations: Sequence[ObservationBlock],
                                return_sites: Optional[Sequence[str]] = ("log_w", "log_v", "mode"),
                                path: Optional[str] = None,
                                num_chains: int = 1,
                                chain_method: str = "parallel",
                                target_accept_prob: float = 0.8,
                                max_tree_depth: int = 10,
                                dense_mass: bool = False,
                                diagnostics_csv_path: Optional[str] = None,
                                diagnostics_run_label: str = "posterior",
                                diagnostics_trace_path: Optional[str] = None,
                                diagnostics_summary_path: Optional[str] = None,
                                max_flat_samples: Optional[int] = None,
                                diagnostics_include_ess: bool = True,
                                init_seed: int = 0,
                                rhat_check_site: Optional[str] = None,
                                rhat_threshold: float = 1.1,
) -> dict[str, jnp.ndarray]:
    """Helper function to construct posterior samples."""
    observation_data = _prepare_observation_data(
        observations=observations,
        L=L,
        K=K,
        require_y=True,
    )
    if chain_method not in ("parallel", "vectorized", "sequential"):
        raise ValueError(
            f"Invalid chain_method: {chain_method}. Use one of: parallel, vectorized, sequential"
        )
    if not (0.0 < target_accept_prob < 1.0):
        raise ValueError(
            f"target_accept_prob must be in (0, 1), got {target_accept_prob}"
        )
    if max_tree_depth < 1:
        raise ValueError(f"max_tree_depth must be >= 1, got {max_tree_depth}")
    if max_flat_samples is not None and max_flat_samples <= 0:
        raise ValueError(f"max_flat_samples must be positive, got {max_flat_samples}")

    run_key = _normalize_mcmc_key(key=jnp.asarray(key, dtype=jnp.uint32), num_chains=num_chains)

    mcmc = _run_nuts_mcmc(
        run_key,
        N=N, L=L, K=K,
        gamma=gamma,
        inverse_link_function=inverse_link_function,
        sigma=sigma,
        overlap_reg_strength=overlap_reg_strength,
        observation_data=observation_data,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
    )

    # Write diagnostics
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    if diagnostics_csv_path is not None:
        diagnostics_root, _ = os.path.splitext(diagnostics_csv_path)
        if diagnostics_trace_path is None:
            diagnostics_trace_path = f"{diagnostics_root}_trace.png"
        if diagnostics_summary_path is None:
            diagnostics_summary_path = f"{diagnostics_root}_summary.txt"

        samples_grouped = mcmc.get_samples(group_by_chain=True)
        diverging = np.asarray(jax.device_get(extra_fields.get("diverging", np.asarray([], dtype=np.bool_))))
        num_divergences = int(np.sum(diverging.astype(np.int32))) if diverging.size > 0 else 0
        full_rows = _collect_mcmc_diagnostics_full_rows(
            samples_by_chain=samples_grouped,
            run_label=diagnostics_run_label,
            num_divergences=num_divergences,
            include_ess=diagnostics_include_ess,
        )
        append_mcmc_diagnostics_csv(path=diagnostics_csv_path, rows=full_rows)
        simple_rows = _collect_mcmc_diagnostics_simple_rows(
            samples_by_chain=samples_grouped,
            run_label=diagnostics_run_label,
            num_divergences=num_divergences,
            include_ess=diagnostics_include_ess,
        )
        append_mcmc_diagnostics_csv(path=f"{diagnostics_root}_simple.csv", rows=simple_rows)
        if diagnostics_summary_path is not None:
            write_divergence_interrogation_report(
                summary_path=diagnostics_summary_path,
                samples_by_chain=samples_grouped,
                extra_fields=extra_fields,
            )

        ## Hard-coded trace for key sites only.
        if diagnostics_trace_path is not None:
            trace_series: list[tuple[str, np.ndarray]] = []
            if "log_w" in samples_grouped:
                log_w_chain = np.asarray(jax.device_get(samples_grouped["log_w"]))
                trace_series.append(("log_w_0", log_w_chain[:, :, 0]))
            if "overlap_penalty" in samples_grouped:
                overlap_chain = np.asarray(jax.device_get(samples_grouped["overlap_penalty"]))
                trace_series.append(("overlap_penalty", overlap_chain))
            if "log_v" in samples_grouped:
                log_v_chain = np.asarray(jax.device_get(samples_grouped["log_v"]))
                fixed_log_v_indices = ((0, 0, 0), (0, min(L - 1, 1), 0), (0, 0, min(K - 1, 1)))
                for n_i, l_i, k_i in fixed_log_v_indices:
                    trace_series.append(
                        (f"log_v_{n_i}_{l_i}_{k_i}", log_v_chain[:, :, n_i, l_i, k_i])
                    )

            if len(trace_series) > 0:
                fig, axes = plt.subplots(len(trace_series), 1, figsize=(10, 2.4 * len(trace_series)), sharex=True)
                axes_arr = np.atleast_1d(axes)
                for ax, (name, values_by_chain) in zip(axes_arr, trace_series):
                    for chain_idx in range(values_by_chain.shape[0]):
                        ax.plot(values_by_chain[chain_idx], alpha=0.8, linewidth=0.8)
                    ax.set_title(name)
                axes_arr[-1].set_xlabel("draw")
                fig.tight_layout()
                os.makedirs(os.path.dirname(diagnostics_trace_path) or ".", exist_ok=True)
                fig.savefig(diagnostics_trace_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

    if rhat_check_site is not None:
        median_rhat = _median_rhat_for_site(mcmc, rhat_check_site)
        if median_rhat is not None and median_rhat > rhat_threshold:
            raise MCMCConvergenceError(
                f"Median R-hat for site '{rhat_check_site}' reached {median_rhat:.3f} (threshold "
                f"{rhat_threshold}); chains disagree too much to trust this posterior.",
                rhat=median_rhat,
            )

    diverging_flat: Optional[np.ndarray] = None
    if "diverging" in extra_fields:
        diverging = np.asarray(jax.device_get(extra_fields["diverging"]))
        if diverging.ndim >= 2:
            diverging_flat = diverging.reshape(-1)
        else:
            diverging_flat = diverging
        diverging_flat = diverging_flat.astype(np.bool_, copy=False)

    samples_all = mcmc.get_samples()
    if max_flat_samples is not None:
        samples_all = {k: v[:max_flat_samples] for k, v in samples_all.items()}
        if diverging_flat is not None:
            diverging_flat = diverging_flat[:max_flat_samples]
    if return_sites is None:
        samples = samples_all
    else:
        missing = [k for k in return_sites if k not in samples_all]
        if missing:
            raise KeyError(f"Requested return_sites are missing from posterior samples: {missing}")
        samples = {k: samples_all[k] for k in return_sites}
    if path is not None:
        packed = _pack_samples_for_disk(samples, K=K)
        if diverging_flat is not None:
            packed["diverging"] = diverging_flat
        np.savez(path, **packed)  # type: ignore[arg-type]

    return samples


def _run_svi_until_plateau(
    svi: SVI,
    key,
    num_steps: int,
    check_every: int,
    patience: int,
    min_delta: float,
    **model_kwargs,
) -> SVIRunResult:
    """
    Run SVI for num_steps
    """
    if patience <= 0:
        return svi.run(key, num_steps, progress_bar=False, stable_update=True, **model_kwargs)

    svi_state = svi.init(key, **model_kwargs)

    def _scan_body(state, _):
        return svi.stable_update(state, **model_kwargs)

    all_losses = []
    best_window_mean: Optional[float] = None
    best_window_step = 0
    steps_run = 0
    while steps_run < num_steps:
        window_steps = min(check_every, num_steps - steps_run)
        svi_state, window_losses = jax.lax.scan(_scan_body, svi_state, None, length=window_steps)
        steps_run += window_steps
        window_losses_np = np.asarray(window_losses)
        all_losses.append(window_losses_np)
        window_mean = float(np.mean(window_losses_np))
        if best_window_mean is None or window_mean < best_window_mean - min_delta:
            best_window_mean = window_mean
            best_window_step = steps_run
        elif steps_run - best_window_step >= patience * check_every:
            print(
                f"SVI ELBO plateaued: windowed-mean loss (over {check_every}-step windows) "
                f"hasn't improved by {min_delta} since step {best_window_step}; stopping at "
                f"step {steps_run}/{num_steps}."
            )
            break

    losses = jnp.asarray(np.concatenate(all_losses))
    params = svi.get_params(svi_state)
    return SVIRunResult(params=params, state=svi_state, losses=losses)


def construct_posterior_samples_svi(
    key,
    N: int,
    L: int,
    K: int,
    gamma: float | dist.Distribution,
    inverse_link_function,
    sigma: float,
    overlap_reg_strength: float,
    num_samples: int,
    observations: Sequence[ObservationBlock],
    diagnostics_csv_path: str,
    return_sites: Optional[Sequence[str]] = ("log_w", "log_v", "mode"),
    path: Optional[str] = None,
    num_svi_steps: int = 100000,
    svi_learning_rate: float = 0.001,
    svi_lr_min_fraction: float = 0.0,
    svi_warmup_frac: float = 0.02,
    svi_num_particles: int = 8,
    svi_grad_clip_ratio: float = 0.1,
    svi_adam_eps: float = 1e-6,
    svi_adam_update_clip: float = 3.0,
    svi_guide_init_scale: float = 0.01,
    svi_check_every: int = 1000,
    svi_patience: int = 0,
    svi_min_delta: float = 0.0,
    init_from_posterior_path: Optional[str] = None,
    init_seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """
    Construct posterior samples using SVI with AutoDiagonalNormal guide.

    Returns:
        Dictionary of posterior samples with keys matching return_sites
    """
    observation_data = _prepare_observation_data(
        observations=observations,
        L=L,
        K=K,
        require_y=True,
    )

    # Create variational guide and SVI
    guide = None
    if init_from_posterior_path is not None:
        try:
            init_params_single = _init_params_from_posterior_samples(
                posterior_npz_path=init_from_posterior_path,
                num_chains=1,
                gamma=gamma,
                seed=init_seed,
            )
            init_params_single = {k: v[0] for k, v in init_params_single.items()}
            guide = AutoDiagonalNormal(
                model,
                init_loc_fn=init_to_value(values=init_params_single),
                init_scale=svi_guide_init_scale,
            )
            print(f"Initializing SVI guide mean from a posterior sample at: {init_from_posterior_path}")
        except NoUsablePosteriorSamplesError as exc:
            print(f"{exc} Falling back to SVI's default guide init instead.")
    if guide is None:
        guide = AutoDiagonalNormal(model, init_scale=svi_guide_init_scale)

    warmup_steps = max(1, int(svi_warmup_frac * num_svi_steps))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=svi_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_svi_steps,
        end_value=svi_learning_rate * svi_lr_min_fraction,
    )
    svi = SVI(
        model=model,
        guide=guide,
        optim=numpyro.optim.optax_to_numpyro(
            optax.chain(
                optax.adaptive_grad_clip(svi_grad_clip_ratio, eps=1e-3),
                optax.scale_by_adam(eps=svi_adam_eps),
                optax.clip(svi_adam_update_clip),
                optax.scale_by_learning_rate(lr_schedule),
            )
        ),
        loss=Trace_ELBO(num_particles=svi_num_particles),
    )

    # Run SVI to optimize the guide parameters
    svi_result = _run_svi_until_plateau(
        svi,
        key,
        num_steps=num_svi_steps,
        check_every=svi_check_every,
        patience=svi_patience,
        min_delta=svi_min_delta,
        N=N,
        L=L,
        K=K,
        gamma=gamma,
        inverse_link_function=inverse_link_function,
        sigma=sigma,
        overlap_reg_strength=overlap_reg_strength,
        observation_data=observation_data,
    )

    # loss plot
    diagnostics_root, _ = os.path.splitext(diagnostics_csv_path)
    loss_plot_path = f"{diagnostics_root}_svi_loss.png"
    skip = len(svi_result.losses) // 4

    fig = plt.figure(figsize=(8, 4))
    plt.plot(svi_result.losses[skip:])
    plt.title(f"SVI Loss Curve (after {skip} steps)")
    plt.xlabel("SVI Step")
    plt.ylabel("ELBO Loss")
    plt.grid()
    fig.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SVI loss curve to {loss_plot_path}")

    losses_np = np.asarray(svi_result.losses)
    num_nonfinite = int(np.sum(~np.isfinite(losses_np)))
    tail = losses_np[-min(1000, len(losses_np)):]
    tail_finite = tail[np.isfinite(tail)]
    tail_mean_str = f"{tail_finite.mean():.4f}" if tail_finite.size > 0 else "all non-finite"
    print(
        f"Final SVI loss: last-step={losses_np[-1]:.4f}, last-1000-steps mean={tail_mean_str} "
        f"(after {len(losses_np)} steps, {num_nonfinite} non-finite steps skipped by stable_update)"
    )

    # Draw samples from the posterior approximation using Predictive
    predictive_fn = Predictive(
        model,
        guide=guide,
        params=svi_result.params,
        num_samples=num_samples,
        return_sites=None if return_sites is None else tuple(return_sites),
    )

    samples = predictive_fn(
        key,
        N=N,
        L=L,
        K=K,
        gamma=gamma,
        inverse_link_function=inverse_link_function,
        sigma=sigma,
        overlap_reg_strength=overlap_reg_strength,
        observation_data=observation_data,
    )

    # Filter samples to requested sites
    if return_sites is None:
        result_samples = samples
    else:
        missing = [k for k in return_sites if k not in samples]
        if missing:
            raise KeyError(f"Requested return_sites are missing from posterior samples: {missing}")
        result_samples = {k: samples[k] for k in return_sites}

    # Save to disk if path provided
    if path is not None:
        np.savez(path, **_pack_samples_for_disk(result_samples, K=K))  # type: ignore[arg-type]

    return result_samples


def _load_theta_arrays(path: str) -> dict:
    """
    Load theta sample arrays saved by predictive/MCMC.
    """
    raw = np.load(path, allow_pickle=True)

    # jnp.save(dict) yields a 0-d object array; np.savez yields an NpzFile.
    if isinstance(raw, np.lib.npyio.NpzFile):
        data = {k: raw[k] for k in raw.files}
    elif isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.shape == ():
            data = raw.item()
        elif raw.shape == (1,):
            data = raw[0].item() if isinstance(raw[0], np.ndarray) else raw[0]
        else:
            raise ValueError(f"Unsupported object array shape for theta samples: {raw.shape}")
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(
            f"Unsupported theta sample file format at {path}. "
            "Expected dict-like object with keys: log_w, log_v, mode."
        )

    required = ("log_w", "log_v", "mode")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in theta samples file {path}: {missing}")
    return data


class GPUThetaBatches:
    """
    Theta samples held entirely on device.
    """

    def __init__(
        self,
        log_w: torch.Tensor,
        log_v: torch.Tensor,
        mode: torch.Tensor,
    ):
        self.log_w = log_w
        self.log_v = log_v
        self.mode = mode

    def __len__(self) -> int:
        return int(self.log_w.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.log_w[idx], self.log_v[idx], self.mode[idx]

    def exclude(self, idx: int) -> "GPUThetaBatches":
        """Return a copy of this split with sample `idx` removed."""
        keep = torch.ones(len(self), dtype=torch.bool, device=self.log_w.device)
        keep[idx] = False
        return GPUThetaBatches(self.log_w[keep], self.log_v[keep], self.mode[keep])

    def _batches(
        self, batch_size: int, indices: torch.Tensor, drop_last: bool,
    ):
        n = indices.shape[0]
        num_batches = n // batch_size if drop_last else -(-n // batch_size)
        for i in range(num_batches):
            idx = indices[i * batch_size:(i + 1) * batch_size]
            yield self.log_w[idx], self.log_v[idx], self.mode[idx]

    def epoch_batches(self, batch_size: int):
        """Shuffled, drop-last batches for one training epoch."""
        perm = torch.randperm(len(self), device=self.log_w.device)
        return self._batches(batch_size, perm, drop_last=True)

    def eval_batches(self, batch_size: int):
        """Sequential batches covering every sample (no shuffling, no drop)."""
        indices = torch.arange(len(self), device=self.log_w.device)
        return self._batches(batch_size, indices, drop_last=False)


def load_theta_gpu_splits(
    path: str,
    test_size: int,
    device: str,
) -> Tuple[GPUThetaBatches, GPUThetaBatches]:
    """Load theta samples fully onto `device` and split into train/test:
    test is the last contiguous test_size samples."""
    data = _load_theta_arrays(path)
    log_w = torch.as_tensor(data["log_w"], dtype=torch.float32, device=device)
    log_v = torch.as_tensor(data["log_v"], dtype=torch.float32, device=device)
    mode = torch.as_tensor(data["mode"], dtype=torch.long, device=device)

    total_size = int(log_w.shape[0])
    assert test_size < total_size, f"test_size {test_size} must be smaller than total dataset size {total_size}"
    assert test_size > 0, f"test_size must be positive, got {test_size}"
    train_size = total_size - test_size

    train_batches = GPUThetaBatches(log_w[:train_size], log_v[:train_size], mode[:train_size])
    test_batches = GPUThetaBatches(log_w[train_size:], log_v[train_size:], mode[train_size:])
    return train_batches, test_batches


def visualize_graphical_model(
    path: str,
    N: int,
    L: int,
    K: int,
    gamma: float | dist.Distribution, 
    inverse_link_function, 
    sigma: float, 
    overlap_reg_strength: float = 0.0,
    observations: Optional[Sequence[ObservationBlock]] = None,
) -> str:
    """Visualize the graphical model"""

    _, ext = os.path.splitext(path)
    if not ext:
        raise ValueError(
            f"path must include an output extension supported by graphviz (for example '.png'): {path}"
        )

    observation_data = _prepare_observation_data(
        observations=observations,
        L=L,
        K=K,
        require_y=False,
    )

    try:
        numpyro.render_model(
            model=model,
            model_args=(N, L, K, gamma, inverse_link_function, sigma, overlap_reg_strength, observation_data),
            filename=path,
        )
    except Exception as e:
        print(f"Failed to render graphical model to {path}: {e}")

    return path
