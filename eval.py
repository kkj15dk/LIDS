import os
import pickle
import warnings
from typing import Any, Dict, Optional, Sequence, Union, List

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.distributions as dist
from torch.amp.autocast_mode import autocast

from matplotlib.ticker import MaxNLocator
class _MaxNLocatorWithOne(MaxNLocator):
    """as place a tick mark at 1"""

    def tick_values(self, vmin, vmax):
        return np.union1d(super().tick_values(vmin, vmax), [1.0])

from matplotlib import colors as mcolors
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt
import colorcet as cc
import seaborn as sns
import logomaker

from experiment import Experiment
from oracle import OraclePLM, Oracle
from model import ExperimentCategorical
from function import FitnessFunction
from MCMC import ObservationBlock

import umap
import matplotlib as mpl

sns.set_theme(style="white")
mpl.rcParams.update({
        "font.size": 16,  # Base font size fallback.
        "axes.titlesize": 16,  # Title text size.
        "axes.labelsize": 16,  # Axis label text size (x and y - kept equal).
        "legend.fontsize": 22,  # Legend label text size.
        "xtick.labelsize": 14,  # X tick label size.
        "ytick.labelsize": 14,  # Y tick label size.
})

# LaTeX text rendering for the paper figures. Only applied by the plotting scripts, so that
# training/simulation does not require a LaTeX installation.
LATEX_RC_PARAMS = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}

UMAP_CMAP = cc.glasbey_dark
UMAP_RESERVED_COLORS = sns.color_palette(UMAP_CMAP, n_colors=3)
UMAP_COLOR_X0 = "#999999"
UMAP_COLOR_F_SAMPLES = UMAP_RESERVED_COLORS[0]
UMAP_COLOR_POSTERIOR_VARIATIONAL = UMAP_RESERVED_COLORS[1]
UMAP_COLOR_POSTERIOR_MCMC = UMAP_RESERVED_COLORS[2]
UMAP_COLOR_POSTERIOR_MCMC_DIVERGENT = "#d62728"
UMAP_Y_LEGEND_CMAP = "YlOrRd"

UMAP_LEGEND_SIZE_Q = 4.0
UMAP_LEGEND_SIZE_POSTERIOR = 6.0
UMAP_LEGEND_SIZE_X0 = 6.0
UMAP_LEGEND_SIZE_X0_TRUE = 8.0


def _umap_q_palette(num_experiments: int) -> list:
    return sns.color_palette(UMAP_CMAP, n_colors=num_experiments + 3)[3:] # type: ignore


def _make_hamming_umap(n_neighbors: int, min_dist: float):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*gradient function is not yet implemented for .* distance metric; inverse_transform will be unavailable.*",
            category=UserWarning,
        )
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="hamming",
        )

def _suppress_umap_inverse_transform_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*gradient function is not yet implemented for .* distance metric; inverse_transform will be unavailable.*",
        category=UserWarning,
    )

def _step_output_dir(output_dir: str, step: int) -> str:
    step_dir = os.path.join(output_dir, f"step_{step:07d}")
    os.makedirs(step_dir, exist_ok=True)
    return step_dir


def _hide_umap_axis_ticks(ax) -> None:
    """
    Remove tick marks and numeric labels from a UMAP embedding axis
    """
    ax.set_xticks([])
    ax.set_yticks([])


def _legend_columns(total_labels: int, max_labels_per_column: int = 45) -> int:
    if total_labels <= 0:
        return 1
    return max(1, int(np.ceil(total_labels / float(max_labels_per_column))))


def _umap_math_q_label(exp_idx: int) -> str:
    return rf"$q_{{{exp_idx}}}$"


def _umap_math_q_with_y_label(exp_idx: int, y_value: float) -> str:
    return rf"$q_{{{exp_idx}}}\,(y={y_value:.3f})$"


def _set_uniform_legend_marker_size(legend) -> None:
    """Set legend marker sizes by label category using top-level constants."""
    if legend is None:
        return

    default_size = float(UMAP_LEGEND_SIZE_Q)

    labels = [text.get_text() for text in legend.get_texts()]
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])

    for handle, label in zip(handles, labels):
        label_l = label.lower()
        if label.startswith("$q_{"):
            target_size = UMAP_LEGEND_SIZE_Q
        elif "true" in label_l:
            target_size = UMAP_LEGEND_SIZE_X0_TRUE
        elif r"$x^\star$" in label or r"x^\star" in label:
            target_size = UMAP_LEGEND_SIZE_X0
        elif "posterior" in label_l:
            target_size = UMAP_LEGEND_SIZE_POSTERIOR
        elif label_l == "f" or " f " in label_l or "$f$" in label_l:
            target_size = UMAP_LEGEND_SIZE_POSTERIOR
        else:
            target_size = default_size

        if hasattr(handle, "get_sizes") and hasattr(handle, "set_sizes"):
            handle.set_sizes(np.full_like(np.asarray(handle.get_sizes(), dtype=float), float(target_size) ** 2, dtype=float))
        elif hasattr(handle, "set_markersize"):
            handle.set_markersize(float(target_size))

def _experiment_from_observation_block(block: ObservationBlock, device: str) -> Experiment:
    """Create a non-trainable Experiment view from one ObservationBlock for visualization."""
    s_total, l, k = map(int, block.log_tau.shape)
    e_total = int(block.log_pi.shape[0])
    log_pi_np = np.array(block.log_pi, dtype=np.float32, copy=True)
    log_tau_np = np.array(block.log_tau, dtype=np.float32, copy=True)
    exp = Experiment(
        vocab_size=k,
        sequence_length=l,
        num_experiments=e_total,
        num_syntheses=s_total,
        tau_warmup_steps=0,
        initialize_syntheses_per_experiment=False,
        freeze_experiment_weights=True,
        experiment_log_weights=torch.tensor(log_pi_np, dtype=torch.float32, device=device),
    ).to(device).requires_grad_(False)
    with torch.no_grad():
        exp._parameters_syntheses_unscaled.copy_(torch.tensor(log_tau_np, dtype=torch.float32, device=device))
    exp.eval()
    return exp

### Visualization
def visualize_tau_and_pi(
                   tau: torch.Tensor,
                   pi: torch.Tensor,
                   vocabulary: Sequence[str],
                   step: int,
                   num_visualized_syntheses: Optional[int] = None,
                   output_dir: str = "logs",
) -> list:
    """
    Create sequence logo plots of the current learned vocabulary distribution

    Args:
        tau: Tensor of shape [num_syntheses, sequence_length, vocab_size]
        pi: Tensor of shape [num_experiments, num_syntheses]
        vocabulary: list/tuple of vocabulary tokens (e.g., amino acids)
        step: current step number
        num_visualized_syntheses: only plot the first this-many syntheses and experiments
        output_dir: directory to save the logoplots

    Returns:
        List of paths to the saved figures
    """

    step_dir = _step_output_dir(output_dir, step)

    # Get the probabilities
    probs = tau.detach().cpu().numpy()[:num_visualized_syntheses]  # [num_visualized_syntheses, sequence_length, vocab_size]

    filepaths = []
    for i, probs_synthesis in enumerate(probs):
        # Create DataFrame for logomaker
        df = {}
        for j, token in enumerate(vocabulary):
            df[token] = probs_synthesis[:, j]
        
        df = pd.DataFrame(df)
        
        # Create the logo plot
        fig, ax = plt.subplots(figsize=(12, 4))
        
        try:
            logo = logomaker.Logo(df, ax=ax)
            ax.set_title(f"Synthesis {i} at Step {step} (tau)")
            ax.set_xlabel("Position in Sequence")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            
            filepath = os.path.join(step_dir, f"syn_{i}_probs_step_{step:07d}.png")
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            filepaths.append(filepath)
        except Exception as e:
            print(f"Error creating logoplot: {e}")
            plt.close()
    
    weights_experiments = pi.detach().cpu().numpy()[:num_visualized_syntheses]  # [num_visualized_syntheses, num_syntheses]
    for i, weights in enumerate(weights_experiments):
        fig, ax = plt.subplots(figsize=(8, 4))
        synthesis_indices = np.arange(len(weights), dtype=int)
        ax.bar(synthesis_indices, weights)
        ax.set_title(f"Weights for Experiment {i} at Step {step} (pi)")
        ax.set_xlabel("Synthesis Index")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.set_xticks(synthesis_indices)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        filepath = os.path.join(step_dir, f"exp_{i}_weights_step_{step:07d}.png")
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        filepaths.append(filepath)
    
    return filepaths


def visualize_umap_discrete(
        vocabulary: Sequence[str],
        observation_blocks: Sequence[ObservationBlock],
        step: int,
        x_0_samples: torch.Tensor,
        num_q_samples_per_experiment: int,
        n_neighbors: int,
        min_dist: float,
        recompute: bool,
        umap_together: bool,
        output_dir: str = "logs",
        umap_cache_dir: Optional[str] = None,
        s: int = 1,
        # simulated true parameters
        simulated_true_name: str = "simulated_true_function",
        x_0_true: Optional[torch.Tensor] = None,
        f_of_x_0_true: Optional[float] = None,
        f_samples: Optional[torch.Tensor] = None,
        map_f_samples: Optional[torch.Tensor] = None,
        posterior_samples_variational: Optional[torch.Tensor] = None,
        posterior_samples_mcmc: Optional[torch.Tensor] = None,
        posterior_samples_mcmc_divergent: Optional[torch.Tensor] = None,
        thompson_sampling: bool = False,
        deduplicate_samples_for_umap: bool = False,
        umap_color_divergences: bool = False,
) -> Optional[str]:
    """
    Build a UMAP embedding from discrete token-index sequences using Hamming distance:
    - x_0 samples (reserved color 0)
    - samples from each q_t (each experiment gets a distinct color)
    - posterior samples from r_phi(x_0|{y,q}_t) (reserved color 1, optional)
    - posterior samples from MCMC (reserved color 2, optional)
    - divergent MCMC samples (optional overlay)
    """

    step_dir = _step_output_dir(output_dir, step)
    simulated_true_dir = os.path.join(step_dir, simulated_true_name)
    os.makedirs(simulated_true_dir, exist_ok=True)

    cache_dir = output_dir if umap_cache_dir is None else umap_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    umap_reducer_path = os.path.join(cache_dir, "umap_step1_reducer.pkl")
    umap_reference_path = os.path.join(cache_dir, "umap_step1_reference.npz")

    if thompson_sampling:
        num_q_samples_per_experiment = 2 # Could be just one, but I'll do 2 for bug-catching purposes.
    q_samples_list = []
    y_e_true_list = []
    for observation_block in observation_blocks:
        exp = _experiment_from_observation_block(observation_block, device=str(x_0_samples.device))
        # Ensure y is a writeable numpy array before converting to torch
        y_np = np.array(observation_block.y, dtype=np.float32, copy=True)
        y = torch.as_tensor(y_np, dtype=torch.float32)
        assert y is not None, "Observation block is missing y."
        y_e_true_list.append(y)
        with torch.no_grad():
            q_samples = exp.sample((num_q_samples_per_experiment,))  # [N_q, E, L]
        q_samples_list.append(q_samples)
    
    q_samples = torch.cat(q_samples_list, dim=1)  # [N_q, sum(E), L]
    q_samples = q_samples.detach().cpu().long()
    y_e_true = torch.cat(y_e_true_list, dim=0)  # [sum(E)]
    num_experiments = y_e_true.shape[0]

    # Deduplicate x_0_samples and q_samples if requested
    x_0_features = x_0_samples.detach().cpu().long()
    if deduplicate_samples_for_umap:
        # Deduplicate x_0_samples
        x_0_features = torch.unique(x_0_features, dim=0)
        # Deduplicate q_samples for each experiment, keep as list of tensors
        q_samples_list = []
        for exp_idx in range(num_experiments):
            q_exp = q_samples[:, exp_idx, :]  # [N_q, L]
            q_exp_unique = torch.unique(q_exp, dim=0)
            q_samples_list.append(q_exp_unique)
        q_samples = q_samples_list  # List[Tensor], each [N_unique, L]
    else:
        # Not deduplicated: keep as [N_q, E, L], convert to list for unified handling
        q_samples = [q_samples[:, exp_idx, :] for exp_idx in range(num_experiments)]
    num_x_0_samples = x_0_features.shape[0]

    # If umap_together is True, don't use cached reference and fit on combined data. Deliberately
    # not gated on step > 1: a reference fit can be built from any round (e.g. a duplicate's last
    # round, to fit on its full accumulated history) and then reused even for round 1's own plot.
    use_cached_reference = not umap_together and os.path.exists(umap_reducer_path) and os.path.exists(umap_reference_path) and not recompute

    if use_cached_reference:
        with open(umap_reducer_path, "rb") as f:
            reducer = pickle.load(f)
        cached = np.load(umap_reference_path)
        x_0_embedding_x = cached["x_0_embedding_x"]
        x_0_embedding_y = cached["x_0_embedding_y"]
    else:
        if umap_together:
            # Fit UMAP on combined x_0 and all q samples as discrete token sequences.
            combined_features = torch.cat([x_0_features] + q_samples, dim=0)  # [N_x + sum(N_q), L]
            combined_feature_array = np.asarray(combined_features, dtype=np.uint16)
            reducer = _make_hamming_umap(n_neighbors=n_neighbors, min_dist=min_dist)
            with warnings.catch_warnings():
                _suppress_umap_inverse_transform_warning()
                combined_embedding = np.asarray(reducer.fit_transform(combined_feature_array))
            # Extract x_0 embedding from the combined embedding
            x_0_embedding = combined_embedding[:num_x_0_samples]
            x_0_embedding_x = x_0_embedding[:, 0]
            x_0_embedding_y = x_0_embedding[:, 1]
        else:
            reducer = _make_hamming_umap(n_neighbors=n_neighbors, min_dist=min_dist)
            x_0_feature_array = np.asarray(x_0_features, dtype=np.uint16)
            with warnings.catch_warnings():
                _suppress_umap_inverse_transform_warning()
                x_0_embedding = np.asarray(reducer.fit_transform(x_0_feature_array))
            x_0_embedding_x = x_0_embedding[:, 0]
            x_0_embedding_y = x_0_embedding[:, 1]
            with open(umap_reducer_path, "wb") as f:
                pickle.dump(reducer, f)
            np.savez(
                umap_reference_path,
                x_0_embedding_x=x_0_embedding_x,
                x_0_embedding_y=x_0_embedding_y,
            )

    def _to_2d_uint_array(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if x is None:
            return None
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D sequence tensor, got shape {arr.shape}")
        return np.asarray(arr, dtype=np.uint16)

    # print x_0 true to a txt file for reference using the vocabulary
    if x_0_true is not None:
        x_0_true_path = os.path.join(simulated_true_dir, f"x_0_true_step_{step:07d}.txt")
        x_0_tokens = [vocabulary[int(i)] for i in x_0_true.detach().cpu().numpy()]
        with open(x_0_true_path, "w") as f:
            f.write("".join(x_0_tokens))


    x_0_true_array = _to_2d_uint_array(x_0_true)
    f_samples_array = _to_2d_uint_array(f_samples)
    map_f_samples_array = _to_2d_uint_array(map_f_samples)
    posterior_var_array = _to_2d_uint_array(posterior_samples_variational)
    posterior_mcmc_array = _to_2d_uint_array(posterior_samples_mcmc)
    posterior_mcmc_divergent_array = _to_2d_uint_array(posterior_samples_mcmc_divergent)

    # Plotting settings
    q_alpha = 0.85
    q_size = 10 * s if thompson_sampling else s
    x_0_alpha = 0.55
    x_0_size = s
    posterior_alpha = 0.55
    posterior_size = 4 * s
    f_alpha = 0.55
    f_size = s


    y_e_true_array: Optional[np.ndarray] = None
    if y_e_true is not None:
        y_e_true_array = np.asarray(y_e_true.detach().cpu() if isinstance(y_e_true, torch.Tensor) else y_e_true, dtype=np.float32).reshape(-1)
        if y_e_true_array.shape[0] != num_experiments:
            raise ValueError(
                f"y_e_true has length {y_e_true_array.shape[0]}, expected {num_experiments}"
            )

    q_legend_text_colors: Dict[str, tuple] = {}
    if y_e_true_array is not None and y_e_true_array.size > 0:
        y_min = float(np.min(y_e_true_array))
        y_max = float(np.max(y_e_true_array))
        if np.isclose(y_min, y_max):
            y_max = y_min + 1.0
        y_norm = mcolors.Normalize(vmin=y_min, vmax=y_max)
        y_cmap = plt.get_cmap(UMAP_Y_LEGEND_CMAP)
        for exp_idx in range(num_experiments):
            q_label = _umap_math_q_with_y_label(exp_idx, float(y_e_true_array[exp_idx]))
            q_legend_text_colors[q_label] = y_cmap(y_norm(float(y_e_true_array[exp_idx])))

    q_embeddings: Dict[str, np.ndarray] = {}
    for exp_idx in range(num_experiments):
        exp_label = f"q_{exp_idx}"
        q_exp = q_samples[exp_idx]  # [N_q, L] (deduplicated or not)
        q_feature_array = np.asarray(q_exp, dtype=np.uint16)
        with warnings.catch_warnings():
            _suppress_umap_inverse_transform_warning()
            q_embeddings[exp_label] = np.asarray(reducer.transform(q_feature_array))

    x_0_true_embedding: Optional[np.ndarray] = None
    f_samples_embedding: Optional[np.ndarray] = None
    map_f_samples_embedding: Optional[np.ndarray] = None
    posterior_var_embedding: Optional[np.ndarray] = None
    posterior_mcmc_embedding: Optional[np.ndarray] = None
    posterior_mcmc_divergent_embedding: Optional[np.ndarray] = None
    with warnings.catch_warnings():
        _suppress_umap_inverse_transform_warning()
        if x_0_true_array is not None:
            x_0_true_embedding = np.asarray(reducer.transform(x_0_true_array))
        if f_samples_array is not None:
            f_samples_embedding = np.asarray(reducer.transform(f_samples_array))
        if map_f_samples_array is not None:
            map_f_samples_embedding = np.asarray(reducer.transform(map_f_samples_array))
        if posterior_var_array is not None:
            posterior_var_embedding = np.asarray(reducer.transform(posterior_var_array))
        if posterior_mcmc_array is not None:
            posterior_mcmc_embedding = np.asarray(reducer.transform(posterior_mcmc_array))
        if posterior_mcmc_divergent_array is not None:
            posterior_mcmc_divergent_embedding = np.asarray(
                reducer.transform(posterior_mcmc_divergent_array)
            )

    palette = _umap_q_palette(num_experiments)
    x0_label = r"$x^\star$"

    # Main plot: x_0 and all q_t on one axis
    ncol = _legend_columns(total_labels=num_experiments + 1, max_labels_per_column=45)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        x_0_embedding_x,
        x_0_embedding_y,
        s=x_0_size,
        color=UMAP_COLOR_X0,
        alpha=x_0_alpha,
        label=x0_label,
    )

    for exp_idx in range(num_experiments):
        exp_label = f"q_{exp_idx}"
        q_embedding = q_embeddings.get(exp_label)
        if q_embedding is not None and q_embedding.size > 0:
            ax.scatter(
                q_embedding[:, 0],
                q_embedding[:, 1],
                s=q_size,
                color=palette[exp_idx],
                alpha=q_alpha,
                label=_umap_math_q_label(exp_idx),
            )

    ax.set_title(rf"UMAP of $x^\star$ and $q_e$ samples at step {step}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    _hide_umap_axis_ticks(ax)
    ax.set_aspect("equal", adjustable="box")
    legend_main = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=ncol)
    _set_uniform_legend_marker_size(legend_main)

    shared_xlim = ax.get_xlim()
    shared_ylim = ax.get_ylim()

    fig.tight_layout()

    filepath = os.path.join(step_dir, f"umap_step_{step:07d}.png")
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Subplot 1: x_0 on the left, all q_t on the right
    figwidth_split = 14

    fig_all, axes_all = plt.subplots(1, 2, figsize=(figwidth_split, 7), sharex=True, sharey=True)

    axes_all[0].scatter(
        x_0_embedding_x,
        x_0_embedding_y,
        s=x_0_size,
        color=UMAP_COLOR_X0,
        alpha=x_0_alpha,
        label=x0_label,
    )
    axes_all[0].set_title(r"$\pi(x^\star)$")
    axes_all[0].set_xlabel("UMAP-1")
    axes_all[0].set_ylabel("UMAP-2")
    _hide_umap_axis_ticks(axes_all[0])
    axes_all[0].set_aspect("equal", adjustable="box")

    for exp_idx in range(num_experiments):
        exp_label = f"q_{exp_idx}"
        q_embedding = q_embeddings.get(exp_label)
        if q_embedding is not None and q_embedding.size > 0:
            axes_all[1].scatter(
                q_embedding[:, 0],
                q_embedding[:, 1],
                s=q_size,
                color=palette[exp_idx],
                alpha=q_alpha,
                label=_umap_math_q_label(exp_idx),
            )
    axes_all[1].set_title(r"All $q_e$")
    axes_all[1].set_xlabel("UMAP-1")
    axes_all[1].set_ylabel("UMAP-2")
    _hide_umap_axis_ticks(axes_all[1])
    axes_all[1].set_aspect("equal", adjustable="box")
    legend_all = axes_all[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=ncol)
    _set_uniform_legend_marker_size(legend_all)
    axes_all[0].set_xlim(shared_xlim)
    axes_all[0].set_ylim(shared_ylim)
    axes_all[1].set_xlim(shared_xlim)
    axes_all[1].set_ylim(shared_ylim)
    fig_all.suptitle(f"UMAP split view at step {step}")
    fig_all.tight_layout()

    filepath_all = os.path.join(step_dir, f"umap_split_all_q_step_{step:07d}.png")
    fig_all.savefig(filepath_all, dpi=130, bbox_inches="tight")
    plt.close(fig_all)

    # Subplot 2: x_0 next to each individual q_t
    for exp_idx in range(num_experiments):
        exp_label = f"q_{exp_idx}"
        q_embedding = q_embeddings.get(exp_label)
        if q_embedding is None or q_embedding.size == 0:
            continue

        fig_pair, axes_pair = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

        axes_pair[0].scatter(
            x_0_embedding_x,
            x_0_embedding_y,
            s=x_0_size,
            color=UMAP_COLOR_X0,
            alpha=x_0_alpha,
            label=x0_label,
        )
        axes_pair[0].set_title(r"$\pi(x^\star)$")
        axes_pair[0].set_xlabel("UMAP-1")
        axes_pair[0].set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(axes_pair[0])
        axes_pair[0].set_aspect("equal", adjustable="box")

        axes_pair[1].scatter(
            q_embedding[:, 0],
            q_embedding[:, 1],
            s=q_size,
            color=palette[exp_idx],
            alpha=q_alpha,
            label=_umap_math_q_label(exp_idx),
        )
        axes_pair[1].set_title(_umap_math_q_label(exp_idx))
        axes_pair[1].set_xlabel("UMAP-1")
        axes_pair[1].set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(axes_pair[1])
        axes_pair[1].set_aspect("equal", adjustable="box")
        axes_pair[0].set_xlim(shared_xlim)
        axes_pair[0].set_ylim(shared_ylim)
        axes_pair[1].set_xlim(shared_xlim)
        axes_pair[1].set_ylim(shared_ylim)

        fig_pair.suptitle(f"UMAP split view x_0 vs {exp_label} at step {step}")
        fig_pair.tight_layout()

        filepath_pair = os.path.join(step_dir, f"umap_split_{exp_label}_step_{step:07d}.png")
        fig_pair.savefig(filepath_pair, dpi=130, bbox_inches="tight")
        plt.close(fig_pair)

    # Optional conditioned plot: x_0_true + q_e + variational posterior + MCMC posterior.
    if (
        x_0_true_embedding is not None
        and f_samples_embedding is not None
        and posterior_mcmc_embedding is not None
        and y_e_true_array is not None
        and f_of_x_0_true is not None
    ):
        fig_cond, ax_cond = plt.subplots(figsize=(figwidth_split, 8))

        ax_cond.scatter(
            x_0_embedding_x,
            x_0_embedding_y,
            s=max(4, 2 * s),
            color=UMAP_COLOR_X0,
            alpha=0.18,
            label=r"$\pi(x^\star)$",
            zorder=0,
        )

        x_true_label = rf"$x^\star$ ($\mathrm{{f}}(x^\star)={f_of_x_0_true:.3f}$)"
        ax_cond.scatter(
            x_0_true_embedding[:, 0],
            x_0_true_embedding[:, 1],
            s=max(24, 20 * s),
            c="black",
            marker="*",
            alpha=0.95,
            label=x_true_label,
            zorder=4,
        )
        if posterior_var_embedding is not None:
            ax_cond.scatter(
                posterior_var_embedding[:, 0],
                posterior_var_embedding[:, 1],
                s=posterior_size,
                color=UMAP_COLOR_POSTERIOR_VARIATIONAL,
                alpha=posterior_alpha,
                label="variational posterior",
                zorder=1,
            )
        ax_cond.scatter(
            posterior_mcmc_embedding[:, 0],
            posterior_mcmc_embedding[:, 1],
            s=posterior_size,
            color=UMAP_COLOR_POSTERIOR_MCMC,
            alpha=posterior_alpha,
            label="MCMC posterior",
            zorder=1,
        )
        if umap_color_divergences and posterior_mcmc_divergent_embedding is not None:
            ax_cond.scatter(
                posterior_mcmc_divergent_embedding[:, 0],
                posterior_mcmc_divergent_embedding[:, 1],
                s=posterior_size,
                color=UMAP_COLOR_POSTERIOR_MCMC_DIVERGENT,
                alpha=min(1.0, posterior_alpha + 0.1),
                label="MCMC posterior (divergent)",
                zorder=3,
            )

        for exp_idx in range(num_experiments):
            exp_label = f"q_{exp_idx}"
            q_embedding = q_embeddings.get(exp_label)
            if q_embedding is None or q_embedding.size == 0:
                continue
            ax_cond.scatter(
                q_embedding[:, 0],
                q_embedding[:, 1],
                s=q_size,
                color=palette[exp_idx],
                alpha=q_alpha,
                label=_umap_math_q_with_y_label(exp_idx, float(y_e_true_array[exp_idx])),
                zorder=2,
            )

        ax_cond.set_title(f"UMAP at round {step}")
        ax_cond.set_xlabel("UMAP-1")
        ax_cond.set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(ax_cond)
        ax_cond.set_aspect("equal", adjustable="box")
        legend_cond = ax_cond.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=ncol)
        _set_uniform_legend_marker_size(legend_cond)
        for text in legend_cond.get_texts():
            text_color = q_legend_text_colors.get(text.get_text())
            if text_color is not None:
                text.set_color(text_color)
        ax_cond.set_xlim(shared_xlim)
        ax_cond.set_ylim(shared_ylim)
        fig_cond.tight_layout()

        filepath_cond = os.path.join(simulated_true_dir, f"umap_conditioned_step_{step:07d}.png")
        fig_cond.savefig(filepath_cond, dpi=130, bbox_inches="tight")
        plt.close(fig_cond)

    # Optional conditioned split view: separate panes for q, variational, mcmc, and x_0.
    if (
        x_0_true_embedding is not None
        and f_samples_embedding is not None
        and posterior_mcmc_embedding is not None
        and y_e_true_array is not None
        and f_of_x_0_true is not None
    ):
        fig_split, axes_split = plt.subplots(2, 2, figsize=(figwidth_split, 12), sharex=True, sharey=True)
        ax_q, ax_var, ax_mcmc, ax_x0 = axes_split.flat

        # q samples
        for exp_idx in range(num_experiments):
            exp_label = f"q_{exp_idx}"
            q_embedding = q_embeddings.get(exp_label)
            if q_embedding is None or q_embedding.size == 0:
                continue
            ax_q.scatter(
                q_embedding[:, 0],
                q_embedding[:, 1],
                s=q_size,
                color=palette[exp_idx],
                alpha=q_alpha,
                label=_umap_math_q_with_y_label(exp_idx, float(y_e_true_array[exp_idx])),
                zorder=2,
            )
        # x_0 prior
        ax_q.scatter(
            x_0_embedding_x,
            x_0_embedding_y,
            s=max(4, 2 * s),
            color=UMAP_COLOR_X0,
            alpha=0.18,
            label=r"$\pi(x^\star)$",
            zorder=0,
        )
        ax_q.scatter(
            x_0_true_embedding[:, 0],
            x_0_true_embedding[:, 1],
            s=max(24, 20 * s),
            c="black",
            marker="*",
            alpha=0.95,
            label=rf"$x^\star$ ($\mathrm{{f}}(x^\star)={f_of_x_0_true:.3f}$)",
            zorder=4,
        )
        ax_q.set_title(r"$q$ & $\pi(x^\star)$")
        ax_q.set_xlabel("UMAP-1")
        ax_q.set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(ax_q)

        # variational posterior samples (optional).
        if posterior_var_embedding is not None:
            ax_var.scatter(
                posterior_var_embedding[:, 0],
                posterior_var_embedding[:, 1],
                s=posterior_size,
                color=UMAP_COLOR_POSTERIOR_VARIATIONAL,
                alpha=posterior_alpha,
                label="variational posterior",
                zorder=1,
            )
            ax_var.scatter(
                x_0_true_embedding[:, 0],
                x_0_true_embedding[:, 1],
                s=max(24, 20 * s),
                c="black",
                marker="*",
                alpha=0.95,
                label=x_true_label,
                zorder=4,
            )
            ax_var.set_title(r"variational posterior $r_\phi(x^\star \mid \mathcal{D})$")
        else:
            ax_var.set_title("Variational posterior unavailable")
        ax_var.set_xlabel("UMAP-1")
        ax_var.set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(ax_var)

        # mcmc posterior samples
        ax_mcmc.scatter(
            posterior_mcmc_embedding[:, 0],
            posterior_mcmc_embedding[:, 1],
            s=posterior_size,
            color=UMAP_COLOR_POSTERIOR_MCMC,
            alpha=posterior_alpha,
            label="MCMC posterior",
            zorder=1,
        )
        if umap_color_divergences and posterior_mcmc_divergent_embedding is not None:
            ax_mcmc.scatter(
                posterior_mcmc_divergent_embedding[:, 0],
                posterior_mcmc_divergent_embedding[:, 1],
                s=posterior_size,
                color=UMAP_COLOR_POSTERIOR_MCMC_DIVERGENT,
                alpha=min(1.0, posterior_alpha + 0.1),
                label="MCMC posterior (divergent)",
                zorder=3,
            )
        ax_mcmc.scatter(
            x_0_true_embedding[:, 0],
            x_0_true_embedding[:, 1],
            s=max(24, 20 * s),
            c="black",
            marker="*",
            alpha=0.95,
            label=x_true_label,
            zorder=4,
        )
        ax_mcmc.set_title(r"MCMC posterior $\mathrm{p}(x^\star \mid \mathcal{D})$")
        ax_mcmc.set_xlabel("UMAP-1")
        ax_mcmc.set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(ax_mcmc)

        # f samples
        ax_x0.scatter(
            f_samples_embedding[:, 0],
            f_samples_embedding[:, 1],
            s=f_size,
            color=UMAP_COLOR_F_SAMPLES,
            alpha=f_alpha,
            label="f",
            zorder=1,
        )
        ax_x0.scatter(
            x_0_true_embedding[:, 0],
            x_0_true_embedding[:, 1],
            s=max(24, 20 * s),
            c="black",
            marker="*",
            alpha=0.95,
            label=x_true_label,
            zorder=4,
        )
        ax_x0.set_title(r"$f$ & $x^\star$")
        ax_x0.set_xlabel("UMAP-1")
        ax_x0.set_ylabel("UMAP-2")
        _hide_umap_axis_ticks(ax_x0)

        for ax in axes_split.flat:
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)
            ax.set_aspect("auto")

        # Keep one consolidated legend outside to avoid clutter in each panel.
        handles, labels = [], []
        for ax in axes_split.flat:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
        legend_split = fig_split.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=ncol,
        )
        _set_uniform_legend_marker_size(legend_split)
        for text in legend_split.get_texts():
            text_color = q_legend_text_colors.get(text.get_text())
            if text_color is not None:
                text.set_color(text_color)

        fig_split.suptitle(f"UMAP at round {step}")
        fig_split.tight_layout()
        filepath_split = os.path.join(simulated_true_dir, f"umap_conditioned_split_step_{step:07d}.png")
        fig_split.savefig(filepath_split, dpi=130, bbox_inches="tight")
        plt.close(fig_split)

    # Optional 2-panel conditioned view: true f samples vs MAP f samples.
    if (
        x_0_true_embedding is not None
        and f_samples_embedding is not None
        and map_f_samples_embedding is not None
    ):
        fig_f_pair, axes_f_pair = plt.subplots(1, 2, figsize=(figwidth_split, 6), sharex=True, sharey=True)
        for ax, emb, color, title in (
            (axes_f_pair[0], f_samples_embedding, UMAP_COLOR_F_SAMPLES, r"True $f$"),
            (axes_f_pair[1], map_f_samples_embedding, UMAP_COLOR_POSTERIOR_MCMC, r"MAP $f$"),
        ):
            ax.scatter(
                emb[:, 0],
                emb[:, 1],
                s=f_size,
                color=color,
                alpha=f_alpha,
                label=title,
                zorder=1,
            )
            ax.scatter(
                x_0_true_embedding[:, 0],
                x_0_true_embedding[:, 1],
                s=max(24, 20 * s),
                c="black",
                marker="*",
                alpha=0.95,
                label=r"$x^\star$",
                zorder=4,
            )
            ax.set_title(title)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            _hide_umap_axis_ticks(ax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)

        fig_f_pair.suptitle(f"UMAP true vs MAP function samples at step {step}")
        fig_f_pair.tight_layout()
        filepath_f_pair = os.path.join(simulated_true_dir, f"umap_conditioned_true_vs_map_step_{step:07d}.png")
        fig_f_pair.savefig(filepath_f_pair, dpi=130, bbox_inches="tight")
        plt.close(fig_f_pair)

    return filepath


def visualize_function(step: int, 
                       vocabulary: Sequence[str],
                       log_w: torch.Tensor, 
                       log_v: torch.Tensor, 
                       path: str,
                       simulated_true_name: str = "simulated_true_function",
) -> list:
    '''
    Visualize a single sampled function by plotting:
    - mixture weights w as a bar chart
    - component token probabilities v as sequence logos (one per component)

    Args:
        log_w: [num_components]
        log_v: [num_components, sequence_length, vocab_size]
        path: output path prefix (directory and basename stem)

    Returns:
        list of filepaths that were successfully written.
    '''
    K = log_v.shape[-1]

    step_dir = _step_output_dir(path, step)
    out_dir = os.path.join(step_dir, simulated_true_name)
    os.makedirs(out_dir, exist_ok=True)

    if log_w.dim() != 1:
        raise ValueError(f"Expected log_w to be 1D [num_components], got shape {tuple(log_w.shape)}")
    if log_v.dim() != 3:
        raise ValueError(
            f"Expected log_v to be 3D [num_components, sequence_length, vocab_size], got shape {tuple(log_v.shape)}"
        )

    log_w_cpu = log_w.detach().cpu().to(torch.float32)
    log_v_cpu = log_v.detach().cpu().to(torch.float32)

    w = torch.exp(log_w_cpu).numpy()
    v = torch.exp(log_v_cpu).numpy()

    filepaths = []

    # Plot mixture weights w.
    fig, ax = plt.subplots(figsize=(8, 4))
    component_idx = np.arange(w.shape[0], dtype=int)
    ax.bar(component_idx, w)
    ax.set_title("Mixture weights w")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Weight")
    # ax.set_ylim(0,)
    ax.set_xticks(component_idx)
    w_path = f"{out_dir}/weights_w.png"
    fig.savefig(w_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    filepaths.append(w_path)

    # Plot each component distribution v as a sequence logo.
    token_names = [str(token) for token in vocabulary]

    for n in range(v.shape[0]):
        df = pd.DataFrame(v[n], columns=token_names)  # [sequence_length, vocab_size]
        fig, ax = plt.subplots(figsize=(12, 3.5))
        try:
            logomaker.Logo(df, ax=ax)
            ax.set_title(f"Component {n} token weights v")
            ax.set_xlabel("Position in sequence")
            ax.set_ylabel("weight")
            ax.set_ylim(0, K)
            v_path = f"{out_dir}/component_{n:02d}_v.png"
            fig.savefig(v_path, dpi=120, bbox_inches='tight')
            filepaths.append(v_path)
        finally:
            plt.close(fig)

    return filepaths


def compute_top_k_true_metrics(
    variational_dist: ExperimentCategorical,
    y_e: torch.Tensor,
    oracle: Oracle,
    k_for_top_k: int,
    beam_width_for_top_k: int,
    beam_chunk_size: int,
    device: str,
    amp_dtype: torch.dtype,
) -> dict[str, float | list[float]]:

    with torch.inference_mode():
        with autocast(device_type=device, dtype=amp_dtype):
            top_k_samples = variational_dist.get_topk(
                y_e=y_e.unsqueeze(0),
                k=k_for_top_k,
                beam_width=beam_width_for_top_k,
                chunk_size=beam_chunk_size,
            ).squeeze(0)
        
        # ORACLE
        x_0_true = oracle.x_0
        f_of_x0_true = oracle.f(x_0_true.unsqueeze(0)).squeeze().item()
        f_top_k = oracle.f(top_k_samples).squeeze()
        
        ratios = f_top_k / f_of_x0_true
        return {
            "true_value": float(f_of_x0_true),
            "top-k_value": float(f_top_k.max().item()),
            "top_k_ratio": float(torch.max(ratios).item()),
            "top-1_value": float(f_top_k[0].item()),
            "top-1_ratio": float(ratios[0].item()),
            "all_ratios": [float(v) for v in ratios.detach().cpu().tolist()],
        }


def compute_map_metrics_from_posterior(
    fitness_function: FitnessFunction,
    oracle: Oracle,
    posterior_log_w: torch.Tensor,
    posterior_log_v: torch.Tensor,
    posterior_mode: torch.Tensor,
) -> Dict[str, Any]:
    """
    Estimate the mode of the posterior over the best design x* = argmax_x f_theta(x), using posterior samples of theta.
    Ties in the vote count are broken using each tied candidate's mean achieved fitness value among the samples.

    Arguments:
        posterior_log_w: [M, num_components]
        posterior_log_v: [M, num_components, sequence_length, vocab_size]
        posterior_mode: [M, num_components, sequence_length]
    """
    x0_samples = fitness_function.argmax(
        log_w=posterior_log_w, log_v=posterior_log_v, candidates=posterior_mode
    )  # [M, L]
    f_values = fitness_function.f(
        x=x0_samples.unsqueeze(1), log_w=posterior_log_w, log_v=posterior_log_v
    ).squeeze(1)  # [M], each sample's own achieved fitness at its own argmax

    unique_rows, inverse_idx, counts = torch.unique(
        x0_samples, dim=0, return_inverse=True, return_counts=True
    )
    max_count = int(counts.max().item())
    tied = (counts == max_count).nonzero(as_tuple=True)[0]
    if tied.numel() == 1:
        winner_idx = int(tied.item())
    else:
        tie_scores = torch.stack([f_values[inverse_idx == cand].mean() for cand in tied])
        winner_idx = int(tied[tie_scores.argmax()].item())
    x_0_map = unique_rows[winner_idx]  # [L]

    x_0_true = oracle.x_0
    y_true = oracle.f(x_0_true)
    y_map = oracle.f(x_0_map)
    ratio = y_map / y_true

    return {
        "MAP_ratio": float(ratio),
        "true_value": float(y_true),
        "map_value": float(y_map),
        "mode_count": max_count,
        "mode_num_ties": int(tied.numel()),
        "num_posterior_samples": int(posterior_log_w.shape[0]),
        "x_0_map": x_0_map.detach().cpu().numpy(),
    }


def top_k_fitness(
        step: int,
        top_k_fitness_values: torch.Tensor,
        output_dir: str = "logs",
    ) -> str:
    """
    Plot histograms of the top-k fitness fraction y/y_max.
    Args:
     - step: current step number (for filename)
     - top_k_fitness_values: [B_theta, k] tensor of the top-k fitness fraction values for each theta (in the interval [0, 1])
     - output_dir: directory to save the histogram image
    Returns:
     - histogram_path: path to the saved histogram image
    """
    B_theta, k = top_k_fitness_values.shape

    # make a histogram of the top_k_fitness values across the B_theta samples
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    values = top_k_fitness_values.flatten().cpu().numpy()  # [k * B_theta] # For the first histogram
    max_values = top_k_fitness_values.max(dim=-1).values.cpu().numpy()  # [B_theta] # For the second histogram
    
    counts_values, _, _ = axes[0].hist(values, bins=30, range=(0.0, 1.0), alpha=0.7, edgecolor='black')
    axes[0].set_title(f"Actual fitness of the top-{k} samples")
    axes[0].set_xlabel("Fitness value")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, float(counts_values.sum()) * 1.1 if counts_values.size > 0 else 1.0)
    axes[0].grid(False)

    counts_max, _, _ = axes[1].hist(max_values, bins=30, range=(0.0, 1.0), alpha=0.7, edgecolor='black')
    axes[1].set_title(f"Max fitness among top-{k} per theta")
    axes[1].set_xlabel("Fitness value")
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, float(counts_max.sum()) * 1.1 if counts_max.size > 0 else 1.0)
    axes[1].grid(False)

    fig.tight_layout()
    histogram_path = os.path.join(output_dir, f"step_{step:07d}_top_{k}_fitness_histogram.png")
    fig.savefig(histogram_path, dpi=130)
    plt.close(fig)

    return histogram_path # [k, B_theta], path


def _parse_map_metrics_file(path: str) -> tuple[float, float]:
    map_ratio: Optional[float] = None
    map_value: Optional[float] = None

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("MAP_ratio:"):
                map_ratio = float(line.split(":", 1)[1].strip())
            elif line.startswith("map_value:"):
                map_value = float(line.split(":", 1)[1].strip())

    if map_ratio is None:
        raise ValueError(f"Missing MAP_ratio in {path}")
    if map_value is None:
        raise ValueError(f"Missing map_value in {path}")
    return map_ratio, map_value


def _parse_entropy_file(path: str) -> float:
    mean_entropy: Optional[float] = None

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("mean_positional_entropy:"):
                mean_entropy = float(line.split(":", 1)[1].strip())
                break

    if mean_entropy is None:
        raise ValueError(f"Missing mean_positional_entropy in {path}")
    return mean_entropy


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """
    Sliding-window max: out[i] = max(values[max(0, i-window+1) : i+1]).
    """
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.max(values[start:i + 1])
    return out


def visualize_map_over_rounds(
        folder_names: Sequence[str],
        labels: Sequence[str],
        outputs_root: str = "outputs",
        output_filename: str = "map_over_rounds.png",
        plot_entropy: bool = False,
        average: bool = False,
        average_duplicates: Optional[int] = None,
        topk: Optional[int] = None,
        max_round: Optional[int] = None,
        mean_ci_max: Optional[float] = None,
        show_ci: bool = False,
        f_max: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        label_styles: Optional[Dict[str, Dict[str, str | tuple[float, float, float]]]] = None,
        legend_label_map: Optional[Dict[str, str]] = None,
        legend_loc: str = "center left",
        legend_bbox_to_anchor: Optional[tuple[float, float]] = (1.02, 0.5),
        legend_ncol: Optional[int] = None,
        legend_deduplicate: bool = False,
        annotate_label: Optional[str] = None,
        annotate_text: str = "Ours",
        annotate_round_frac: float = 0.65,
        annotate_offset: tuple[float, float] = (30.0, 30.0),
        annotate_tip_offset: tuple[float, float] = (0.0, 0.0),
        annotate_arrow_scale: float = 1.0,
        label_size: float = 26,
        title_pad: Optional[float] = None,
) -> str:
    """
    Plot MAP ratio over rounds for each duplicate or the average per folder.

    topk: if given, max over that round and the (topk - 1) rounds before it.

    max_round: if given, rounds beyond it are dropped.

    mean_ci_max: The true maximum of map_value.

    show_ci: if True, shade the mean's 95% CI band.

    f_max: if given, draws a black dotted horizontal reference line at this value.

    annotate_label: if given, draws a small arrow at that label's curve

    label_size: font size of the title and the x/y-axis labels.

    title_pad: gap in points between the title and the top of the axes (None = matplotlib default).

    title: if given, sets the plot's title, overriding the default title.

    xlabel: if given, overrides the default x-axis label ("experimental round").

    ylabel: if given, overrides the default y-axis label.
    """
    if len(folder_names) == 0:
        raise ValueError("folder_names must contain at least one folder")
    if len(folder_names) != len(labels):
        raise ValueError("folder_names and labels must have the same length")
    if average_duplicates is not None and average_duplicates < 1:
        raise ValueError(f"average_duplicates must be >= 1, got {average_duplicates}")
    if topk is not None and topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if show_ci and not average:
        raise ValueError("show_ci=True requires average=True")
    if average and mean_ci_max is None:
        raise ValueError("mean_ci_max is required when average=True (needed for the mean line itself, not just show_ci)")
    if mean_ci_max is not None and mean_ci_max <= 0.0:
        raise ValueError(f"mean_ci_max must be positive, got {mean_ci_max}")

    series: list[tuple[str, str, list[int], list[float], Optional[list[float]]]] = []
    for folder_name, label in zip(folder_names, labels):
        base_dir = os.path.join(outputs_root, folder_name)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Folder not found: {base_dir}")

        duplicate_dirs = sorted(
            [d for d in os.listdir(base_dir) if d.startswith("duplicate_") and os.path.isdir(os.path.join(base_dir, d))]
        )
        if len(duplicate_dirs) == 0:
            raise FileNotFoundError(f"No duplicate_* folders found in {base_dir}")

        for duplicate_name in duplicate_dirs:
            dup_dir = os.path.join(base_dir, duplicate_name)
            round_dirs = sorted(
                [
                    d for d in os.listdir(dup_dir)
                    if d.startswith("round_") and d.endswith("_obs") and os.path.isdir(os.path.join(dup_dir, d))
                ],
                key=lambda d: int(d.split("_")[1]),
            )

            rounds: list[int] = []
            map_values: list[float] = []
            map_ratios: list[float] = []
            entropy_values: Optional[list[float]] = [] if plot_entropy else None
            for round_dir in round_dirs:
                round_idx = int(round_dir.split("_")[1])
                if max_round is not None and round_idx > max_round:
                    break  # round_dirs is sorted ascending, so every later one exceeds it too
                metrics_path = os.path.join(dup_dir, round_dir, "map_metrics.txt")
                if not os.path.exists(metrics_path):
                    warnings.warn(f"Missing map_metrics.txt; skipping round {round_idx} for {metrics_path}", RuntimeWarning)
                    continue
                map_ratio, map_value = _parse_map_metrics_file(metrics_path)
                rounds.append(round_idx)
                # map_values.append(map_ratio) # to normalize to 0-1
                map_values.append(map_value)
                if plot_entropy:
                    entropy_path = os.path.join(dup_dir, round_dir, "posterior_x0_entropy.txt")
                    if not os.path.exists(entropy_path):
                        warnings.warn(
                            f"Missing posterior_x0_entropy.txt; skipping entropy for {entropy_path}",
                            RuntimeWarning,
                        )
                        if entropy_values is not None:
                            entropy_values.append(float("nan"))
                    else:
                        try:
                            entropy_value = _parse_entropy_file(entropy_path)
                        except ValueError as exc:
                            warnings.warn(
                                f"Failed to parse entropy from {entropy_path}: {exc}",
                                RuntimeWarning,
                            )
                            entropy_value = float("nan")
                        if entropy_values is not None:
                            entropy_values.append(entropy_value)

            if len(rounds) == 0:
                warnings.warn(f"No valid rounds with map_metrics.txt found under {dup_dir}; skipping this duplicate", RuntimeWarning)
                continue

            if topk is not None:
                map_values = _rolling_max(np.asarray(map_values, dtype=np.float64), topk).tolist()

            series.append((label, duplicate_name, rounds, map_values, entropy_values))

        if len([s for s in series if s[0] == label]) == 0:
            raise FileNotFoundError(f"No map_metrics.txt files found under {base_dir}/duplicate_*/round_*_obs")

    label_styles = {} if label_styles is None else label_styles
    legend_label_map = {} if legend_label_map is None else legend_label_map

    def _style_for(
        label: str,
        fallback_label: Optional[str] = None,
    ) -> Dict[str, str | tuple[float, float, float]]:
        style = label_styles.get(label)
        if style is not None:
            return style
        if fallback_label is not None:
            fallback = label_styles.get(fallback_label)
            if fallback is not None:
                return fallback
        return {}

    def _legend_label(label: str) -> str:
        return legend_label_map.get(label, label)

    base_colors = sns.color_palette("deep", n_colors=max(len(folder_names), 1))
    color_map: Dict[str, str | tuple[float, float, float]] = {}
    for idx, label in enumerate(labels):
        style = _style_for(label)
        color_map[label] = style.get("color", base_colors[idx])

    fig, ax = plt.subplots(figsize=(10, 6))

    def _forward_fill_to_rounds(
        rounds_in: list[int],
        values_in: list[float],
        target_rounds: list[int],
    ) -> np.ndarray:
        out = np.empty(len(target_rounds), dtype=np.float64)
        last_value = float("nan")
        idx = 0
        n = len(rounds_in)
        for i, r in enumerate(target_rounds):
            while idx < n and rounds_in[idx] <= r:
                last_value = float(values_in[idx])
                idx += 1
            out[i] = last_value
        return out

    def _logit_mean_ci_band(
        stack: np.ndarray, M: float, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-round mean and ~95% CI for the mean. The mean itself is the plain arithmetic mean across duplicates. 
        The CI is constructed via the delta method and a logistic transformation. Uses a t-critical value.

        Arguments:
            stack: [num_duplicates, num_rounds] array of map_values
            M: the true maximum of map_value.
            confidence: the confidence level for the CI.

        Returns (mean, lower, upper).
        """
        stack = np.asarray(stack, dtype=np.float64)
        n = np.sum(~np.isnan(stack), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_hat = np.nanmean(stack, axis=0)
            se_hat = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)

            eps = 1e-6 * M
            mean_clipped = np.clip(mean_hat, eps, M - eps)
            logit_hat = np.log(mean_clipped / (M - mean_clipped))
            se_logit = se_hat * M / (mean_clipped * (M - mean_clipped))

            t_crit = stats.t.ppf(0.5 + confidence / 2.0, df=np.maximum(n - 1, 1))
            half_width = t_crit * se_logit

            lower = (1.0 / (1.0 + np.exp(-(logit_hat - half_width)))) * M
            upper = (1.0 / (1.0 + np.exp(-(logit_hat + half_width)))) * M

            degenerate_point = (se_hat == 0.0) & (n >= 2)
            lower = np.where(degenerate_point, mean_hat, lower)
            upper = np.where(degenerate_point, mean_hat, upper)
            lower = np.where(n < 2, np.nan, lower)
            upper = np.where(n < 2, np.nan, upper)

        return np.clip(mean_hat, 0.0, M), lower, upper

    annotate_target: Optional[tuple[float, float]] = None
    for label_idx, label in enumerate(labels):
        label_series = [s for s in series if s[0] == label]
        if len(label_series) == 0:
            continue
        if average and average_duplicates is not None:
            label_series = label_series[:average_duplicates]
            if len(label_series) == 0:
                continue
        base_color = color_map[label]
        base_style = _style_for(label)
        base_linestyle = base_style.get("linestyle", "-")
        if average:
            all_rounds = sorted({r for _, _, rounds, _, _ in label_series for r in rounds})
            if len(all_rounds) == 0:
                continue
            map_stack = []
            entropy_stack = []
            for _, _, rounds, map_values, entropy_values in label_series:
                map_stack.append(_forward_fill_to_rounds(rounds, map_values, all_rounds))
                if plot_entropy and entropy_values is not None:
                    entropy_stack.append(_forward_fill_to_rounds(rounds, entropy_values, all_rounds))
            assert mean_ci_max is not None
            mean_p, mean_ci_lower, mean_ci_upper = _logit_mean_ci_band(np.vstack(map_stack), M=mean_ci_max)
            map_avg = mean_p
            ax.plot(
                all_rounds,
                map_avg,
                color=base_style.get("color", base_color),
                linewidth=2.6,
                linestyle=base_linestyle,
                label=_legend_label(label),
            )
            if annotate_label is not None and _legend_label(label) == annotate_label:
                idx = int(round(annotate_round_frac * (len(all_rounds) - 1)))
                annotate_target = (float(all_rounds[idx]), float(map_avg[idx]))
            if show_ci:
                ax.fill_between(
                    all_rounds,
                    mean_ci_lower,
                    mean_ci_upper,
                    color=base_style.get("color", base_color),
                    alpha=0.15,
                    linewidth=0,
                    label="_nolegend_",
                )
            if plot_entropy and len(entropy_stack) > 0:
                entropy_avg = np.nanmean(np.vstack(entropy_stack), axis=0)
                entropy_label = f"{label} entropy"
                entropy_style = _style_for(entropy_label, label)
                ax.plot(
                    all_rounds,
                    entropy_avg,
                    color=entropy_style.get("color", base_color),
                    linewidth=2.0,
                    linestyle=entropy_style.get("linestyle", "dotted"),
                    label=_legend_label(entropy_label),
                )
        else:
            dup_colors = sns.light_palette(base_color, n_colors=len(label_series) + 2)[2:]
            offsets = np.linspace(-0.1, 0.1, len(label_series)) if len(label_series) > 1 else np.array([0.0])
            for offset, color, (_, duplicate_name, rounds, map_values, entropy_values) in zip(offsets, dup_colors, label_series):
                dup_id = int(duplicate_name.split("_", 1)[1]) if "_" in duplicate_name else duplicate_name
                dup_label = f"{label} {dup_id}"
                rounds_plot = np.asarray(rounds, dtype=np.float32) + float(offset)
                dup_style = _style_for(dup_label, label)
                dup_color = dup_style.get("color", color)
                dup_linestyle = dup_style.get("linestyle", base_linestyle)
                ax.plot(
                    rounds_plot,
                    map_values,
                    color=dup_color,
                    linewidth=2.0,
                    linestyle=dup_linestyle,
                    label=_legend_label(dup_label),
                )
                if annotate_label is not None and _legend_label(label) == annotate_label:
                    idx = int(round(annotate_round_frac * (len(rounds_plot) - 1)))
                    annotate_target = (float(rounds_plot[idx]), float(map_values[idx]))
                if plot_entropy and entropy_values is not None:
                    dup_entropy_label = f"{dup_label} entropy"
                    dup_entropy_style = _style_for(dup_entropy_label, f"{label} entropy")
                    ax.plot(
                        rounds_plot,
                        entropy_values,
                        color=dup_entropy_style.get("color", dup_color),
                        linewidth=1.6,
                        linestyle=dup_entropy_style.get("linestyle", "dotted"),
                        label=_legend_label(dup_entropy_label),
                    )

    if f_max is not None:
        ax.axhline(f_max, color="black", linestyle=":", linewidth=1.5, label=r"$\mathrm{f}(x^{\star})$")

    if annotate_label is not None and annotate_target is not None:
        tip_dx, tip_dy = annotate_tip_offset
        ax.annotate(
            annotate_text,
            xy=annotate_target,
            xycoords=mtransforms.offset_copy(ax.transData, fig=fig, x=tip_dx, y=tip_dy, units="points"),
            xytext=(annotate_offset[0] - tip_dx, annotate_offset[1] - tip_dy),
            textcoords="offset points",
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=1.5 * annotate_arrow_scale,
                mutation_scale=10 * annotate_arrow_scale,
                shrinkA=0,
                shrinkB=5,
            ),
            zorder=10,
        )

    if plot_entropy:
        ax.set_title("$\\frac{{y}}{{y_{{\\max}}}}$ and posterior entropy over rounds")
        ax.set_ylabel("$\\frac{{y}}{{y_{{\\max}}}}$ & ${{\\mathcal{{H}}}}(p(x^\\star\\mid \\mathcal{{D}})$")
    else:
        ax.set_ylabel("activity")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, pad=title_pad)
    ax.set_xlabel(xlabel if xlabel is not None else "experimental round")
    ax.xaxis.label.set_size(label_size) # type: ignore
    ax.yaxis.label.set_size(label_size) # type: ignore
    ax.title.set_size(label_size) # type: ignore
    ax.tick_params(axis="both", labelsize=16)
    # pin the x-axis
    max_round_plotted = max(
        (r for _, _, rnds, _, _ in series for r in rnds),
        default=None,
    )
    if max_round_plotted is not None:
        ax.set_xlim(1.0, float(max_round_plotted))
    else:
        ax.set_xlim(left=1.0)
    ax.xaxis.set_major_locator(_MaxNLocatorWithOne(integer=True))
    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not label.startswith("_")
    ]
    if legend_deduplicate and filtered:
        seen = set()
        dedup_handles = []
        dedup_labels = []
        for handle, label in filtered:
            if label in seen:
                continue
            seen.add(label)
            dedup_handles.append(handle)
            dedup_labels.append(label)
        handles = dedup_handles
        labels = dedup_labels
    else:
        handles = [handle for handle, _ in filtered]
        labels = [label for _, label in filtered]

    legend = None
    legend_outside = legend_bbox_to_anchor is not None
    if handles:
        legend_kwargs = {
            "loc": legend_loc,
            "fontsize": 20,
            "frameon": True,
            "framealpha": 0.8,
            "edgecolor": "0.8",
            "fancybox": False,
        }
        if legend_outside:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            legend_kwargs["borderaxespad"] = 0.0
        if legend_ncol is not None:
            legend_kwargs["ncol"] = legend_ncol
        legend = ax.legend(handles, labels, **legend_kwargs)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_path = os.path.join("plots", output_filename)
    extra_artists = (legend,) if (legend is not None and legend_outside) else ()
    fig.savefig(output_path, dpi=130, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)

    return output_path

