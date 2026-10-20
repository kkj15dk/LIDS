from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_F
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim import optimize_acqf_discrete_local_search
from gpytorch.kernels import AdditiveKernel, ProductKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from lock_gp.lock_kernel import LinearGlobalLOCKKernel, NonlinearLocalLOCKKernel
from lock_gp.blosum50 import get_blosum50_matrix

from eval import _experiment_from_observation_block
from oracle import Oracle
from MCMC import ObservationBlock

# vocabulary used by some bump-oracle runs.
_Q_SPACE_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


class _BatchedLinearGlobalLOCKKernel(LinearGlobalLOCKKernel):
    """
    Batched version of LinearGlobalLOCKKernel (lock_gp.lock_kernel).
    """

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        if diag:
            return self.num_positions * torch.ones(x1.shape[:-2], dtype=x1.dtype, device=x1.device)
        global_exponent = self.log_global_exponent.exp().to(dtype=x1.dtype, device=x1.device)
        blosum = self.blosum.to(dtype=x1.dtype, device=x1.device)
        blosum = (global_exponent * blosum).exp()
        return torch.einsum("...bla,aA,...BlA->...bB", x1, blosum, x2)


class _BatchedNonlinearLocalLOCKKernel(NonlinearLocalLOCKKernel):
    """
    Batched version of NonlinearLocalLOCKKernel (lock_gp.lock_kernel).
    """

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        if diag:
            return torch.ones(x1.shape[:-2], dtype=x1.dtype, device=x1.device)
        global_exponent = self.log_global_exponent.exp().to(dtype=x1.dtype, device=x1.device)
        local_exponents = self.log_local_exponents.exp().to(dtype=x1.dtype, device=x1.device)
        blosum = self.blosum.to(dtype=x1.dtype, device=x1.device)
        blosum = global_exponent * local_exponents[:, None, None] * blosum
        return torch.einsum("...bla,...BlA,laA->...bB", x1, x2, blosum).exp()


def _build_batched_lock_kernel(num_positions: int, alphabet: Sequence[str]) -> gpytorch.kernels.Kernel:
    """
    Build a batched version of the lock kernel.
    """
    linear1 = _BatchedLinearGlobalLOCKKernel(num_positions=num_positions, alphabet=alphabet)
    linear2 = _BatchedLinearGlobalLOCKKernel(num_positions=num_positions, alphabet=alphabet)
    nonlinear = _BatchedNonlinearLocalLOCKKernel(num_positions=num_positions, alphabet=alphabet)
    return AdditiveKernel(
        ScaleKernel(
            ProductKernel(nonlinear, linear1),
            outputscale_prior=GammaPrior(2.0, 2.0),
            outputscale_constraint=GreaterThan(1e-4),
        ),
        ScaleKernel(
            linear2,
            outputscale_prior=GammaPrior(2.0, 2.0),
            outputscale_constraint=GreaterThan(1e-4),
        ),
    )


class _OneHotLockKernel(gpytorch.kernels.Kernel):
    """
    Adapts to the one-hot (..., L, A) input for lock_gp.lock_kernel's which has a different vocabulary.
    """

    has_lengthscale = False

    def __init__(self, sequence_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        lock_alphabet, _ = get_blosum50_matrix()
        # q_to_lock[q] = index of _Q_SPACE_ALPHABET[q] within lock_alphabet
        q_to_lock = torch.tensor([lock_alphabet.index(aa) for aa in _Q_SPACE_ALPHABET], dtype=torch.long)
        self.register_buffer("q_to_lock", q_to_lock)
        self.lock_alphabet_size = len(lock_alphabet)  # 21 (20 amino acids + gap token, unused by our data)
        self.lock_kernel = _build_batched_lock_kernel(num_positions=sequence_length, alphabet=lock_alphabet)

    def _to_lock_onehot(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., L] float64 token indices in q-space
        lock_indices = self.q_to_lock.to(device=x.device)[x.round().long()]  # [..., L]
        return torch_F.one_hot(lock_indices, num_classes=self.lock_alphabet_size).to(dtype=x.dtype)  # [..., L, A]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        return self.lock_kernel(self._to_lock_onehot(x1), self._to_lock_onehot(x2), diag=diag, **params)


def _build_kernel(
    kernel_name: str,
    sequence_length: int,
    vocab_size: int,
) -> gpytorch.kernels.Kernel:

    if kernel_name == "hamming":
        # exponential Hamming kernel
        return CategoricalKernel(ard_num_dims=sequence_length)
    elif kernel_name == "lock":
        if vocab_size != len(_Q_SPACE_ALPHABET):
            raise ValueError(
                f"kernel='lock' requires vocab_size={len(_Q_SPACE_ALPHABET)} (the standard amino "
                f"acids), got vocab_size={vocab_size!r}. LOCK is protein-specific - not applicable "
                "to e.g. the 4-letter DNA vocabulary some bump-oracle runs use."
            )
        return _OneHotLockKernel(sequence_length=sequence_length)
    elif kernel_name == "h_imq":
        raise NotImplementedError(f"kernel={kernel_name!r} is planned but not implemented yet.")
    else:
        raise ValueError(f"Unsupported kernel={kernel_name!r}. Use one of: hamming, h_imq, lock")


def _build_acquisition_function(acquisition_name: str, model: SingleTaskGP, best_f: float):
    if acquisition_name == "log_expected_improvement":
        # qLogExpectedImprovement (MC-based).
        return qLogExpectedImprovement(model=model, best_f=best_f)
    else:
        raise ValueError(
            f"Unsupported acquisition={acquisition_name!r}. Use one of: log_expected_improvement"
        )


def _optimize_discrete(
    acq_function,
    q: int,
    vocab_size: int,
    sequence_length: int,
    device: str,
    num_restarts: int,
    raw_samples: int,
    max_batch_size: int,
) -> torch.Tensor:
    """
    Optimize acq_function over the full K^L sequences via discrete local search.
    """
    discrete_choices = [
        torch.arange(vocab_size, dtype=torch.float64, device=device) for _ in range(sequence_length)
    ]
    # No need to spend memory on gradients here
    with torch.no_grad():
        candidates, _ = optimize_acqf_discrete_local_search(
            acq_function=acq_function,
            discrete_choices=discrete_choices,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            max_batch_size=max_batch_size,
        )
    return candidates.round().to(dtype=torch.long)


def observation_blocks_to_xy(
    blocks: Sequence[ObservationBlock],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert to (x, y)  from saved ObservationBlocks.
    Only works for delta mass experiments (gp_acqf, or thompson/argmax_lifted).

    Returns: (train_X [N,L] float64, train_Y [N,1] float64).
    """
    x_chunks = []
    y_chunks = []
    for block in blocks:
        experiment = _experiment_from_observation_block(block, device=device)
        candidates = experiment.argmax()  # [E, L] long
        # Delta-mass sanity check: every experiment's own candidate should carry ~all its
        # probability mass (log_prob ~= 0) under q_e - see docstring above.
        self_log_prob = experiment.log_prob(candidates).diagonal()
        if not torch.allclose(self_log_prob, torch.zeros_like(self_log_prob), atol=1e-4):
            raise ValueError(
                "observation_blocks_to_xy: a block's Experiment is not delta-mass"
            )
        if block.y is None:
            raise ValueError("observation_blocks_to_xy: block.y is None; every gp_acqf block must have y.")

        y_np = np.array(block.y, dtype=np.float64, copy=True)
        x_chunks.append(candidates.to(dtype=torch.float64))
        y_chunks.append(torch.as_tensor(y_np, dtype=torch.float64, device=device))

    train_X = torch.cat(x_chunks, dim=0)  # [N, L]
    train_Y = torch.cat(y_chunks, dim=0).unsqueeze(-1)  # [N, 1]
    return train_X, train_Y


def _subsample_for_mll_fit(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bound the dataset for _fit_gp's MLL optimization for OOM safety.
    """
    n = train_X.shape[0]
    if max_points <= 0 or n <= max_points:
        return train_X, train_Y

    y_flat = train_Y.squeeze(-1)
    n_top = max(1, max_points // 4)
    top_idx = torch.topk(y_flat, k=min(n_top, n)).indices

    remaining_mask = torch.ones(n, dtype=torch.bool, device=train_X.device)
    remaining_mask[top_idx] = False
    remaining_idx = torch.nonzero(remaining_mask, as_tuple=True)[0]

    n_fill = max_points - top_idx.numel()
    if n_fill > 0 and remaining_idx.numel() > 0:
        perm = torch.randperm(remaining_idx.numel())[: min(n_fill, remaining_idx.numel())]
        keep_idx = torch.cat([top_idx, remaining_idx[perm]])
    else:
        keep_idx = top_idx

    return train_X[keep_idx], train_Y[keep_idx]


def _build_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    sigma: float,
    kernel_name: str,
    sequence_length: int,
    vocab_size: int,
) -> SingleTaskGP:
    """
    Build SingleTaskGP with known observation noise.
    """
    train_Yvar = torch.full_like(train_Y, float(sigma) ** 2)
    return SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        covar_module=_build_kernel(kernel_name, sequence_length, vocab_size),
    ).to(train_X.device)


def _fit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    sigma: float,
    kernel_name: str,
    sequence_length: int,
    vocab_size: int,
    max_train_points: int = 0,
) -> SingleTaskGP:
    """
    Fit a SingleTaskGP with known observation noise.
    Fits with fit_gpytorch_mll_torch (Adam).
    """
    fit_X, fit_Y = _subsample_for_mll_fit(train_X, train_Y, max_train_points)
    fitted_model = _build_gp(fit_X, fit_Y, sigma, kernel_name, sequence_length, vocab_size)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(fitted_model.likelihood, fitted_model), optimizer=fit_gpytorch_mll_torch)

    if fit_X.shape[0] == train_X.shape[0]:
        return fitted_model

    final_model = _build_gp(train_X, train_Y, sigma, kernel_name, sequence_length, vocab_size)
    final_model.mean_module.load_state_dict(fitted_model.mean_module.state_dict())
    final_model.covar_module.load_state_dict(fitted_model.covar_module.state_dict())
    return final_model


def select_candidates(
    previous_blocks: Sequence[ObservationBlock],
    sequence_length: int,
    vocab_size: int,
    num_experiments: int,
    sigma: float,
    kernel_name: str,
    acquisition_name: str,
    num_restarts: int,
    raw_samples: int,
    max_batch_size: int,
    max_train_points: int,
    device: str,
) -> torch.Tensor:
    """
    Select candidate sequences for the current round.

    Returns: [num_experiments, sequence_length] long.
    """
    if len(previous_blocks) == 0:
        return torch.randint(0, vocab_size, (num_experiments, sequence_length), device=device)

    train_X, train_Y = observation_blocks_to_xy(previous_blocks, device=device)
    model = _fit_gp(train_X, train_Y, sigma=sigma, kernel_name=kernel_name, sequence_length=sequence_length, vocab_size=vocab_size, max_train_points=max_train_points)
    acqf = _build_acquisition_function(acquisition_name, model=model, best_f=float(train_Y.max()))
    return _optimize_discrete(
        acqf,
        q=num_experiments,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        device=device,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        max_batch_size=max_batch_size,
    )


def compute_map_metrics(
    conditioning_blocks: Sequence[ObservationBlock],
    oracle: Oracle,
    sigma: float,
    kernel_name: str,
    num_restarts: int,
    raw_samples: int,
    max_batch_size: int,
    max_train_points: int,
    device: str,
) -> Dict[str, Any]:
    """
    Compute MAP metrics for the current round.

    Returns a dictionary with the following keys:
    - "MAP_ratio": The ratio of the MAP value to the true value.
    - "true_value": The true value.
    - "map_value": The MAP value.
    - "num_observations": The number of observations.
    - "x_0_map": The MAP point.
    """
    train_X, train_Y = observation_blocks_to_xy(conditioning_blocks, device=device)
    sequence_length = train_X.shape[-1]
    model = _fit_gp(train_X, train_Y, sigma=sigma, kernel_name=kernel_name, sequence_length=sequence_length, vocab_size=int(oracle.K), max_train_points=max_train_points)

    x_map = _optimize_discrete(
        PosteriorMean(model),
        q=1,
        vocab_size=int(oracle.K),
        sequence_length=sequence_length,
        device=device,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        max_batch_size=max_batch_size,
    ).squeeze(0)  # [L]

    y_true = oracle.f(oracle.x_0.unsqueeze(0)).squeeze(0)
    y_map = oracle.f(x_map.unsqueeze(0)).squeeze(0)

    return {
        "MAP_ratio": float(y_map / y_true),
        "true_value": float(y_true),
        "map_value": float(y_map),
        "num_observations": int(train_X.shape[0]),
        "x_0_map": x_map.detach().cpu().numpy(),
    }
