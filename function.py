import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import torch.distributions as dist
import numpyro.distributions as np_dist
from typing import Callable, Optional, Tuple, Union, List

from experiment import Experiment


### helper functions
def one_hot_encode(sequences: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    One-hot encode a batch of sequences.
     - sequences: [batch_size, sequence_length] tensor of integer indices representing the sequences
     - vocab_size: size of the vocabulary (number of unique tokens)
    Returns:
     - one_hot: [batch_size, vocab_size, sequence_length] one-hot encoded representation of the sequences
    """
    batch_size, sequence_length = sequences.size()
    one_hot = F.one_hot(sequences, num_classes=vocab_size)  # [batch_size, sequence_length, vocab_size]
    one_hot = one_hot.transpose(-2, -1)  # [batch_size, vocab_size, sequence_length]
    return one_hot.float()

def one_hot_decode(one_hot: torch.Tensor) -> torch.Tensor:
    """
    Decode one-hot encoded sequences back to integer indices.
     - one_hot: [batch_size, vocab_size, sequence_length] one-hot encoded representation of the sequences
    Returns:
     - sequences: [batch_size, sequence_length] tensor of integer indices representing the sequences
    """
    one_hot = one_hot.transpose(-2, -1)  # [batch_size, sequence_length, vocab_size]
    sequences = torch.argmax(one_hot, dim=-1)  # [batch_size, sequence_length]
    return sequences

def compute_log_linear_predictor(log_w, 
               log_v, 
               log_pi, 
               log_tau, 
               inverse_link_function, 
) -> jnp.ndarray:
    """
    Compute mean of likelihood under GLM.
    """
    assert log_pi is not None, "If log_tau is provided, log_pi must also be provided"
    assert log_tau.ndim == 3, f"log_tau must have 3 dimensions, but has shape {log_tau.shape}"
    assert log_pi.ndim == 2, f"log_pi must have 2 dimensions, but has shape {log_pi.shape}"
    S = log_tau.shape[0]
    L = log_tau.shape[1]
    K = log_tau.shape[2]
    E = log_pi.shape[0]
    assert log_pi.shape == (E, S), f"log_pi shape {log_pi.shape} does not match expected shape {(E, S)}"
    assert log_tau.shape == (S, L, K), f"log_tau shape {log_tau.shape} does not match expected shape {(S, L, K)}"

    # Likelihood computation in log-space
    # Broadcast shapes: log_tau [S,L,K], log_v [N,L,K] -> inner product over L,K
    log_tau_exp = jnp.expand_dims(log_tau, axis=1)  # [S, 1, L, K]
    log_v_exp = jnp.expand_dims(log_v, axis=0)  # [1, N, L, K]
    
    # Sum inner products: logsumexp over K (vocab), sum over L (positions)
    log_inner_product = log_tau_exp + log_v_exp  # [S, N, L, K]
    log_inner_product = jax.nn.logsumexp(log_inner_product, axis=-1)  # [S, N, L]
    log_inner_product = jnp.sum(log_inner_product, axis=-1)  # [S, N]

    # Weighted sum over components: add log_w and marginalize
    log_w_exp = jnp.expand_dims(log_w, axis=0)  # [1, N]
    log_h_of_tau = log_inner_product + log_w_exp  # [S, N]
    log_f_of_tau = jax.nn.logsumexp(log_h_of_tau, axis=-1)  # [S]
    
    # Marginalize over syntheses: add log_pi and logsumexp
    log_f_of_q = log_f_of_tau[None, :] + log_pi  # [1, S] + [E, S] -> broadcast [E, S]
    log_linear_predictor = jax.nn.logsumexp(log_f_of_q, axis=-1)  # [E]
    
    return log_linear_predictor

### Generalized Linear Model (GLM)
class GLM(nn.Module):
    def __init__(self):
        super().__init__()

    def _dist(self, mean: torch.Tensor, sigma: Union[torch.Tensor, float]) -> dist.Distribution:
        raise NotImplementedError("GLM subclasses must implement _dist().")
    
    @property
    def monotone_increasing_link(self) -> bool:
        """Whether the link function and inverse link function are monotone increasing. This is important to evaluate x_0 = argmax_x inverse_link_function(f_theta(x)). Should be implemented by each GLM subclass."""
        raise NotImplementedError("GLM subclasses must implement monotone_increasing_link property.")
    
    @property
    def monotone_decreasing_link(self) -> bool:
        """Whether the link function and inverse link function are monotone decreasing. This is important to evaluate x_0 = argmax_x inverse_link_function(f_theta(x)). Should be implemented by each GLM subclass."""
        raise NotImplementedError("GLM subclasses must implement monotone_decreasing_link property.")

    def link_function(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("GLM subclasses must implement link_function().")

    def inverse_link_function(self, x: torch.Tensor) -> torch.Tensor:
        """Also known as the mean function, maps from the linear predictor to the mean of the distribution."""
        raise NotImplementedError("GLM subclasses must implement inverse_link_function().")
    
    def log_prob_null(self, y_star: torch.Tensor, y_ts: torch.Tensor, std_floor: torch.Tensor) -> torch.Tensor:
        """Log probability of y under the null model trained on the data. Should be implemented by each GLM subclass."""
        raise NotImplementedError("GLM subclasses must implement log_prob_null().")

    def rsample(self, linear_predictor: torch.Tensor, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        mean = self.inverse_link_function(linear_predictor)
        dist = self._dist(mean, sigma)
        return dist.rsample()

    def log_prob(self, linear_predictor: torch.Tensor, sigma: Union[torch.Tensor, float], y: torch.Tensor) -> torch.Tensor:
        mean = self.inverse_link_function(linear_predictor)
        dist = self._dist(mean, sigma)
        return dist.log_prob(y)

class NormalGLM(GLM):
    def __init__(self):
        super().__init__()

    def _dist(self, mean: torch.Tensor, sigma: Union[torch.Tensor, float]) -> dist.Normal:
        """
        Given the GLM parameters (the mean), return a Normal distribution with that mean and provided standard deviation.
         - mean: the mean of the distribution
         - sigma: standard deviation of the distribution
        """
        return dist.Normal(loc=mean, scale=sigma)
    
    @property
    def monotone_increasing_link(self) -> bool:
        '''
        Wheter the link function and inverse link function are monotone increasing.
        This is important to evaluate x_0 = argmax_x inverse_link_function(f_theta(x))
        '''
        return True  # The identity link function is monotone increasing
    
    @property
    def monotone_decreasing_link(self) -> bool:
        '''
        Wheter the link function and inverse link function are monotone decreasing.
        This is important to evaluate x_0 = argmax_x inverse_link_function(f_theta(x))
        '''
        return False  # The identity link function is not monotone decreasing

    def mean(self, linear_predictor: torch.Tensor) -> torch.Tensor:
        return self.inverse_link_function(linear_predictor)  # The mean of the normal distribution is given by the inverse link function applied to the linear predictor

    def link_function(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Identity link function for normal distribution

    def inverse_link_function(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Identity link function for normal distribution

### Fitness function class
class FitnessFunction(nn.Module):
    def __init__(self, 
                 num_components: int,
                 sequence_length: int,
                 vocab_size: int,
                 glm: GLM,
    ):
        super().__init__()
        self.N = num_components
        self.L = sequence_length
        self.K = vocab_size
        self.glm = glm

        assert self.glm.monotone_increasing_link, "GLM must have a monotone increasing link function to use argmax."

    ### Integral calculation by inner product of two categoricals <f_theta, q_t>
    # As f_theta and q_t are both categorical distributions, we can compute the integral exactly as the inner product of their probability vectors.
    def log_functional_inner_product(self,
                                     log_tau: torch.Tensor,
                                     log_pi: torch.Tensor,
                                     log_w: torch.Tensor, 
                                     log_v: torch.Tensor, 
    ) -> torch.Tensor:
        '''
        Compute <f,q>
        Args:
            log_tau: [num_syntheses, sequence_length, vocab_size]
            log_pi: [num_experiments, num_syntheses]
            log_w: [batch_size, num_components]
            log_v: [batch_size, num_components, sequence_length, vocab_size]
        Returns:
            log_inner_product: [batch_size, num_experiments] - the log of the inner product between f_theta and q
        '''
        log_tau = log_tau.unsqueeze(1).unsqueeze(0) # [1, num_syntheses, 1, sequence_length, vocab_size]
        log_pi = log_pi.unsqueeze(0) # [1, num_experiments, num_syntheses]
        log_v = log_v.unsqueeze(1) # [batch_size, 1, num_components, sequence_length, vocab_size]
        log_w = log_w.unsqueeze(1) # [batch_size, 1, num_components]

        # inner product
        log_inner_product = log_v + log_tau # [batch_size, num_syntheses, num_components, sequence_length, vocab_size]
        log_inner_product = torch.logsumexp(log_inner_product, dim=-1)  # [batch_size, num_syntheses, num_components, sequence_length]

        # product over L
        log_inner_product = log_inner_product.sum(dim=-1)  # [batch_size, num_syntheses, num_components]
        
        # multiply by w
        log_inner_product = log_inner_product + log_w  # [batch_size, num_syntheses, num_components]

        # sum over N
        log_inner_product = torch.logsumexp(log_inner_product, dim=-1)  # [batch_size, num_syntheses]
        log_inner_product = log_inner_product.unsqueeze(1) # [batch_size, 1, num_syntheses]

        # multiply by pi
        log_inner_product = log_inner_product + log_pi  # [batch_size, num_experiments, num_syntheses]
        log_inner_product = torch.logsumexp(log_inner_product, dim=-1)  # [batch_size, num_experiments]

        return log_inner_product

    def rsample_y(
        self,
        log_tau: torch.Tensor,
        log_pi: torch.Tensor,
        log_w: torch.Tensor,
        log_v: torch.Tensor,
        sigma: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        log_integral_estimate = self.log_functional_inner_product(
            log_tau = log_tau,
            log_pi = log_pi,
            log_w = log_w,
            log_v = log_v,
        ) # [B_theta, num_experiments]

        integral_estimate = log_integral_estimate.exp() # [B_theta, num_experiments]
        linear_predictor = integral_estimate # [B_theta, num_experiments]

        if deterministic:
            mean_y = self.glm.inverse_link_function(linear_predictor) # [B_theta, num_experiments]
            return mean_y

        # sample with reparameterization trick
        y = self.glm.rsample(linear_predictor=linear_predictor, sigma=sigma) # [B_theta, num_experiments]
        return y

    def log_f(self, x: torch.Tensor, log_w: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        '''
        x: [..., num_samples, sequence_length]
        log_w: [..., num_components]
        log_v: [..., num_components, sequence_length, vocab_size]
        '''
        # For each sample x_s: log f(x_s) = logsumexp_n [log_w_n + sum_l log_v_{n,l,x_s(l)}].
        # x includes an explicit sample axis, so output shape is [..., num_samples].
        x_idx = x.unsqueeze(-2).unsqueeze(-1)  # [..., num_samples, 1, sequence_length, 1]
        x_idx = x_idx.expand(*x.shape[:-2], x.shape[-2], log_v.shape[-3], x.shape[-1], 1)  # [..., num_samples, num_components, sequence_length, 1]

        log_v_expanded = log_v.unsqueeze(-4).expand(*x.shape[:-2], x.shape[-2], *log_v.shape[-3:])  # [..., num_samples, num_components, sequence_length, vocab_size]
        log_v_x = torch.gather(log_v_expanded, dim=-1, index=x_idx).squeeze(-1)  # [..., num_samples, num_components, sequence_length]

        log_f = log_v_x.sum(dim=-1) + log_w.unsqueeze(-2)  # [..., num_samples, num_components]
        log_f = torch.logsumexp(log_f, dim=-1)  # [..., num_samples]

        return log_f

    def argmax(self, log_w: torch.Tensor, log_v: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            candidates: [..., num_candidates, sequence_length]
            log_w: [..., num_components]
            log_v: [..., num_components, sequence_length, vocab_size]
        Returns:
            x_0: [..., sequence_length]
        '''
        # compute f_theta for each candidate and take the argmax
        log_f_candidates = self.log_f(x=candidates, log_w=log_w, log_v=log_v) # [..., num_candidates]
        idx = torch.argmax(log_f_candidates, dim=-1) # [...]
        batch_idx = torch.arange(log_w.shape[0], device=log_w.device) # [...]
        x_0 = candidates[batch_idx, idx] # [..., sequence_length]
        return x_0

    def f(self, x: torch.Tensor, log_w: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: [..., num_samples, sequence_length]
            log_w: [..., num_components]
            log_v: [..., num_components, sequence_length, vocab_size]
        Returns:
            f_theta(x): [..., num_samples]
        '''
        return torch.exp(self.log_f(x, log_w, log_v))
    
    def _dist(self, log_w: torch.Tensor, log_v: torch.Tensor) -> dist.MixtureSameFamily:
        '''
        Get the mixture distribution proportional to the function f_theta(x).
        Useful for visualizing different functions.
        Arguments:
            log_w: [num_components] - log probabilities of the mixture components
            log_v: [num_components, sequence_length, vocab_size] - log probabilities of the categorical distribution for each component
        '''
        w_dist = dist.Categorical(logits=log_w)  # [num_components]
        v_dist = dist.Independent(
            dist.Categorical(logits=log_v),
            reinterpreted_batch_ndims=1
        )  # batch: [num_components], event: [sequence_length]

        # Get the mixture distribution
        mixture_dist = dist.MixtureSameFamily(
            mixture_distribution=w_dist,
            component_distribution=v_dist,
        )  # [num_experiments, sequence_length, vocab_size]
        return mixture_dist

    def sample_f_as_dist(self, log_w: torch.Tensor, log_v: torch.Tensor, num_samples: int) -> torch.Tensor:
        '''
        Sample from the distribution defined by f_theta(x) (normalized).
        '''
        # Sample from the categorical distribution defined by the unnormalized log probabilities log_f(x)
        mixture_dist = self._dist(log_w=log_w, log_v=log_v)
        samples = mixture_dist.sample((num_samples,))  # [num_samples, sequence_length]
        return samples

    def generate_summary_statistics(self, 
                                    theta_samples: dict, 
                                    path: str,
    ):
        '''
        Generate histogram of max_x f_theta(x) for the sampled theta values and save to a file.
        '''
        log_w = torch.as_tensor(theta_samples["log_w"], dtype=torch.float32) # [B, num_components]
        log_v = torch.as_tensor(theta_samples["log_v"], dtype=torch.float32) # [B, num_components, sequence_length, vocab_size]
        candidates = torch.as_tensor(theta_samples["mode"], dtype=torch.long) # [B, sequence_length]

        x_0 = self.argmax(log_w=log_w, log_v=log_v, candidates=candidates).unsqueeze(1) # [B, 1, sequence_length]
        f_x_0 = self.f(x=x_0, log_w=log_w, log_v=log_v) # [B]

        # Plot histogram of f_x_0 with robust bin selection.
        f_x_0_np = f_x_0.detach().cpu().numpy()

        # Handle non-finite values.
        finite_vals = f_x_0_np[np.isfinite(f_x_0_np)]
        num_dropped = f_x_0_np.size - finite_vals.size
        if num_dropped > 0:
            print(f"generate_summary_statistics: dropping {num_dropped}/{f_x_0_np.size} non-finite max_x f_theta(x) values before plotting.")

        def _epsilon_bins(center: float) -> list:
            eps = max(1e-8, 1e-6 * abs(center))
            return [center - eps, center + eps]

        plt.figure(figsize=(8, 6))
        title = r'Distribution of $\max_x f_\theta(x)$'
        if finite_vals.size == 0:
            title += ' (no finite values to plot)'
        else:
            unique_vals = np.unique(finite_vals)
            if unique_vals.size == 1:
                plt.hist(finite_vals, bins=_epsilon_bins(float(unique_vals[0])), density=True)
            else:
                # Use up to 50 bins but not more than the number of unique values.
                bin_count = min(50, int(unique_vals.size))
                try:
                    plt.hist(finite_vals, bins=bin_count, density=True)
                except ValueError:
                    print(
                        f"generate_summary_statistics: {unique_vals.size} unique max_x f_theta(x) "
                        "values are too close together to bin at float precision; widening to a "
                        "single bin around their mean."
                    )
                    title += ' (near-identical values; widened bin)'
                    plt.hist(finite_vals, bins=_epsilon_bins(float(finite_vals.mean())), density=True)
        plt.xlabel(r'$\max_x f_\theta(x)$')
        plt.ylabel('Density')
        plt.title(title)
        plt.savefig(path)
        plt.close()