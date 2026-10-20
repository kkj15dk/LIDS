import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from typing import Optional, Tuple

import os

### Variational synthesis model q_t parameterized by psi
## Simple categorical distribution over the vocabulary
class Experiment(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 sequence_length: int,
                 num_experiments: int,
                 num_syntheses: int,
                 tau_warmup_steps: int = 0,
                 initialize_syntheses_per_experiment: bool = True,
                 freeze_experiment_weights: bool = False,
                 experiment_log_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_experiments = num_experiments
        self.num_syntheses = num_syntheses
        self._parameters_syntheses_unscaled = nn.Parameter(
            torch.randn(self.num_syntheses, sequence_length, vocab_size),
            requires_grad=True,
        )  # [num_syntheses, sequence_length, vocab_size] - unnormalized log probabilities of the categorical distribution (tau)

        # Get weights if initializing evenly distributed
        if initialize_syntheses_per_experiment: # if True, initialize so every synthesis is used at least once, with the fewest syntheses possible per experiment
            assert self.num_experiments > 0 and self.num_syntheses > 0, "num_experiments and num_syntheses must both be positive"
            assert experiment_log_weights is None, "experiment_log_weights must be None if initialize_syntheses_per_experiment is True, but got a tensor of shape {experiment_log_weights.shape}"
            experiment_weights = torch.zeros(self.num_experiments, self.num_syntheses)

            if self.num_experiments >= self.num_syntheses:
                # Minimum per experiment is exactly one synthesis.
                for e in range(self.num_experiments):
                    s = e % self.num_syntheses
                    experiment_weights[e, s] = 1.0
            else:
                # Need multiple syntheses per experiment; distribute as evenly as possible.
                base = self.num_syntheses // self.num_experiments
                remainder = self.num_syntheses % self.num_experiments
                start = 0
                for e in range(self.num_experiments):
                    count = base + (1 if e < remainder else 0)
                    end = start + count
                    experiment_weights[e, start:end] = 1.0 / count
                    start = end
            
            other_weights = 0.01 / self.num_syntheses # add small weight to all syntheses to ensure nonzero gradients for all parameters at initialization.
            experiment_log_weights = torch.log(experiment_weights + other_weights) # add small constant for numerical stability before taking log

        # Initialize the weights, either with the provided experiment_log_weights, or as random values.
        if experiment_log_weights is not None:
            assert experiment_log_weights.shape == (self.num_experiments, self.num_syntheses), f"experiment_log_weights must have shape ({self.num_experiments}, {self.num_syntheses}), but got {experiment_log_weights.shape}"

            # static experiment log weights when provided, otherwise learnable parameters initialized to random values
            self.parameters_experiments = nn.Parameter(
                experiment_log_weights.clone().detach(),
                requires_grad=True,
            )
        else:
            ### Mixture weights for each experiment (pi)
            self.parameters_experiments = nn.Parameter(
                torch.randn(self.num_experiments, self.num_syntheses),
                requires_grad=True,
            )  # [num_experiments, num_syntheses_total] - unnormalized log probabilities of the mixture weights over the syntheses, one for each experiment (pi)

        # Optionally keep the experiment weights fixed
        if freeze_experiment_weights:
            assert experiment_log_weights is not None, "If you freeze experiment weights, you should provide experiment_log_weights or set initialize_syntheses_per_experiment to specify the fixed weights, but got None."
            self.parameters_experiments.requires_grad_(False)

        if tau_warmup_steps > 0:
            self._parameters_syntheses_unscaled.requires_grad_(False)
            self.tau_is_frozen = True
        else:
            self.tau_is_frozen = False

    @property
    def logits_syntheses(self) -> torch.Tensor:
        # OBS: this is what should be used within this class for syntheses parameters!
        return self._parameters_syntheses_unscaled

    def _dist(self):
        probs_syntheses = self.probs_syntheses.unsqueeze(0).expand(
            self.num_experiments, -1, -1, -1
        )  # [num_experiments, num_syntheses, sequence_length, vocab_size]
        syntheses_dist = dist.Independent(
            dist.Categorical(probs=probs_syntheses),
            reinterpreted_batch_ndims=1
        )  # batch: [num_experiments, num_syntheses], event: [sequence_length]
        # Get the mixture weights over the syntheses for each experiment
        experiments_dist = dist.Categorical(probs=self.weights_experiments)  # [num_experiments, num_syntheses]

        # Get the mixture distribution
        mixture_dist = dist.MixtureSameFamily(
            mixture_distribution=experiments_dist,
            component_distribution=syntheses_dist,
        )  # [num_experiments, sequence_length, vocab_size]
        return mixture_dist
    
    def sample(self, shape: Tuple[int, ...]) -> torch.Tensor:
        '''
        Sample from the mixture distribution defined by the model parameters.
        Args:
            shape: the shape of the samples to draw
        Returns:
            samples: [num_samples, num_experiments, sequence_length] - samples drawn from the mixture distribution for each experiment
        '''
        return self._dist().sample(shape)

    def log_prob(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
        Get the log probability of the given input_ids under the mixture distribution defined by the model parameters.
        Args:
            input_ids: [num_samples, sequence_length] - input sequences for each experiment
        Returns:
            log_prob: [num_samples, num_experiments] - log probability of each input sequence under the mixture distribution for each experiment
        '''
        input_ids_expanded = input_ids.unsqueeze(1).expand(-1, self.num_experiments, -1) # [num_samples, num_experiments, sequence_length]
        return self._dist().log_prob(input_ids_expanded) # [num_samples, num_experiments]

    def log_prob_own(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
        Get the log probability of input_ids[..., e, :] under q_e only, for every e - unlike
        log_prob, this does not broadcast each input against every other experiment's mixture.
        Intended for input_ids produced by sample(), where dim -2 already indexes experiments.
        Args:
            input_ids: [..., num_experiments, sequence_length] - one input sequence per experiment
        Returns:
            log_prob: [..., num_experiments] - log probability of input_ids[..., e, :] under q_e
        '''
        return self._dist().log_prob(input_ids)

    def argmax(self) -> torch.Tensor:
        '''
        Get the argmax of the mixture distribution.
        Returns:
            x_0: [num_experiments, sequence_length]
        '''
        candidates = self.log_probs_syntheses.argmax(dim=-1)  # [num_syntheses, sequence_length]
        scores = self.log_prob(candidates)  # [num_syntheses, num_experiments]
        best_synthesis = torch.argmax(scores, dim=0)  # [num_experiments]
        return candidates[best_synthesis]  # [num_experiments, sequence_length]


    @property
    def probs_syntheses(self):
        '''
        'tau'
        Get the probabilities of the categorical distribution for each synthesis.
        Returns:
            probs_syntheses: [num_syntheses, sequence_length, vocab_size] - probabilities of the categorical distribution for each synthesis
        '''
        probs = F.softmax(self.logits_syntheses, dim=-1)
        return probs

    @property
    def log_probs_syntheses(self):
        '''
        'log_tau'
        Get the log probabilities of the categorical distribution for each synthesis.
        Returns:
            log_probs_syntheses: [num_syntheses, sequence_length, vocab_size] - log probabilities of the categorical distribution for each synthesis
        '''
        log_probs = F.log_softmax(self.logits_syntheses, dim=-1)
        return log_probs

    @property
    def weights_experiments(self):
        '''
        'pi'
        Get the mixture weights for each experiment.
        Technically this doesn't have to be over the simplex, but let's do so for now.
        Returns:
            weights_experiments: [num_experiments, num_syntheses] - mixture weights for each experiment
        '''
        weights = F.softmax(self.parameters_experiments, dim=-1)
        return weights
    
    @property
    def log_weights_experiments(self):
        '''
        'log_pi'
        Get the log mixture weights for each experiment.
        Technically this doesn't have to be over the simplex, but let's do so for now.
        Returns:
            log_weights_experiments: [num_experiments, num_syntheses] - log mixture weights for each experiment
        '''
        log_weights = F.log_softmax(self.parameters_experiments, dim=-1)
        return log_weights
    

def load_experiment_from_thompson_samples(
    source_run_dir: str,
    device: str,
) -> Experiment:
    """Load the Experiment from the run directory where the Thompson samples were saved."""
    experiment_path = os.path.join(source_run_dir, "thompson_experiment.pt")
    if not os.path.isfile(experiment_path):
        raise FileNotFoundError(f"Thompson experiment state dict not found at {experiment_path}")
    experiment_state_dict = torch.load(experiment_path, map_location=device)
    experiment = Experiment(
        vocab_size=experiment_state_dict["_parameters_syntheses_unscaled"].shape[-1],
        sequence_length=experiment_state_dict["_parameters_syntheses_unscaled"].shape[-2],
        num_experiments=experiment_state_dict["parameters_experiments"].shape[0],
        num_syntheses=experiment_state_dict["_parameters_syntheses_unscaled"].shape[0],
        tau_warmup_steps=0,
        initialize_syntheses_per_experiment=False,
        freeze_experiment_weights=False,
    ).to(device)
    experiment.load_state_dict(experiment_state_dict)
    experiment.eval()
    return experiment

def load_experiment_from_eig_checkpoint(
    ckpt_path: str,
    device: str,
) -> Experiment:
    """Load the Experiment submodule from an eig_model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    eig_state = checkpoint["eig_model"]

    tau_logits = eig_state["experiment._parameters_syntheses_unscaled"].to(device=device, dtype=torch.float32)
    pi_logits = eig_state["experiment.parameters_experiments"].to(device=device, dtype=torch.float32)

    experiment = Experiment(
        vocab_size=int(tau_logits.shape[-1]),
        sequence_length=int(tau_logits.shape[-2]),
        num_experiments=int(pi_logits.shape[0]),
        num_syntheses=int(tau_logits.shape[0]),
        tau_warmup_steps=0,
        initialize_syntheses_per_experiment=False,
        freeze_experiment_weights=False,
        experiment_log_weights=pi_logits,
    ).to(device)

    exp_state = {k[len("experiment."):]: v for k, v in eig_state.items() if k.startswith("experiment.")}
    experiment.load_state_dict(exp_state, strict=True)
    experiment.eval()
    return experiment


def thompson_samples_to_experiment(
        thompson_samples: torch.Tensor,
        sequence_length: int,
        vocab_size: int,
        num_experiments: int,
) -> Experiment:
    '''
    Convert the thompson samples to an Experiment object.
    Args:
        thompson_samples: [num_experiments, sequence_length] - the thompson samples for each experiment
        sequence_length: the length of the sequences
        vocab_size: the size of the vocabulary
        num_experiments: the number of experiments
    Returns:
        experiment: an Experiment object representing the thompson samples
    '''
    device = thompson_samples.device
    experiment = Experiment(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        num_experiments=num_experiments,
        num_syntheses=num_experiments,  # we will have one synthesis per experiment, since each synthesis is a delta mass on the thompson sample for that experiment
        tau_warmup_steps=0,  # not relevant for Thompson sampling, since we are not doing any optimization
    ).to(device)

    # Create the parameters for the syntheses distribution (delta mass on Thompson samples).
    parameters_syntheses = torch.full((num_experiments, sequence_length, vocab_size), fill_value=float("-inf"), device=device)
    token_ids = thompson_samples.to(dtype=torch.long, device=parameters_syntheses.device)
    if token_ids.shape != (num_experiments, sequence_length):
        raise ValueError(
            "thompson_samples must have shape "
            f"({num_experiments}, {sequence_length}), got {tuple(token_ids.shape)}"
        )
    if torch.any((token_ids < 0) | (token_ids >= vocab_size)):
        raise ValueError(f"Found token ids outside [0, {vocab_size - 1}] in thompson_samples")

    exp_idx = torch.arange(num_experiments, device=parameters_syntheses.device).unsqueeze(1)
    pos_idx = torch.arange(sequence_length, device=parameters_syntheses.device).unsqueeze(0)
    parameters_syntheses[exp_idx, pos_idx, token_ids] = 0.0

    # Create the parameters for the experiments distribution (log(I))
    parameters_experiments = torch.full((num_experiments, num_experiments), fill_value=float("-inf"), device=device)  # log probabilities for uniform distribution over syntheses
    for i in range(num_experiments):
        parameters_experiments[i, i] = 0.0  # log(1) = 0, so this creates a delta mass on the i-th synthesis for the i-th experiment
    
    experiment._parameters_syntheses_unscaled.data.copy_(parameters_syntheses)
    experiment.parameters_experiments.data.copy_(parameters_experiments)

    assert torch.equal(
        experiment.log_probs_syntheses,
        parameters_syntheses,
    ), "The log probabilities of the syntheses do not match the expected delta mass distribution."
    assert torch.equal(
        experiment.log_weights_experiments,
        parameters_experiments,
    ), "The log mixture weights of the experiments do not match the expected delta mass distribution."

    return experiment