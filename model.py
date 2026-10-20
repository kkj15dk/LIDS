import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# Variational distribution r_phi(x* | {y}_1..e).
class ExperimentCategorical(nn.Module):
    def __init__(self,
                 vocab_size: int, 
                 sequence_length: int, 
                 num_experiments: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.fc = nn.Linear(num_experiments, vocab_size * sequence_length)
    
    def _dist(self, y_e: torch.Tensor) -> "dist.Independent":
        logits = self.fc(y_e)  # [B_theta, vocab_size * sequence_length]
        logits = logits.view(-1, self.sequence_length, self.vocab_size)  # [B_theta, sequence_length, vocab_size]
        return dist.Independent(dist.Categorical(logits=logits), reinterpreted_batch_ndims=1)

    def forward(self, x_0: torch.Tensor, y_e: torch.Tensor) -> torch.Tensor:
        '''
        log probabilities log r_phi (x_0 | {y}_1..e) for a batch of sequences x_0 and experiments y_e.
        Args:
            x_0: [B_theta, sequence_length] - sequences of token indices
            y_e: [B_theta, E] - experiment indices
        Returns:
            log_r: [B_star] - log probabilities of the sequences under the variational distribution, conditioned on the experiments
        '''
        dist = self._dist(y_e)
        log_probs = dist.log_prob(x_0)

        return log_probs

    def sample(self, num_samples: int, y_e: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        '''
        Sample sequences from the variational distribution r_phi(x_0 | {y}_1..e) for a batch of experiments y_e.
        Args:
            num_samples: number of sequences to sample for each experiment
            y_e: [B_theta, E] - experiment indices
        Returns:
            samples: [B_theta, num_samples, sequence_length] - sampled sequences
        '''
        dist_obj = self._dist(y_e)
        # dist.sample((num_samples,)) returns [num_samples, B_theta, sequence_length].
        samples = dist_obj.sample((num_samples,)).permute(1, 0, 2).contiguous()
        return samples # [B_theta, num_samples, sequence_length]

    def get_topk(self, y_e: torch.Tensor, k: int, beam_width: int, chunk_size: int | None = None) -> torch.Tensor:
        '''
        Generate the exact top-k sequences of an independent position-wise categorical.
        This is implemented with batched top-k dynamic programming over positions.
        Args:
            y_e: [B_theta, E] - experiment values
            k: number of sequences to generate for each experiment
            beam_width: unused for this model; kept for API compatibility
            chunk_size: unused for this model; kept for API compatibility
        Returns:
            sequences: [B_theta, k, sequence_length] - most likely sequences
        '''
        assert y_e.dim() == 2, "y_e should be of shape [B_theta, E]"
        if k <= 0:
            raise ValueError("k must be > 0")
        if beam_width <= 0:
            raise ValueError("beam_width must be > 0")

        dist_obj = self._dist(y_e)
        logits = dist_obj.base_dist.logits  # [B_theta, sequence_length, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)  # [B_theta, sequence_length, vocab_size]

        batch_size = y_e.shape[0]
        seq_len = self.sequence_length
        vocab_size = self.vocab_size

        total_sequences = vocab_size ** seq_len
        if k > total_sequences:
            raise ValueError(f"k={k} exceeds total number of sequences K^L={total_sequences}")

        # Initialize from first position: top min(k, K) tokens.
        width = min(k, vocab_size)
        cur_scores, cur_tokens = torch.topk(log_probs[:, 0, :], k=width, dim=1)  # [B_theta, width]
        cur_sequences = cur_tokens.unsqueeze(-1)  # [B_theta, width, 1]

        # Extend sequences position-by-position, keeping exact top-k partial sequences.
        for pos in range(1, seq_len):
            pos_log_probs = log_probs[:, pos, :]  # [B_theta, vocab_size]

            # Candidate sums from cartesian product of current beams and position tokens.
            candidate_scores = cur_scores.unsqueeze(-1) + pos_log_probs.unsqueeze(1)  # [B_theta, width, vocab_size]
            candidate_scores = candidate_scores.reshape(batch_size, -1)  # [B_theta, width * vocab_size]

            next_width = min(k, candidate_scores.shape[1])
            next_scores, next_indices = torch.topk(candidate_scores, k=next_width, dim=1)

            parent_idx = next_indices // vocab_size  # [B_theta, next_width]
            next_token = next_indices % vocab_size   # [B_theta, next_width]

            gather_idx = parent_idx.unsqueeze(-1).expand(-1, -1, cur_sequences.shape[-1])
            next_sequences = cur_sequences.gather(dim=1, index=gather_idx)
            cur_sequences = torch.cat([next_sequences, next_token.unsqueeze(-1)], dim=-1)
            cur_scores = next_scores

        return cur_sequences[:, :k, :]