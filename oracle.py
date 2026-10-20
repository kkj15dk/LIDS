from torch import nn
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
import torch.distributions as dist
import math
from typing import Tuple, List, Optional, cast, Any, Union

from function import GLM, NormalGLM
from experiment import Experiment, thompson_samples_to_experiment
from function import FitnessFunction
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM
from timing import print_time

def cross_entropy(logits, target, reduction='mean'):
    return F.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)


class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_0: torch.Tensor = torch.empty(0, dtype=torch.long) # subclasses overwrite with the sequence with the highest activity

    def f(self, x: torch.Tensor) -> torch.Tensor:
        '''
        The oracle function f(x) that maps a sequence x to a real-valued activity. This is the function we want to optimize.

        Args:
            x: a batch of sequences, shape [batch_size, sequence_length]
        Returns:
            activity: the computed activity values for the input sequences, shape [batch_size]
        '''
        raise NotImplementedError("Subclasses should implement this method")

    def log_functional_inner_product(self, 
                                     experiment: Experiment,
                                     return_diagnostics: bool = False,
        ) -> Tuple[torch.Tensor, Any]:
        '''
        Compute the functional inner product <f, q> where q is the distribution of sequences defined by the experiment.

        Args:
            experiment: the experiment defining the distribution q
        Returns:
            log_fq: the computed log functional inner product, shape [num_experiments]
            diagnostics: any additional information for debugging or analysis
        '''
        raise NotImplementedError("Subclasses should implement this method")

    def rsample_y(self,
                  experiment: Experiment,
                  sigma: float,
                  return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        '''
        Sample a noisy observation y from the oracle given an experiment.

        Args:
            experiment: the experiment to sample from
            sigma: the noise level
        Returns:
            y: the noisy observation
            diagnostics: any additional information for debugging or analysis
        '''
        return NotImplementedError("Subclasses should implement this method") # type: ignore

    def sample_f_as_dist(
            self,
            num_samples: int,
    ) -> torch.Tensor:
        '''
        Sample from f(x) for visualization purposes.
        '''
        raise NotImplementedError("Subclasses should implement this method")

class OracleBump(Oracle):
    def __init__(
        self,
        num_components: int,
        sequence_length: int,
        vocab_size: int,
        log_w_true: torch.Tensor,
        log_v_true: torch.Tensor,
        glm: GLM,
    ):
        '''
        Oracle for f(x), the molecule-activity map, that has a "bump" of high activity around a specific sequence (the center).
        Every double mutation abolishes activity.

        Args:
            log_w_true: the true log w parameters of the oracle, shape [N]
            log_v_true: the true log v parameters of the oracle, shape [N, L, K]
            glm: the generalized linear model (noise model)
        '''
        super().__init__()
        self.N = num_components
        self.L = sequence_length
        self.K = vocab_size
        self.log_w_true = log_w_true
        self.log_v_true = log_v_true
        self.glm = glm
        assert log_w_true.shape == (num_components,)
        assert log_v_true.shape == (num_components, sequence_length, vocab_size)

        candidates = torch.argmax(self.log_v_true, dim=-1) # [N, L]
        log_f_candidates = self._log_f(x=candidates) # [N]
        idx = torch.argmax(log_f_candidates, dim=-1)
        self.x_0 = candidates[idx] # [L]

    def _enumerate_single_mutation_support(self) -> torch.Tensor:
        """
        Enumerate x with Hamming distance < 2 from x_0_true (0 or 1 mutation).
        Returns: [1 + L * (K - 1), L]
        """
        x_0_true = self.x_0
        if x_0_true.ndim != 1:
            x_0_true = x_0_true.squeeze(0)
        if x_0_true.ndim != 1:
            raise ValueError(f"Expected x_0_true to be 1D, got shape {tuple(x_0_true.shape)}")

        x_0_cpu = x_0_true.detach().cpu()
        L = int(x_0_cpu.shape[0])
        support = [x_0_cpu]
        for pos in range(L):
            token_at_pos = int(x_0_cpu[pos].item())
            for token in range(self.K):
                if token == token_at_pos:
                    continue
                x_mut = x_0_cpu.clone()
                x_mut[pos] = token
                support.append(x_mut)
        return torch.stack(support, dim=0).to(device=x_0_true.device)
    
    def _log_f(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: [..., num_samples, sequence_length]
        '''
        log_w = self.log_w_true
        log_v = self.log_v_true

        # For each sample x_s: log f(x_s) = logsumexp_n [log_w_n + sum_l log_v_{n,l,x_s(l)}].
        # x includes an explicit sample axis, so output shape is [..., num_samples].
        x_idx = x.unsqueeze(-2).unsqueeze(-1)  # [..., num_samples, 1, sequence_length, 1]
        x_idx = x_idx.expand(*x.shape[:-2], x.shape[-2], log_v.shape[-3], x.shape[-1], 1)  # [..., num_samples, num_components, sequence_length, 1]

        log_v_expanded = log_v.unsqueeze(-4).expand(*x.shape[:-2], x.shape[-2], *log_v.shape[-3:])  # [..., num_samples, num_components, sequence_length, vocab_size]
        log_v_x = torch.gather(log_v_expanded, dim=-1, index=x_idx).squeeze(-1)  # [..., num_samples, num_components, sequence_length]

        log_f = log_v_x.sum(dim=-1) + log_w.unsqueeze(-2)  # [..., num_samples, num_components]
        log_f = torch.logsumexp(log_f, dim=-1)  # [..., num_samples]

        return log_f

    def _f_unmasked(self, x: torch.Tensor) -> torch.Tensor:
        '''
        The non-oracle version (where double mutants still have activity)
        Arguments:
            x: [..., num_samples, sequence_length]
        Returns:
            f_theta(x): [..., num_samples]
        '''
        return torch.exp(self._log_f(x=x))

    def f(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        f'(x) = f(x) * 1(|x*, x| < 2) where x* = x_0_true.
        """
        x_0_true = self.x_0

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x_0_true.ndim != 1:
            x_0_true = x_0_true.squeeze(0)
        if x_0_true.ndim != 1:
            raise ValueError(f"Expected x_0_true to be 1D, got shape {tuple(x_0_true.shape)}")

        f_val = self._f_unmasked(x=x)
        hamming = (x != x_0_true).sum(dim=-1)
        mask = (hamming < 2).to(dtype=f_val.dtype)
        return f_val * mask

    def _log_q_experiment(self, experiment: Experiment, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log q_t(x) for each experiment.
        x: [num_sequences, sequence_length]
        Returns: [num_sequences, num_experiments]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D [num_sequences, L], got shape {tuple(x.shape)}")
        log_tau = experiment.log_probs_syntheses  # [S, L, K]
        log_pi = experiment.log_weights_experiments  # [E, S]

        num_sequences, L = int(x.shape[0]), int(x.shape[1])
        if L != int(log_tau.shape[1]):
            raise ValueError(
                f"Sequence length mismatch: x has L={L}, log_tau has L={int(log_tau.shape[1])}"
            )

        x = x.to(dtype=torch.long)
        log_tau_exp = log_tau.unsqueeze(0).expand(num_sequences, -1, -1, -1)  # [M, S, L, K]
        x_idx = x[:, None, :, None].expand(num_sequences, log_tau.shape[0], L, 1)  # [M, S, L, 1]
        log_tau_x = torch.gather(log_tau_exp, dim=-1, index=x_idx).squeeze(-1)  # [M, S, L]
        log_prob_s = log_tau_x.sum(dim=-1)  # [M, S]
        log_q = torch.logsumexp(log_prob_s[:, None, :] + log_pi[None, :, :], dim=-1)  # [M, E]
        return log_q

    def log_functional_inner_product_masked(
        self,
        experiment: Experiment,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Compute log <f', q_t> by enumerating sequences within 1 mutation of x_0_true.
        """
        log_w = self.log_w_true
        log_v = self.log_v_true

        if log_w.ndim != 1 or log_v.ndim != 3:
            raise ValueError(
                f"Expected log_w/log_v shapes [N]/[N,L,K], got {tuple(log_w.shape)} / {tuple(log_v.shape)}"
            )

        support = self._enumerate_single_mutation_support()
        log_f_support = self._log_f(x=support)  # [M]
        log_q_support = self._log_q_experiment(experiment=experiment, x=support)  # [M, E]
        log_inner = torch.logsumexp(log_f_support[:, None] + log_q_support, dim=0)  # [E]
        return log_inner, None

    def rsample_y(
        self,
        experiment: Experiment,
        sigma: Union[torch.Tensor, float],
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        log_integral_estimate, diagnostics = self.log_functional_inner_product_masked(
            experiment=experiment,
            return_diagnostics=return_diagnostics,
        )
        integral_estimate = log_integral_estimate.exp()
        linear_predictor = integral_estimate

        y = self.glm.rsample(linear_predictor=linear_predictor, sigma=sigma)
        return y, diagnostics

    def sample_f_as_dist(
        self,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample from f'(x) by enumerating sequences within 1 mutation of x_0_true.
        """

        # Keep all tensors on the same device as the model parameters.
        support = self._enumerate_single_mutation_support()  # [M, L]
        log_f_support = self._log_f(x=support)  # [M]
        log_f_support = log_f_support - torch.logsumexp(log_f_support, dim=0)
        dist_support = dist.Categorical(logits=log_f_support)
        idx = dist_support.sample((num_samples,))
        return support[idx]


class OraclePLM(Oracle):
    def __init__(self,
                 wt_str: str,
                 continuation_L: int,
                 beta: float,
                 glm: GLM,
                 device: torch.device,
                 num_samples: int,
                 batch_size: int,
                 ckpt: str = 'progen2-base',
                 model_name: str = 'progen2-base',
                 tokenizer_file: str = 'progen/progen2/tokenizer.json',
                 dtype: torch.dtype = torch.float16,
                 defensive_alpha: float = 0.5,
    ):
        '''
        Oracle for f(x), the molecule-activity map, using a PLM
            f(x) = ReLU(log [p(x|x_prefix)/p(x_wt|x_prefix)] + beta)
        where p(x|x_prefix) is the likelihood of the PLM generating sequence x given prefix x_prefix, and x_wt is the wild-type sequence.

        Args:
            wt_str: the wild-type sequence
            continuation_L: the length of the continuation (L)
            beta: the threshold for activity.
            vocab_size: the size of the vocabulary
            glm: a GLM model to convert f(x) to measured activity
            device: the device to run the model on
            num_samples: budget used to estimate <f,q_e> for each experiment
            ckpt: the checkpoint of the PLM to use
            model_name: the name of the PLM to use (for sanity check)
            tokenizer_file: the file containing the tokenizer for the PLM
            dtype: the data type to use for the PLM
            defensive_alpha: proportion (in [0, 1])
        '''

        super().__init__()
        self.K = 20 # number of amino acids
        self.L = continuation_L
        self.wt_str = wt_str
        assert len(wt_str) > self.L, "The wild-type sequence must be longer than the continuation length L"
        self.beta = beta
        self.glm = glm
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.ckpt = ckpt
        self.model_name = model_name
        self.dtype = dtype
        assert 0.0 <= defensive_alpha <= 1.0, f"defensive_alpha must be in [0, 1], got {defensive_alpha}"
        self.defensive_alpha = defensive_alpha

        # load model
        self.model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=dtype) # type: ignore
        self.model.to(self.device)
        self.model.eval()

        # load tokenizer and encode the wt sequence
        with open(tokenizer_file, 'r') as f:
            self.tokenizer: Tokenizer = Tokenizer.from_str(f.read())
        self.wt_ids = self.encode_sequence(wt_str).to(self.device) # [N + L]
        self.prefix_ids = self.wt_ids[:len(self.wt_ids) - self.L] # [N]
        self.continuation_ids = self.wt_ids[len(self.wt_ids) - self.L:] # [L]

        # shapes
        self.N = len(self.prefix_ids)
        self.max_length = self.N + self.L

        # lazily built by _build_prefix_kv_cache on first _ll call - see that method's docstring
        self._prefix_kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None

        print(f"OraclePLM prefix (wt_str={wt_str!r}): {self.tokenizer.decode(self.prefix_ids.tolist())!r}")

        # tokenizer vocabulary
        vocab = self.tokenizer.get_vocab()
        print(f"Tokenizer vocab size: {len(vocab)}")
        print(f"Tokenizer vocab: {vocab}")
        # bad words
        self.aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.bad_words = [[tok_id] for token, tok_id in vocab.items() if token not in self.aas]
        print(f"Bad token ids: {self.bad_words}")
        print(f"Bad tokens: {[token for token, tok_id in vocab.items() if [tok_id] in self.bad_words]}")

        # dictionary for plm -> aa conversion
        aa_to_q = {aa: i for i, aa in enumerate(self.aas)}
        plm_to_q = torch.full((len(vocab),), -1, dtype=torch.long)
        q_to_plm = torch.full((len(self.aas),), -1, dtype=torch.long)
        for aa in self.aas:
            plm_to_q[vocab[aa]] = aa_to_q[aa]
            q_to_plm[aa_to_q[aa]] = vocab[aa]
        self.plm_to_q = plm_to_q.to(self.device)
        self.q_to_plm = q_to_plm.to(self.device)
        self.x_0 = self.continuation_ids_q # set x_0 to the wild-type continuation in q-space

        # sanity check
        with print_time('sanity cross-entropy'):

            def ce(tokens):
                with torch.no_grad():
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype): # type: ignore
                        target = torch.tensor(self.tokenizer.encode(tokens).ids).to(self.device)
                        logits = self.model(target, labels=target).logits

                        # shift
                        logits = logits[:-1, ...]
                        target = target[1:]

                        return cross_entropy(logits=logits, target=target).item()

            x_uniref90bfd30 = '2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1'
            x_oas = '1EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPWKGLEYVSAISSNGGSTYYANSVKGRFTISRDNSKNTLYLQMGSLRAEDMAVYYCARDESGYSYGWGYYFDYWGQGTLVTVSS2'
            x_bfd90 = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'

            checkpoint_x_ce = {
                'progen2-small': (x_uniref90bfd30, 2.4),
                'progen2-medium': (x_uniref90bfd30, 1.9),
                'progen2-base': (x_uniref90bfd30, 1.9),
                'progen2-large': (x_uniref90bfd30, 1.8),
                'progen2-xlarge': (x_uniref90bfd30, 1.0),
                'progen2-oas': (x_oas, 0.3),
                'progen2-BFD90': (x_bfd90, 1.3),
            }

            ce_eval = ce(checkpoint_x_ce[self.model_name][0])
            ce_target = checkpoint_x_ce[self.model_name][1]

            print(ce_target, ce_eval, abs(ce_eval - ce_target))

            assert abs(ce_eval - ce_target) < 0.1
            
            # precompute the log-likelihood of the wildtype continuation of the prefix
            self.ll_wt = self._ll(self.wt_ids.unsqueeze(0)) # [1]

    @property
    def continuation_ids_q(self):
        return self.plm_to_q[self.continuation_ids]

    @property
    def continuation_ids_plm(self):
        return self.continuation_ids

    def encode_sequence(self, sequence: str, remove_eos: bool = True) -> torch.Tensor:
        sequence_plm = torch.tensor(self.tokenizer.encode(sequence, add_special_tokens=True).ids, dtype=torch.long)
        if remove_eos:
            sequence_plm = sequence_plm[:-1]
        return sequence_plm

    def sequences_plm_to_sequences_q(self, sequences_plm: torch.Tensor) -> torch.Tensor:
        '''
        Convert sequences generated by the PLM (token ids) to sequences in the original space (amino acid ids).

        Args:
            sequences_plm: [num_samples, sequence_length] - token ids of the sequences generated by the PLM
        '''
        sequences_plm = sequences_plm.long()
        sequences_q = self.plm_to_q[sequences_plm]
        if torch.any(sequences_q < 0):
            raise ValueError('sequences_plm contains tokens that are not one of the 20 amino acids')
        return sequences_q
    
    def sequences_q_to_sequences_plm(self, sequences_q: torch.Tensor) -> torch.Tensor:
        '''
        Convert sequences in the original space (amino acid ids) to sequences in the PLM token space.

        Args:
            sequences_q: [num_samples, sequence_length] - amino acid ids of the sequences
        '''
        sequences_q = sequences_q.long()
        if torch.any((sequences_q < 0) | (sequences_q >= len(self.aas))):
            raise ValueError('sequences_q contains amino acid ids outside the q-space alphabet')
        sequences_plm = self.q_to_plm[sequences_q]
        if torch.any(sequences_plm < 0):
            raise ValueError('sequences_q contains amino acid ids that are not in the 20 amino acids')
        return sequences_plm

    def _build_prefix_kv_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        '''
        KV cache for the shared prefix of the wild-type sequence.
        '''
        prefix_minus_last = self.prefix_ids[:-1].unsqueeze(0) # [1, N-1]
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=prefix_minus_last, use_cache=True)
        return out.past_key_values # tuple of (key, value) per layer, each [1, num_heads, N-1, head_dim]

    def _ll(self, sequences_plm: torch.Tensor) -> torch.Tensor:
        '''
        Compute log p(x | x_prefix) for a batch of input sequences.
        Args:
            sequences: token ids of the input sequences, shape [batch_size, sequence_length]
        Returns:
            ll_x: the computed log-likelihoods for the input sequences, shape [batch_size]
        '''
        if sequences_plm.ndim == 1:
            sequences_plm = sequences_plm.unsqueeze(0)
        bs = sequences_plm.shape[0]
        assert torch.equal(sequences_plm[:, :self.N], self.prefix_ids.unsqueeze(0).expand(bs, -1)), (
            "_ll assumes every row shares this oracle's fixed prefix (required for the KV cache to be valid)"
        )
        if self._prefix_kv_cache is None:
            self._prefix_kv_cache = self._build_prefix_kv_cache()

        continuation = sequences_plm[:, self.N:] # [bs, L]
        last_prefix_token = self.prefix_ids[-1].view(1, 1).expand(bs, -1) # [bs, 1]
        live_input = torch.cat([last_prefix_token, continuation], dim=1) # [bs, L+1]
        expanded_kv = tuple(
            (k.expand(bs, -1, -1, -1), v.expand(bs, -1, -1, -1)) for k, v in self._prefix_kv_cache
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=live_input, past_key_values=expanded_kv, use_cache=False).logits # [bs, L+1, vocab_size]

            # shift
            logits = logits[:, :-1, ...] # [batch_size, L, vocab_size], predicts continuation_1..L
            logits = logits.transpose(1, 2) # [batch_size, vocab_size, L]

            ll_x = -F.cross_entropy(input = logits, target = continuation, reduction = 'none') # [batch_size, L]
            ll_x = ll_x.sum(dim=-1) # [batch_size]

        return ll_x

    def _log_f_from_ll(self, log_p: torch.Tensor) -> torch.Tensor:
        '''
        Compute log_f(x) given a precomputed log_p = log p(x|x_prefix), where
        f(x) = ReLU(log [p(x|x_prefix)/p(x_wt|x_prefix)] + beta)

        Args:
            log_p: precomputed log p(x|x_prefix) for a batch of sequences, shape [batch_size]
        Returns:
            log_f: the computed log_f(x) values for the input sequences, shape [batch_size]
        '''
        return torch.log(F.relu(log_p - self.ll_wt + self.beta))

    def _log_f(self, sequences_plm: torch.Tensor) -> torch.Tensor:
        '''
        Compute log_f(x) for a batch of sequences x, where f(x) = ReLU(log [p(x|x_prefix)/p(x_wt|x_prefix)] + beta)

        Args:
            sequences_plm: a batch of sequences to evaluate, shape [batch_size, sequence_length]
        Returns:
            log_f: the computed log_f(x) values for the input sequences, shape [batch_size]
        '''
        return self._log_f_from_ll(self._ll(sequences_plm))

    def f(self,
          sequences_q: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute f(x) for a batch of sequences x, where f(x) = ReLU(log [p(x|x_prefix)/p(x_wt|x_prefix)] + beta)

        Args:
            sequences_q: a batch of sequences to evaluate, shape [batch_size, sequence_length]
        Returns:
            f_values: the computed f(x) values for the input sequences, shape [batch_size]
        '''
        sequences_plm = self.sequences_q_to_sequences_plm(sequences_q)
        if sequences_plm.ndim == 1:
            sequences_plm = sequences_plm.unsqueeze(0)

        prefix_ids = self.prefix_ids.unsqueeze(0).expand(sequences_plm.shape[0], -1)
        sequences_plm = torch.cat([prefix_ids, sequences_plm], dim=-1)

        f_values = F.relu(self._ll(sequences_plm) - self.ll_wt + self.beta)
        return f_values

    def _sample_plm_chunk(self, input_ids: torch.Tensor, count: int) -> torch.Tensor:
        return cast(torch.Tensor, self.model.generate(
            input_ids=input_ids,
            max_length=self.max_length,
            do_sample=True,
            num_return_sequences=count,
            bad_words_ids=self.bad_words, # prevent generation of bad tokens (that are not amino acids) # type: ignore
            pad_token_id=self.tokenizer.token_to_id('<|pad|>'), # type: ignore
        )) # [count, sequence_length]

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Generate a batch of sequences from the PLM given the prefix.

        Args:
            num_samples: the number of sequences to generate
        Returns:
            sequences_q: the generated sequences in AA space, shape [num_samples, sequence_length]
            sequences_plm: the generated sequences in PLM token space, shape [num_samples, sequence_length + prefix_length]
        '''

        self.model.eval()
        with torch.no_grad():
            input_ids = self.prefix_ids.unsqueeze(0).to(self.device) # [1, prefix_length]
            generation_batch_size = self.batch_size
            sequences_plm_chunks: List[torch.Tensor] = []
            sequences_q_chunks: List[torch.Tensor] = []

            for batch_start in range(0, num_samples, generation_batch_size):
                batch_size = min(generation_batch_size, num_samples - batch_start)
                sequences_plm_chunk = self._sample_plm_chunk(input_ids, batch_size)

                # bad_words_ids doesn't always fully prevent a banned token apparently
                max_retries = 5
                for retry in range(max_retries + 1):
                    bad_rows = torch.where((self.plm_to_q[sequences_plm_chunk[:, self.N:].long()] < 0).any(dim=1))[0]
                    if bad_rows.numel() == 0:
                        break
                    if retry == max_retries:
                        raise RuntimeError(
                            f"generate(): {bad_rows.numel()} row(s) still contain a banned token "
                            f"after {max_retries} retries - giving up."
                        )
                    print(
                        f"generate(): bad_words_ids leaked a banned token into {bad_rows.numel()} "
                        f"of {batch_size} row(s) (retry {retry + 1}/{max_retries}); re-sampling those rows.",
                        flush=True,
                    )
                    sequences_plm_chunk[bad_rows] = self._sample_plm_chunk(input_ids, bad_rows.numel())

                sequences_plm_truncated = sequences_plm_chunk[:, self.N:] # remove prefix, keep only continuation
                assert sequences_plm_truncated.shape == (batch_size, self.L)

                sequences_plm_chunks.append(sequences_plm_chunk)
                sequences_q_chunks.append(self.sequences_plm_to_sequences_q(sequences_plm_truncated))

            sequences_plm = torch.cat(sequences_plm_chunks, dim=0)
            sequences_q = torch.cat(sequences_q_chunks, dim=0)

        return sequences_q, sequences_plm # type: ignore

    def log_functional_inner_product(self,
                                     experiment: Experiment,
                                     return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        E = experiment.num_experiments
        S = experiment.num_syntheses
        L = experiment.sequence_length
        K = experiment.vocab_size
        assert K == 20 # sanity check, ProGen2 is trained on 20 amino acids
        assert L == self.L # sanity check, the sequence length of the experiment should match the continuation length of the oracle

        # <f, q_e> = E_{q_e}[f(x)], estimated per experiment by defensive importance sampling
        if self.defensive_alpha <= 0.0:
            n_p, n_q = self.num_samples, 0
        elif self.defensive_alpha >= 1.0:
            n_p, n_q = 0, self.num_samples
        else:
            n_p = max(1, round(self.num_samples * (1.0 - self.defensive_alpha)))
            n_q = max(1, round(self.num_samples * self.defensive_alpha))
        local_alpha = n_q / (n_p + n_q)
        is_interior_mixture = n_p > 0 and n_q > 0
        if is_interior_mixture:
            log_alpha = math.log(local_alpha)
            log_1m_alpha = math.log(1.0 - local_alpha)

        log_term_shared: Optional[torch.Tensor] = None
        log_w_shared: Optional[torch.Tensor] = None
        if n_p > 0:
            log_p_shared_chunks: List[torch.Tensor] = []
            log_f_shared_chunks: List[torch.Tensor] = []
            log_q_shared_chunks: List[torch.Tensor] = [] # each [chunk_size, num_experiments]
            for batch in range(0, n_p, self.batch_size):
                batch_size = min(self.batch_size, n_p - batch)
                samples_q, samples_plm = self.generate(batch_size) # [batch_size, L], [batch_size, N + L]
                log_p = self._ll(samples_plm) # [batch_size]
                log_p_shared_chunks.append(log_p)
                log_f_shared_chunks.append(self._log_f_from_ll(log_p))
                log_q_shared_chunks.append(experiment.log_prob(samples_q)) # [batch_size, num_experiments]
            log_p_shared = torch.cat(log_p_shared_chunks, dim=0) # [n_p]
            log_f_shared = torch.cat(log_f_shared_chunks, dim=0) # [n_p]
            log_q_shared = torch.cat(log_q_shared_chunks, dim=0) # [n_p, num_experiments]

            if is_interior_mixture:
                log_psi_shared = torch.logaddexp(log_1m_alpha + log_p_shared.unsqueeze(1), log_alpha + log_q_shared) # [n_p, num_experiments]
                log_w_shared = log_q_shared - log_psi_shared # [n_p, num_experiments]
            else: 
                # special case, just normal MC
                log_w_shared = log_q_shared - log_p_shared.unsqueeze(1) # [n_p, num_experiments]
            log_term_shared = log_f_shared.unsqueeze(1) + log_w_shared # [n_p, num_experiments]

        log_term_dedicated: Optional[torch.Tensor] = None
        log_w_dedicated: Optional[torch.Tensor] = None
        if n_q > 0:
            samples_q_dedicated = experiment.sample((n_q,)) # [n_q, num_experiments, L]
            log_q_dedicated = experiment.log_prob_own(samples_q_dedicated) # [n_q, num_experiments], log q_e(x) only
            samples_q_dedicated_flat = samples_q_dedicated.reshape(n_q * E, -1) # [n_q * num_experiments, L]
            prefix_ids = self.prefix_ids.unsqueeze(0).expand(samples_q_dedicated_flat.shape[0], -1)
            samples_plm_dedicated_flat = torch.cat(
                [prefix_ids, self.sequences_q_to_sequences_plm(samples_q_dedicated_flat)], dim=-1
            )

            log_p_dedicated_chunks: List[torch.Tensor] = []
            log_f_dedicated_chunks: List[torch.Tensor] = []
            total_dedicated = samples_plm_dedicated_flat.shape[0]
            for batch in range(0, total_dedicated, self.batch_size):
                batch_size = min(self.batch_size, total_dedicated - batch)
                log_p_chunk = self._ll(samples_plm_dedicated_flat[batch:batch + batch_size])
                log_p_dedicated_chunks.append(log_p_chunk)
                log_f_dedicated_chunks.append(self._log_f_from_ll(log_p_chunk))
            log_f_dedicated = torch.cat(log_f_dedicated_chunks, dim=0).reshape(n_q, E)

            if is_interior_mixture:
                log_p_dedicated = torch.cat(log_p_dedicated_chunks, dim=0).reshape(n_q, E)
                log_psi_dedicated = torch.logaddexp(log_1m_alpha + log_p_dedicated, log_alpha + log_q_dedicated) # [n_q, num_experiments]
                log_w_dedicated = log_q_dedicated - log_psi_dedicated # [n_q, num_experiments]
            else:
                # importance weight is exactly 1 (plain Monte Carlo).
                log_w_dedicated = torch.zeros_like(log_q_dedicated)
            log_term_dedicated = log_f_dedicated + log_w_dedicated # [n_q, num_experiments]

        log_terms = torch.cat([t for t in (log_term_shared, log_term_dedicated) if t is not None], dim=0) # [n_p + n_q, num_experiments]
        log_fq = torch.logsumexp(log_terms, dim=0) - math.log(log_terms.shape[0]) # [num_experiments]

        log_ws: Optional[List[torch.Tensor]] = None
        if return_diagnostics:
            w_parts = [w for w in (log_w_shared, log_w_dedicated) if w is not None]
            log_ws = [torch.cat(w_parts, dim=0).detach().cpu()] # [n_p + n_q, num_experiments]

        return log_fq, log_ws

    def rsample_y(self, experiment: Experiment, sigma: float, return_diagnostics: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        '''
        Sample a noisy observation y from the oracle given an experiment.

        Args:
            experiment: the experiment to sample from
            sigma: the noise level
            return_diagnostics: whether to return additional diagnostic information (e.g. importance weights) for debugging or analysis

        Returns:
            y: the noisy observation
        '''
        log_fq, log_ws = self.log_functional_inner_product(experiment, return_diagnostics=return_diagnostics)
        fq = torch.exp(log_fq.to(dtype=torch.float32)) # [num_experiments]
        y = self.glm.rsample(linear_predictor=fq, sigma=sigma) # [num_experiments]

        return y, log_ws if return_diagnostics else None

    def sample_f_as_dist(
        self,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample from p(x), the plm. This is technically not like sampling from f, but it is close to it.
        """
        samples_q, _ = self.generate(num_samples) # [num_samples, L]
        return samples_q


# https://www.uniprot.org/uniprotkb/H8WR56/entry
# rubisco sequence:
rubisco = 'MASSILSSAVVASVNSASPAQASMVAPFTGLKSSAGFPITRKNNVDITTLASNGGKVQCMKVWPPLGLRKFETLSYLPDMSNEQLSKECDYLLRNGWVPCVEFDIGSGFVYRENHRSPGYYDGRYWTMWKLPMFGCTDSSQVIQEIEEAKKEYPDAFIRVIGFDNVRQVQCISFIAYKPPRFYSS'

# https://www.uniprot.org/uniprotkb/Q99ZW2/entry
# PETase sequence:
PETase = 'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS'

# https://www.uniprot.org/uniprotkb/C1DTU5/entry
# SazCA sequence:
SazCA = 'MKKFILSILSLSIVSIAGEHAILQKNAEVHHWSYEGENGPENWAKLNPEYFWCNLKNQSPVDISDNYKVHAKLEKLHINYNKAVNPEIVNNGHTIQVNVLEDFKLNIKGKEYHLKQFHFHAPSEHTVNGKYYPLEMHLVHKDKDGNIAVIGVFFKEGKANPELDKVFKNALKEEGSKVFDGSININALLPPVKNYYTYSGSLTTPPCTEGVLWIVLKQPITASKQQIELFKSIMKHNNNRPTQPINSRYILESN'

# Oracle
# Just a test script #
def main():
    N = 95 # prefix length
    L = 5 # continuation length
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    beta = 10 # beta is the threshold for activity. Only sequences with log-likelihood ratio above negative beta will have non-zero activity.
    wt_str = PETase[:N+L] # use the first N+L amino acids of the EryA1 sequence as the wild-type sequence for the oracle
    wt_str = '1' + wt_str # add beginning token (ProGen2 specific)

    oracle = OraclePLM(
        wt_str=wt_str,
        continuation_L=L,
        beta=beta,
        glm=NormalGLM(),
        device=device,
        num_samples=2048,
        batch_size=128,
        ckpt='progen2-base',
        model_name='progen2-base',
        tokenizer_file='progen/progen2/tokenizer.json',
        dtype=torch.float16,
    )

    experiment = Experiment(
        vocab_size=20,
        sequence_length=L,
        num_experiments=5,
        num_syntheses=10,
    ).to(device)

    experiment_wt = thompson_samples_to_experiment(
        thompson_samples = oracle.sequences_plm_to_sequences_q(oracle.continuation_ids.unsqueeze(0)), # [1, L]
        sequence_length = L,
        vocab_size = 20,
        num_experiments = 1,
    ).to(device)


    log_fq, log_ws = oracle.log_functional_inner_product(experiment)
    print(f"log <f, q>: {log_fq}")
    log_fq_wt, log_ws = oracle.log_functional_inner_product(experiment_wt)
    print(f"log <f, q_wt>: {log_fq_wt}") # experiment which is a delta mixture on the wt
    log_f_wt = oracle._log_f(oracle.wt_ids.unsqueeze(0))
    print(f"log f(wt): {log_f_wt}")


if __name__ == "__main__":
    main()