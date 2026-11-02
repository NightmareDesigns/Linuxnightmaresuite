# === FORCE CUDA TO SEE BOTH GPUs ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
_NIGHTMARE_DEBUG = os.environ.get("NIGHTMARE_DEBUG", "0") == "1"

"""
Token Sampler — From Scratch
Converts raw logits into token selections using various sampling strategies.

Strategies:
- Greedy (argmax) — always pick the highest probability token
- Temperature — scale probabilities to control randomness
- Top-K — only sample from the K most likely tokens
- Top-P (nucleus) — sample from smallest set of tokens that sum to P probability
- Repetition penalty — reduce probability of recently generated tokens
"""

import math
import nightmare_np as np
from typing import Optional
from tensor_ops import softmax


class Sampler:
    """
    Token sampling with configurable strategies.
    
    Usage:
        sampler = Sampler(temperature=0.7, top_k=40, top_p=0.9)
        next_token = sampler.sample(logits)

    Strategies: greedy, temperature, top-k, top-p (nucleus), min-P, top-A,
    tail-free, repetition / frequency / presence penalties, DRY (no-repeat
    n-gram), Mirostat v1/v2 (target-surprise feedback) and XTC (test-time
    exploration).
    """
    
    def __init__(
        self,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_a: float = 0.0,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.05,
        presence_penalty: float = 0.05,
        dry_multiplier: float = 0.8,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_range: int = 0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        mirostat_m: int = 100,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.1,
        xtc_min_tokens: int = 2,
        seed: Optional[int] = None
    ):
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.top_a = float(top_a)
        self.repetition_penalty = float(repetition_penalty)
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)
        self.dry_multiplier = float(dry_multiplier)
        self.dry_base = float(dry_base)
        self.dry_allowed_length = int(dry_allowed_length)
        self.dry_range = int(dry_range)
        self.mirostat = int(mirostat)
        self.mirostat_tau = float(mirostat_tau)
        self.mirostat_eta = float(mirostat_eta)
        self.mirostat_m = int(mirostat_m)
        self.xtc_probability = float(xtc_probability)
        self.xtc_threshold = float(xtc_threshold)
        self.xtc_min_tokens = int(xtc_min_tokens)
        
        # Random state
        self.rng = np.random.default_rng(seed)
        
        # Track generated tokens for repetition and DRY penalty
        self.generated_tokens = []
        self.token_counts = {}
        
        # Mirostat controller state (mu adapts toward the target surprise)
        self._mirostat_mu = 2.0 * self.mirostat_tau
        self._n_vocab_est = 0

    def params(self) -> dict:
        """Return all active sampling parameters as a dict (for API/GUI)."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "top_a": self.top_a,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "dry_multiplier": self.dry_multiplier,
            "dry_base": self.dry_base,
            "dry_allowed_length": self.dry_allowed_length,
            "dry_range": self.dry_range,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_m": self.mirostat_m,
            "xtc_probability": self.xtc_probability,
            "xtc_threshold": self.xtc_threshold,
            "xtc_min_tokens": self.xtc_min_tokens,
        }
    
    def sample(self, logits: np.ndarray, mask: Optional[set] = None) -> int:
        """
        Sample the next token from logits.

        Optional `mask` (set/frozenset of allowed token ids, produced by
        constrained decoding) forces every token outside the mask off before
        the normal pipeline runs.
        
        Pipeline:
        1. Apply DRY (Don't Repeat Yourself) n-gram penalty
        2. Apply token repetition penalty
        3. Apply frequency and presence penalties
        4. Apply temperature scaling
        5. Apply Top-K filtering
        6. Apply Top-A filtering
        7. Apply Min-P filtering
        8. Apply Top-P (nucleus) filtering
        9. Sample from the resulting distribution
        """
        # Handle ctypes arrays from GPU engine
        if not isinstance(logits, np.ndarray):
            logits = np.ctypeslib.as_array(logits).copy().astype(np.float32)
        else:
            logits = logits.copy().astype(np.float32)

        # Constrained decoding: respect the allowed-token mask first
        if mask is not None and len(mask) > 0:
            lo = np.zeros(logits.shape[0], dtype=bool)
            idx = np.fromiter((int(i) for i in mask if 0 <= int(i) < logits.shape[0]),
                              dtype=np.int64)
            lo[idx] = True
            logits[~lo] = -np.inf
        
        # === DEBUG: Check for degenerate logits that indicate upstream corruption ===
        if _NIGHTMARE_DEBUG:
            if np.any(np.isnan(logits)):
                print(f"[Sampler DEBUG] WARNING: NaN in logits! Count: {np.sum(np.isnan(logits))}")
            if np.any(np.isinf(logits)):
                print(f"[Sampler DEBUG] WARNING: Inf in logits! Count: {np.sum(np.isinf(logits))}")
            print(f"[Sampler DEBUG] Logits: min={logits.min():.2f} max={logits.max():.2f} mean={logits.mean():.2f} std={logits.std():.2f}")
        
        # 1. DRY (Don't Repeat Yourself) n-gram repetition penalty
        if self.dry_multiplier > 0.0 and len(self.generated_tokens) > self.dry_allowed_length:
            logits = self._apply_dry_penalty(logits)

        # 2. Token repetition penalty
        if self.repetition_penalty != 1.0 and self.generated_tokens:
            logits = self._apply_repetition_penalty(logits)
        
        # 3. Frequency and presence penalties
        if (self.frequency_penalty > 0 or self.presence_penalty > 0) and self.token_counts:
            logits = self._apply_frequency_presence_penalty(logits)
        
        # Greedy mode: just argmax (mirostat overrides greedy behaviour)
        if self.temperature == 0.0 and self.mirostat == 0:
            token = int(np.argmax(logits))
            self._record_token(token)
            return token
        
        # 4. Temperature scaling
        if self.mirostat:
            # Mirostat v1/v2 replaces the normal top-k/top-p pipeline.
            return self._sample_mirostat(logits, self.mirostat)
        
        logits = logits / self.temperature
        
        # 5. Top-K filtering
        if self.top_k > 0:
            logits = self._apply_top_k(logits)

        # 6. Top-A filtering (p_i >= top_a * p_max^2)
        if self.top_a > 0.0:
            logits = self._apply_top_a(logits)

        # 7. Min-P filtering (p_i >= min_p * p_max)
        if self.min_p > 0.0:
            logits = self._apply_min_p(logits)
        
        # 8. Top-P (nucleus) filtering
        if self.top_p < 1.0:
            logits = self._apply_top_p(logits)
        
        # 8b. XTC: with probability xtc_probability, drop the over-confident
        # tokens so the model is forced to explore the distribution tail.
        if self.xtc_probability > 0.0 and self.rng.random() < self.xtc_probability:
            logits = self._apply_xtc(logits)
        
        # 9. Fast subset conversion to probabilities & sampling
        finite_mask = logits > -np.inf
        finite_idx = np.where(finite_mask)[0]
        
        if len(finite_idx) == 0:
            token = int(np.argmax(logits))
        elif len(finite_idx) == 1:
            token = int(finite_idx[0])
        else:
            sub_logits = logits[finite_idx]
            sub_probs = softmax(sub_logits)
            sub_probs = np.nan_to_num(sub_probs, nan=0.0, posinf=0.0, neginf=0.0)
            prob_sum = float(sub_probs.sum())
            
            if prob_sum <= 0.0:
                token = int(finite_idx[np.argmax(sub_logits)])
            else:
                sub_probs = sub_probs / prob_sum  # Re-normalize
                token = int(self.rng.choice(finite_idx, p=sub_probs))
        
        self._record_token(token)
        return token

    def _apply_dry_penalty(self, logits: np.ndarray) -> np.ndarray:
        """
        DRY (Don't Repeat Yourself) n-gram repetition penalty.
        Finds matching token suffixes in past generation and penalizes matching continuations.
        Penalty = multiplier * (base ** (match_length - allowed_length))
        """
        history = self.generated_tokens
        hist_len = len(history)
        if self.dry_range > 0:
            history = history[-self.dry_range:]
            hist_len = len(history)

        match_lengths = {}
        for length in range(self.dry_allowed_length, min(64, hist_len)):
            suffix = history[-length:]
            # Search backwards through earlier history for this suffix
            for i in range(hist_len - length - 1, -1, -1):
                if history[i:i + length] == suffix:
                    next_tok = history[i + length]
                    match_lengths[next_tok] = max(match_lengths.get(next_tok, 0), length)

        for tok, match_len in match_lengths.items():
            if tok < len(logits) and match_len >= self.dry_allowed_length:
                penalty = self.dry_multiplier * (self.dry_base ** (match_len - self.dry_allowed_length))
                logits[tok] -= penalty

        return logits

    def _apply_min_p(self, logits: np.ndarray) -> np.ndarray:
        """
        Min-P filtering: Scale threshold relative to the highest probability token.
        Preserves rich vocabulary while strictly eliminating low-probability tail noise.
        """
        valid_mask = logits > -np.inf
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return logits

        sub_logits = logits[valid_indices]
        probs = softmax(sub_logits)
        max_prob = float(np.max(probs))
        threshold = max_prob * self.min_p

        low_prob_mask = probs < threshold
        cutoff_indices = valid_indices[low_prob_mask]
        logits[cutoff_indices] = -np.inf
        return logits

    def _apply_top_a(self, logits: np.ndarray) -> np.ndarray:
        """
        Top-A filtering: Threshold = top_a * (max_prob ** 2).
        """
        valid_mask = logits > -np.inf
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return logits

        sub_logits = logits[valid_indices]
        probs = softmax(sub_logits)
        max_prob = float(np.max(probs))
        threshold = self.top_a * (max_prob ** 2)

        low_prob_mask = probs < threshold
        cutoff_indices = valid_indices[low_prob_mask]
        logits[cutoff_indices] = -np.inf
        return logits
    
    def _apply_repetition_penalty(self, logits: np.ndarray) -> np.ndarray:
        """
        Repetition penalty: reduce logits of previously generated tokens.
        
        For each previously generated token:
          - If logit > 0: divide by penalty
          - If logit < 0: multiply by penalty
        """
        lookback = min(256, len(self.generated_tokens))
        unique_tokens = set(self.generated_tokens[-lookback:])  # Lookback over recent token history
        
        for token_id in unique_tokens:
            if token_id < len(logits):
                if logits[token_id] > 0:
                    logits[token_id] /= self.repetition_penalty
                else:
                    logits[token_id] *= self.repetition_penalty
        
        return logits
    
    def _apply_frequency_presence_penalty(self, logits: np.ndarray) -> np.ndarray:
        """
        Frequency penalty: penalize proportional to how often a token appeared.
        Presence penalty: flat penalty for any token that appeared at all.
        """
        for token_id, count in self.token_counts.items():
            if token_id < len(logits):
                logits[token_id] -= (
                    self.frequency_penalty * count + 
                    self.presence_penalty * (1 if count > 0 else 0)
                )
        return logits
    
    def _apply_top_k(self, logits: np.ndarray) -> np.ndarray:
        """
        Top-K: Keep only the K highest logits, set the rest to -inf.
        """
        if self.top_k >= len(logits):
            return logits
        
        # Find the K-th largest value
        top_k_indices = np.argpartition(logits, -self.top_k)[-self.top_k:]
        threshold = logits[top_k_indices].min()
        
        # Mask everything below threshold
        logits[logits < threshold] = -np.inf
        return logits
    
    def _apply_top_p(self, logits: np.ndarray) -> np.ndarray:
        """
        Top-P (nucleus sampling): Fast candidate sort and cutoff.
        """
        valid_mask = logits > -np.inf
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return logits
        
        # If vocabulary is huge and top-k was not applied, partition top 256
        if len(valid_indices) > 256:
            top_k_sub = np.argpartition(logits[valid_indices], -256)[-256:]
            valid_indices = valid_indices[top_k_sub]
            
        sub_logits = logits[valid_indices]
        sort_order = np.argsort(sub_logits)[::-1]
        sorted_indices = valid_indices[sort_order]
        sorted_logits = sub_logits[sort_order]
        
        # Softmax on top subset
        probs = softmax(sorted_logits)
        cumulative_probs = np.cumsum(probs)
        
        # Cutoff index
        cutoff_idx = int(np.searchsorted(cumulative_probs, self.top_p)) + 1
        
        # Remove tokens beyond the cutoff
        removed_indices = sorted_indices[cutoff_idx:]
        logits[removed_indices] = -np.inf
        
        return logits
    
    def _record_token(self, token_id: int):
        """Track a generated token."""
        self.generated_tokens.append(token_id)
        self.token_counts[token_id] = self.token_counts.get(token_id, 0) + 1
    
    def _sample_mirostat(self, logits: np.ndarray, mode: int) -> int:
        """
        Mirostat-1 / Mirostat-2 sampling (target-surprise feedback controller).
        
        The controller maintains mu (the allowed "surprise budget") which is
        tightened or relaxed after every sample based on how far the observed
        surprise (-log2 p) is from the target tau:
            mu <- mu - eta * (observed_surprise - tau)
        
        v1: estimates the effective exponent s_hat of the token distribution
            (Zipf-style tail fit over the top m candidates) and derives a
            top-k whose entropy is close to the current mu budget.
        v2: directly truncates the ranked distribution at every token whose
            surprise exceeds mu, then renormalises and samples; mu is free to
            float above 2*tau, unlike v1.
        """
        probs = softmax(logits)
        order = np.argsort(probs)[::-1]
        n = len(order)
        self._n_vocab_est = n

        if mode == 1:
            # Estimate s_hat from the slope of ln(p) vs ln(rank) over the
            # highest-ranked candidates (Zipfian tail fit).
            n_pairs = max(0, min(self.mirostat_m - 1, n - 1))
            sum_ti_bi = 0.0
            sum_ti_sq = 0.0
            for i in range(n_pairs):
                p0 = float(probs[order[i]])
                p1 = float(probs[order[i + 1]])
                if p0 <= 0.0 or p1 <= 0.0:
                    break
                t_i = math.log((i + 2) / (i + 1))
                b_i = math.log(p0 / p1)
                sum_ti_bi += t_i * b_i
                sum_ti_sq += t_i * t_i
            s_hat = sum_ti_bi / sum_ti_sq if sum_ti_sq > 0.0 else 1.0

            eps = s_hat - 1.0
            if abs(eps) < 1e-9:
                k = n
            else:
                denom = 1.0 - (n ** (-eps))
                if abs(denom) < 1e-12:
                    k = n
                else:
                    k = ((eps * (2.0 ** self._mirostat_mu)) / denom) ** (1.0 / s_hat)
            k = min(max(int(k), 1), n)
            cand = order[:k]
            sub = probs[cand]
            sub = sub / sub.sum()
            idx = int(self.rng.choice(cand, p=sub))
        else:
            # v2: keep every ranked token whose surprise stays within mu.
            keep = []
            for tid in order:
                surprise = -math.log2(float(probs[tid]))
                if surprise <= self._mirostat_mu:
                    keep.append(int(tid))
                else:
                    break
            if not keep:
                keep = [int(order[0])]
            sub = probs[keep]
            sub = sub / sub.sum()
            idx = int(self.rng.choice(keep, p=sub))

        observed_surprise = -math.log2(float(probs[idx]))
        error = observed_surprise - self.mirostat_tau
        self._mirostat_mu = self._mirostat_mu - self.mirostat_eta * error

        self._record_token(idx)
        return idx

    def _apply_xtc(self, logits: np.ndarray) -> np.ndarray:
        """
        XTC (test-time exploration): with the configured probability, silence
        the overly-confident candidates.
        
        - If the single most confident token exceeds the threshold, drop ONLY
          that token so the model must pick a genuine alternative.
        - Otherwise, drop every token whose probability exceeds the threshold,
          sampling instead from the flatter tail.
        Always leave at least xtc_min_tokens live candidates.
        """
        valid_idx = np.where(logits > -np.inf)[0]
        if len(valid_idx) < 2 or len(valid_idx) < self.xtc_min_tokens:
            return logits
        valid = logits[valid_idx]
        probs = softmax(valid)
        order = np.argsort(probs)[::-1]
        ordered_ids = valid_idx[order]

        if probs[order[0]] > self.xtc_threshold:
            drop = [ordered_ids[0]]
        else:
            drop = []
            for pos in range(1, len(order)):
                if probs[order[pos]] > self.xtc_threshold:
                    drop.append(ordered_ids[pos])
                else:
                    break
        if (len(valid_idx) - len(drop)) < self.xtc_min_tokens:
            return logits
        if drop:
            logits[drop] = -np.inf
        return logits

    def reset(self):
        """Reset token history (for new generation)."""
        self.generated_tokens = []
        self.token_counts = {}
        self._mirostat_mu = 2.0 * self.mirostat_tau
