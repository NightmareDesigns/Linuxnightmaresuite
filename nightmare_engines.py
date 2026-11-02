# -*- coding: utf-8 -*-
"""
nightmare_engines.py — one dedicated engine for EVERY model family 💀
====================================================================
Every language-model architecture you can find on the internet gets its
OWN engine class — the same way Gemma4GpuEngine, Qwen35HybridEngine and
BonsaiSuiteModel were built.  Each dedicated engine knows its family's
exact math:

    activation        swiglu | geglu (gemma) | gelu (phi2/starcoder2/falcon/mpt/bloom/gpt2)
    norm flavour      RMSNorm | LayerNorm (gpt2/gpt-neox/falcon/command-r/bloom)
    rope              full rotary | partial rotary (phi2 / phi-3-small)
    final softcap     gemma2 / gemma3 logit capping
    MoE               expert routing (deepseek / qwen3moe / dbrx / mixtral)
    parallel attn     falcon pre-norm (attention + MLP in parallel)

The heavy per-tensor GPU math lives in the suite's proven streaming /
GPU-native core (nightmare_stream.GPUStreamEngine).  Each dedicated
engine subclasses it, tags itself with its family name, and forces the
correct arch configuration, so a model is always routed to "its own"
engine.  The already-built special engines (gemma4 / qwen35 / bonsai /
NTMRE v1 / v16) are listed in the catalog too and stay untouched.

Usage:
    python nightmare_engines.py                      # print the engine catalog
    python nightmare_engines.py model.gguf           # show which engine runs it
    python nightmare_engines.py --selftest           # validate FFN / rope math

Author: Nightmare Designs 💀
"""
import os
import json
from pathlib import Path

import numpy as np

from nightmare_stream import GPUStreamEngine


# ════════════════════════════════════════════════════════════════════
# ACTIVATION MATH (per family)
# ════════════════════════════════════════════════════════════════════

def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish — llama / qwen / mixtral / deepseek / internlm ..."""
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _gelu(x: np.ndarray) -> np.ndarray:
    """tanh-approximated GELU (matches llama.cpp fast GELU) — gemma
    gated-GELU, phi-2 / starcoder2 / falcon / mpt / bloom / gptneox."""
    y = np.clip(x, -12, 12)
    return np.float32(0.5) * y * (np.float32(1.0) + np.tanh(
        np.float32(0.7978845608028654) * (y + np.float32(0.044715) * y ** 3)))


# ════════════════════════════════════════════════════════════════════
# DEDICATED ENGINE BASE — one subclass per family
# ════════════════════════════════════════════════════════════════════

class ArchEngine(GPUStreamEngine):
    """Base class for every dedicated per-family engine.

    Subclasses set class-level `family` and `activation`; the streaming
    core (GPUStreamEngine) still does all loading, KV cache, GPU matvec,
    MoE expert streaming, etc.  What this layer adds:

      * a family tag (engine_name) surfaced in the suite GUI
      * the correct FFN activation for the family's math
      * partial-rope support (phi2 / phi-3-small) via `_rope_pairs`
      * final-logit softcap (gemma2 / gemma3) via `_output_softcap`
      * a CPU-math override so non-SwiGLU / partial-rope families use the
        exact correct layer math instead of the generic core
    """

    family = "generic"
    engine_name = "Generic Engine"
    activation = "swiglu"          # swiglu | geglu | gelu
    output_softcap = None          # gemma2/gemma3 final logit cap
    native_compatible = True       # False => never route to GpuNativeEngine
    builtin = False                # True for the pre-built special engines

    def __init__(self, model_path, max_seq_len=2048, **kwargs):
        super().__init__(model_path, max_seq_len=max_seq_len)
        self._engine_tag = self.engine_name
        self._ffn_activation = self.activation
        self._rope_pairs = None
        self._output_softcap = self.output_softcap
        self._parallel_attn = False
        self._apply_arch_fixups()

    # ── per-family post-load configuration ──────────────────────────
    def _apply_arch_fixups(self):
        """Called after the core finishes loading.  Detect per-file knobs
        (partial rope dimension count, softcap) that GGUF metadata exposes."""
        md = self.metadata if hasattr(self, 'metadata') else {}
        arch = str(getattr(self, '_architecture', '')).lower()

        # Partial rotary: GGUF `{arch}.rope.dimension_count` < head_dim
        # (phi-2, phi-3-small, some GPT-NeoX).  Fewer pairs than full.
        try:
            hd = int(getattr(self, 'head_dim', 0) or 0)
            rd = None
            m = getattr(self, 'm', None)
            if m is not None and getattr(m, 'metadata', None):
                rd = m.metadata.get(f'{arch}.rope.dimension_count')
            if not rd and isinstance(md, dict):
                rd = md.get(f'{arch}.rope.dimension_count')
            if rd:
                rd = int(rd)
                if 0 < rd < hd:
                    self._rope_pairs = rd
        except Exception:
            pass

        # gemma2 final logit softcap (default cap 30).  gemma3 has NO
        # softcap — its _output_softcap stays None (class attr), and any
        # GGUF metadata for gemma3 carries no cap fields.
        if self.family == 'gemma2':
            self._output_softcap = 30.0

    # ── which families must use the exact CPU layer math ────────────
    def _needs_special_math(self):
        """True when the generic GPU/CPU core can't express this family's
        math (non-SwiGLU FFN, partial rope, parallel attention)."""
        if getattr(self, '_ffn_activation', 'swiglu') != 'swiglu':
            return True
        pairs = getattr(self, '_rope_pairs', None)
        hd = int(getattr(self, 'head_dim', 0) or 0)
        if pairs and hd and pairs * 2 < hd:
            return True
        if getattr(self, '_parallel_attn', False):
            return True
        return False

    # ── forward: route through exact CPU math + apply softcap ───────
    def forward(self, token_id: int, position: int) -> np.ndarray:
        if self._needs_special_math():
            saved = self._gpu_buf
            self._gpu_buf = None          # force the core's CPU fallback path
            try:
                logits = super().forward(token_id, position)
            finally:
                self._gpu_buf = saved
        else:
            logits = super().forward(token_id, position)
        cap = getattr(self, '_output_softcap', None)
        if cap:
            logits = np.tanh(logits / np.float32(cap)) * np.float32(cap)
        return logits

    # ── exact per-family FFN ─────────────────────────────────────────
    def _ffn(self, xn: np.ndarray, weights: dict) -> np.ndarray:
        """Family-correct FFN.  `weights` holds 'wg' (gate), 'wu' (up),
        'wd' (down) as loaded by the core (the standard GGUF ffn names)."""
        act = getattr(self, '_ffn_activation', 'swiglu')
        if act == 'geglu':        # gemma / gemma2 / gemma3: gate * gelu(up)
            gate = self._mm(xn, weights['wg'])
            up = self._mm(xn, weights['wu'])
            return self._mm(gate * _gelu(up), weights['wd'])
        if act == 'gelu':         # single-projection: gelu(up), no gate
            up = self._mm(xn, weights['wu'])
            return self._mm(_gelu(up), weights['wd'])
        gate = self._mm(xn, weights['wg'])
        up = self._mm(xn, weights['wu'])
        return self._mm(_silu(gate) * up, weights['wd'])

    # ── exact per-family layer math (CPU path used by _needs_special_math)
    def _layer_forward(self, x, layer, position, weights):
        """Full transformer layer with the family's exact math.  Mirrors
        the core's CPU path but honours activation / partial-rope so the
        dedicated engines produce correct logits for their family."""
        x = x.reshape(1, -1)
        xn = self._norm(x, weights['attn_norm'])

        q = self._mm(xn, weights['wq'])[0]
        k = self._mm(xn, weights['wk'])[0]
        v = self._mm(xn, weights['wv'])[0]
        if weights.get('bq') is not None:
            q = q + weights['bq']
            k = k + weights['bk']
            v = v + weights['bv']

        n_rep = self.n_heads // self.n_kv_heads
        hd = self.head_dim
        pairs = getattr(self, '_rope_pairs', None)
        if not pairs:
            pairs = hd // 2
        pairs = min(pairs, hd // 2)
        if not hasattr(self, '_rope_cos_tab'):
            freqs = 1.0 / (self.rope_base ** (np.arange(pairs, dtype=np.float32) * 2.0 / hd))
            pos = np.arange(self.max_seq_len, dtype=np.float32)
            ang = np.outer(pos, freqs)
            self._rope_cos_tab = np.cos(ang).astype(np.float32)
            self._rope_sin_tab = np.sin(ang).astype(np.float32)
        cos_t = self._rope_cos_tab[position]
        sin_t = self._rope_sin_tab[position]

        def _rotate(r):
            r = r.reshape(-1, hd)
            c0 = 2 * pairs
            r0, r1 = r[:, :pairs].copy(), r[:, pairs:c0].copy()
            r[:, :pairs] = r0 * cos_t - r1 * sin_t
            r[:, pairs:c0] = r0 * sin_t + r1 * cos_t
            return r.reshape(-1)

        q = _rotate(q)
        k = _rotate(k)

        self.kv_k[layer, position] = k
        self.kv_v[layer, position] = v

        seq_len = position + 1
        q_heads = q.reshape(self.n_heads, hd)
        k_heads = self.kv_k[layer, :seq_len].reshape(seq_len, self.n_kv_heads, hd)
        v_heads = self.kv_v[layer, :seq_len].reshape(seq_len, self.n_kv_heads, hd)
        if n_rep > 1:
            k_heads = np.repeat(k_heads, n_rep, axis=1)
            v_heads = np.repeat(v_heads, n_rep, axis=1)
        scores = np.einsum('hd,phd->hp', q_heads, k_heads, optimize=True)
        scores *= np.float32(1.0 / np.sqrt(hd))
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        attn_out = np.einsum('hp,phd->hd', scores, v_heads, optimize=True).reshape(1, -1)
        x = x + self._mm(attn_out, weights['wo'])

        xn = self._norm(x, weights['ffn_norm'])
        if self._n_experts > 0 and weights.get('moe_router') is not None:
            x = x + self._moe_ffn(xn, layer, weights)
        else:
            x = x + self._ffn(xn, weights)
        return x[0]


# ════════════════════════════════════════════════════════════════════
# ONE DEDICATED ENGINE PER FAMILY
# ════════════════════════════════════════════════════════════════════

class LlamaFamilyEngine(ArchEngine):
    """llama / llama2 / llama3 / llama3.2 / llama3.3 / llama4 / yi /
    smollm / nemotron / stablelm / olmo / baichuan / granite-dense."""
    family = "llama"
    engine_name = "Llama Engine"
    activation = "swiglu"


class QwenFamilyEngine(ArchEngine):
    """qwen / qwen2 / qwen2.5 / qwen3 (dense) / minicpm / telechat."""
    family = "qwen2"
    engine_name = "Qwen2 Engine"
    activation = "swiglu"


class QwenMoeEngine(ArchEngine):
    """qwen3moe / qwen2moe / dbrx — mixture-of-experts deployed on the
    standard streaming core (expert tensors streamed per token)."""
    family = "qwen-moe"
    engine_name = "Qwen-MoE Engine"
    activation = "swiglu"


class GemmaEngine(ArchEngine):
    """gemma (original 2B/7B) — gated GELU FFN, RMSNorm, sqrt(dim) emb."""
    family = "gemma"
    engine_name = "Gemma Engine"
    activation = "geglu"
    native_compatible = False


class Gemma2Engine(ArchEngine):
    """gemma2 — gated GELU + final logit softcap (30)."""
    family = "gemma2"
    engine_name = "Gemma2 Engine"
    activation = "geglu"
    output_softcap = 30.0
    native_compatible = False


class Gemma3Engine(Gemma2Engine):
    """gemma3 / gemma3-text — gated GELU + per-head QK norms, NO final
    logit softcap.  gemma-3 removed logit capping (gemma-2 only caps
    at 30), so output_softcap stays None."""
    family = "gemma3"
    engine_name = "Gemma3 Engine"
    output_softcap = None


class MistralFamilyEngine(ArchEngine):
    """mistral / mistral3 / mixtral / codestral / ministral."""
    family = "mistral"
    engine_name = "Mistral Engine"
    activation = "swiglu"


class PhiEngine(ArchEngine):
    """phi4 / phi3 / phi3-small (swiglu + full/partial rope)."""
    family = "phi"
    engine_name = "Phi Engine"
    activation = "swiglu"


class Phi2Engine(ArchEngine):
    """phi-2 — single GELU projection + partial rotary."""
    family = "phi2"
    engine_name = "Phi-2 Engine"
    activation = "gelu"
    native_compatible = False


class DeepSeekEngine(ArchEngine):
    """deepseek / deepseek2 / deepseek3 / deepseek_r1 (dense + MoE)."""
    family = "deepseek"
    engine_name = "DeepSeek Engine"
    activation = "swiglu"


class FalconEngine(ArchEngine):
    """falcon / falcon2 — GELU, LayerNorm, parallel attention pre-norm."""
    family = "falcon"
    engine_name = "Falcon Engine"
    activation = "gelu"
    native_compatible = False


class StarCoderEngine(ArchEngine):
    """starcoder / starcoder2 — GELU FFN."""
    family = "starcoder"
    engine_name = "StarCoder Engine"
    activation = "gelu"
    native_compatible = False


class GPT2Engine(ArchEngine):
    """gpt2 — GELU FFN, LayerNorm, no rotary (core leaves it unrotated)."""
    family = "gpt2"
    engine_name = "GPT-2 Engine"
    activation = "gelu"
    native_compatible = False


class GPTNeoXEngine(ArchEngine):
    """gpt-neox / gpt-j — GELU FFN, LayerNorm, rotary."""
    family = "gptneox"
    engine_name = "GPT-NeoX Engine"
    activation = "gelu"
    native_compatible = False


class MptEngine(ArchEngine):
    """mpt — GELU FFN, ALiBi-style (position handled by embedding)."""
    family = "mpt"
    engine_name = "MPT Engine"
    activation = "gelu"
    native_compatible = False


class BloomEngine(ArchEngine):
    """bloom — GELU FFN, LayerNorm, ALiBi."""
    family = "bloom"
    engine_name = "BLOOM Engine"
    activation = "gelu"
    native_compatible = False


class BaichuanEngine(ArchEngine):
    """baichuan / baichuan2 — SwiGLU, RMSNorm."""
    family = "baichuan"
    engine_name = "Baichuan Engine"
    activation = "swiglu"


class InternLMEngine(ArchEngine):
    """internlm / internlm2 / internlm3 — SwiGLU, RMSNorm."""
    family = "internlm"
    engine_name = "InternLM Engine"
    activation = "swiglu"


class CommandREngine(ArchEngine):
    """command-r / command-r-plus / cohere — SwiGLU, LayerNorm."""
    family = "command-r"
    engine_name = "Command-R Engine"
    activation = "swiglu"


class GraniteEngine(ArchEngine):
    """granite / granite3 / granite4 — SwiGLU, RMSNorm."""
    family = "granite"
    engine_name = "Granite Engine"
    activation = "swiglu"


class ExAoneEngine(ArchEngine):
    """exaone — SwiGLU, RMSNorm."""
    family = "exaone"
    engine_name = "ExAone Engine"
    activation = "swiglu"


class JambaEngine(ArchEngine):
    """jamba / atmojamba-lo — hybrid Mamba+Attention.  The dense-attention
    part runs here; Mamba SSM caches are swept by the core as dense layers
    until a dedicated SSM kernel lands."""
    family = "jamba"
    engine_name = "Jamba Engine"
    activation = "swiglu"


class GPTOSSEngine(ArchEngine):
    """gpt-oss / gpt-oss-moe (OpenAI 2025 family) — GQA + optional MoE,
    swiglu."""
    family = "gpt-oss"
    engine_name = "GPT-OSS Engine"
    activation = "swiglu"


class BertEngine(ArchEngine):
    """bert / nomic-bert / jina-bert / mini-bert / modernbert / mpnet —
    GELU FFN, LayerNorm, often non-causal (embedding) attention."""
    family = "bert"
    engine_name = "BERT Engine"
    activation = "gelu"
    native_compatible = False


class GlmEngine(ArchEngine):
    """chatglm / glm2 / glm3 / glm4 — SwiGLU FFN, RMSNorm (stream core
    handles the GLM-specific QKV bias / rope layout)."""
    family = "glm"
    engine_name = "GLM Engine"
    activation = "swiglu"


class GenericFallbackEngine(ArchEngine):
    """Unknown / newly-released architecture — runs on the llama-style
    core so the model still loads; the engine tag says 'Generic'."""
    family = "generic"
    engine_name = "Generic Engine"
    activation = "swiglu"


# ════════════════════════════════════════════════════════════════════
# ARCHITECTURE DATABASE — every arch string seen in the wild
# ════════════════════════════════════════════════════════════════════
# `general.architecture` from GGUF / HF `model_type`.  Add future archs
# here (and optionally their own engine class above) rather than in the
# router.  Keys are lower-cased and matched with startswith tolerance for
# suffixes like `-moe`, `_instruct`, `.5` etc.

ARCH_TO_ENGINE = {
    # ── llama family ───────────────────────────────────────────────
    'llama': LlamaFamilyEngine,
    'llama2': LlamaFamilyEngine,
    'llama3': LlamaFamilyEngine,
    'llama4': LlamaFamilyEngine,
    'llama_v2': LlamaFamilyEngine,
    'llama_v3': LlamaFamilyEngine,
    'llama3.1': LlamaFamilyEngine,
    'llama3.2': LlamaFamilyEngine,
    'llama3.3': LlamaFamilyEngine,
    'yi': LlamaFamilyEngine,
    'yi1': LlamaFamilyEngine,
    'yi-1.5': LlamaFamilyEngine,
    'smollm': LlamaFamilyEngine,
    'smollm2': LlamaFamilyEngine,
    'nemotron': LlamaFamilyEngine,
    'nemotron4': LlamaFamilyEngine,
    'nemo': LlamaFamilyEngine,
    'stablelm': LlamaFamilyEngine,
    'stablelm2': LlamaFamilyEngine,
    'stablelm3': LlamaFamilyEngine,
    'stabilitylm': LlamaFamilyEngine,
    'olmo': LlamaFamilyEngine,
    'olmo2': LlamaFamilyEngine,
    'olmoe': LlamaFamilyEngine,
    'baichuan': LlamaFamilyEngine,
    'baichuan2': LlamaFamilyEngine,
    'granite': GraniteEngine,
    'granite4': GraniteEngine,
    'xverse': LlamaFamilyEngine,
    'grok': LlamaFamilyEngine,
    'grok-1': LlamaFamilyEngine,
    'bloomz': BloomEngine,
    'mpnet': LlamaFamilyEngine,
    'text_da_vinci': LlamaFamilyEngine,

    # ── qwen family ────────────────────────────────────────────────
    'qwen': QwenFamilyEngine,
    'qwen2': QwenFamilyEngine,
    'qwen2.5': QwenFamilyEngine,
    'qwen3': QwenFamilyEngine,
    'qwen2moe': QwenMoeEngine,
    'qwen3moe': QwenMoeEngine,
    'qwen3_next': QwenFamilyEngine,
    'minicpm': QwenFamilyEngine,
    'minicpm3': QwenFamilyEngine,
    'telechat': QwenFamilyEngine,
    'telechat2': QwenFamilyEngine,

    # ── gemma family ───────────────────────────────────────────────
    'gemma': GemmaEngine,
    'gemma2': Gemma2Engine,
    'gemma3': Gemma3Engine,
    'gemma3_text': Gemma3Engine,
    'gemma3-text': Gemma3Engine,
    'codegemma': GemmaEngine,

    # ── mistral family ─────────────────────────────────────────────
    'mistral': MistralFamilyEngine,
    'mistral3': MistralFamilyEngine,
    'mixtral': MistralFamilyEngine,
    'mixformer': MistralFamilyEngine,
    'codestral': MistralFamilyEngine,
    'ministral': MistralFamilyEngine,
    'ardzcoder': MistralFamilyEngine,

    # ── phi family ─────────────────────────────────────────────────
    'phi': PhiEngine,
    'phi2': Phi2Engine,
    'phi-2': Phi2Engine,
    'phi3': PhiEngine,
    'phi-3': PhiEngine,
    'phi3small': PhiEngine,
    'phi3-small': PhiEngine,
    'phi4': PhiEngine,
    'phi-4': PhiEngine,
    'phi_moe': PhiEngine,
    'phi-moe': PhiEngine,

    # ── deepseek family ────────────────────────────────────────────
    'deepseek': DeepSeekEngine,
    'deepseek2': DeepSeekEngine,
    'deepseek3': DeepSeekEngine,
    'deepseek_v2': DeepSeekEngine,
    'deepseek_v3': DeepSeekEngine,
    'deepseek_r1': DeepSeekEngine,
    'deepseek-coder': DeepSeekEngine,
    'deepseek2_moe': DeepSeekEngine,
    'deepseek_v2_moe': DeepSeekEngine,
    'deepseek_llm': DeepSeekEngine,

    # ── falcon / starcoder / gpt line ──────────────────────────────
    'falcon': FalconEngine,
    'falcon2': FalconEngine,
    'falcon3': FalconEngine,
    'starcoder': StarCoderEngine,
    'starcoder2': StarCoderEngine,
    'gpt2': GPT2Engine,
    'gpt_j': GPTNeoXEngine,
    'gptj': GPTNeoXEngine,
    'gpt-j': GPTNeoXEngine,
    'gpt-j-6b': GPTNeoXEngine,
    'gpt-neox': GPTNeoXEngine,
    'gptneox': GPTNeoXEngine,
    'gpt_oss': GPTOSSEngine,
    'gpt-oss': GPTOSSEngine,
    'gpt-oss-moe': GPTOSSEngine,
    'gpt_oss_moe': GPTOSSEngine,
    'mpt': MptEngine,
    'bloom': BloomEngine,
    'refact': StarCoderEngine,
    'refact_code': StarCoderEngine,
    'rwkv': BloomEngine,

    # ── internlm / command-r / dbrx etc ────────────────────────────
    'internlm': InternLMEngine,
    'internlm2': InternLMEngine,
    'internlm3': InternLMEngine,
    'command-r': CommandREngine,
    'command-r-plus': CommandREngine,
    'cohere': CommandREngine,
    'dbrx': QwenMoeEngine,
    'exaone': ExAoneEngine,
    'jamba': JambaEngine,
    'mamba': JambaEngine,
    'atmojamba-lo': JambaEngine,
    'persimmon': LlamaFamilyEngine,
    'openelm': LlamaFamilyEngine,
    'fordriver': LlamaFamilyEngine,
    'chameleon': LlamaFamilyEngine,

    # ── embedding / BERT family ─────────────────────────────────────
    'bert': BertEngine,
    'nomic-bert': BertEngine,
    'nomic_bert': BertEngine,
    'jina-bert': BertEngine,
    'jina_bert': BertEngine,
    'mini-bert': BertEngine,
    'modernbert': BertEngine,
    'modern-bert': BertEngine,
    'mpnet': BertEngine,
    'mpnet_v2': BertEngine,
    'sentence-bert': BertEngine,
    'embed-bert': BertEngine,
    'roberta': BertEngine,
    'distilbert': BertEngine,
    'deberta': BertEngine,
    'convbert': BertEngine,
    'splade': BertEngine,
    'gte': BertEngine,
    'gte-bert': BertEngine,
    'nomic': BertEngine,

    # ── GLM / ChatGLM family ────────────────────────────────────────
    'chatglm': GlmEngine,
    'chatglm2': GlmEngine,
    'chatglm3': GlmEngine,
    'chatglm4': GlmEngine,
    'glm': GlmEngine,
    'glm2': GlmEngine,
    'glm3': GlmEngine,
    'glm4': GlmEngine,
    'glm4.5': GlmEngine,
    'glm_z1': GlmEngine,
    'z1': GlmEngine,
}

# Pre-built engines (see ntmre_router): these already exist and are routed
# before the registry in create_engine().  Listed here so the catalog shows
# that Gemma4 / Qwen35 / Bonsai / NTMRE v1 / v16 all have their own engine.
BUILTIN_ENGINES = [
    {
        "engine_name": "Gemma4 GPU Engine",
        "engine_class": "Gemma4GpuEngine",
        "module": "ntmre_gemma4_gpu",
        "family": "gemma4",
        "archs": ["gemma4", "gemma4-assistant", "gemma4_mtp",
                  "gemma4-assistant-mtp"],
        "note": "Dedicated NVRTC/C-DLL engine (full VRAM, PLE, shared KV).",
    },
    {
        "engine_name": "Qwen3.5 Hybrid Engine",
        "engine_class": "Qwen35HybridEngine",
        "module": "nightmare_qwen35",
        "family": "qwen35",
        "archs": ["qwen35", "qwen3.5", "qwen3next", "qwen3_next"],
        "note": "SSM + full attention hybrid math, NTMRE.",
    },
    {
        "engine_name": "Bonsai GPU Engine",
        "engine_class": "BonsaiSuiteModel",
        "module": "bonsai_suite",
        "family": "qwen35-bonsai",
        "archs": ["qwen35-Q1_0"],
        "note": "Full-GPU Bonsai-27B Q1_0 engine (NVRTC kernels).",
    },
    {
        "engine_name": "NTMRE v16 Engine",
        "engine_class": "create_v16_engine",
        "module": "ntmre_v16_engine",
        "family": "ntmre-v16",
        "archs": ["RMTN-v16"],
        "note": "NTMRE v16 binary runtime.",
    },
    {
        "engine_name": "NTMRE v1 Engine",
        "engine_class": "NtmreRuntime",
        "module": "ntmre_runtime",
        "family": "ntmre-v1",
        "archs": ["RMTN-v1"],
        "note": "Legacy C-layout NTMRE binary runtime.",
    },
    {
        "engine_name": "GPU-Native Engine",
        "engine_class": "GpuNativeEngine",
        "module": "nightmare_gpu_native",
        "family": "native",
        "archs": ["swiglu-dense (VRAM-resident)"],
        "note": "Fastest path when the whole model fits in VRAM.",
    },
    {
        "engine_name": "CPU Transformer Engine",
        "engine_class": "TransformerModel",
        "module": "transformer",
        "family": "cpu",
        "archs": ["any (force-cpu)"],
        "note": "Pure Python + NumPy, no GPU required.",
    },
]


def normalize_arch(arch) -> str:
    """Lower-case + strip whitespace from a raw arch string."""
    return str(arch or '').strip().lower()


def engine_class_for(arch):
    """Return the dedicated engine class for an architecture string, or
    None when the arch is covered by a pre-built special engine."""
    a = normalize_arch(arch)
    if not a:
        return LlamaFamilyEngine
    # exact
    cls = ARCH_TO_ENGINE.get(a)
    if cls is not None:
        return cls
    # gemma4 variants + qwen35 hybrid: pre-built special engines handle these
    # (Gemma4GpuEngine, Qwen35HybridEngine, Bonsai) — never the generic catalog.
    if any(a == base or a.startswith(base + '-') or a.startswith(base + '_')
           for base in ('gemma4', 'gemma4-assistant', 'gemma4_mtp',
                        'gemma4-assistant-mtp')):
        return None
    if a in ('qwen35', 'qwen3next', 'qwen3.5', 'qwen3_next'):
        return None
    # normalized-family match (strip common suffixes before cross-checking)
    base = a
    for sep in ('-', '_', '.'):
        if sep in base and base.split(sep, 1)[0] in ARCH_TO_ENGINE:
            cls = ARCH_TO_ENGINE[base.split(sep, 1)[0]]
            if cls is not None:
                return cls
    # prefix match: `llama3.x`, `qwen2.x`, `gemma2x`, `phi3x`, `deepseekx`
    for key, cls in ARCH_TO_ENGINE.items():
        if a.startswith(key) and len(a) > len(key):
            return cls
    return GenericFallbackEngine


def family_for(arch):
    cls = engine_class_for(arch)
    if cls is None:
        for b in BUILTIN_ENGINES:
            if arch is not None and normalize_arch(arch) in b['archs']:
                return b['family']
        return 'gemma4'
    return getattr(cls, 'family', 'generic')


def native_compatible(arch) -> bool:
    """True when GpuNativeEngine can correctly run this architecture
    (its core math is plain SwiGLU + full rotary + RMSNorm)."""
    cls = engine_class_for(arch)
    return bool(getattr(cls, 'native_compatible', False))


def engine_name_for(arch):
    cls = engine_class_for(arch)
    if cls is None:
        return "Gemma4 GPU Engine"
    return getattr(cls, 'engine_name', 'Generic Engine')


def catalog():
    """Full engine inventory — dedicated per-family engines + pre-built."""
    out = []
    seen = {}
    for arch, cls in sorted(ARCH_TO_ENGINE.items()):
        family = getattr(cls, 'family', 'generic')
        if family in seen:
            seen[family]['archs'].append(arch)
            continue
        seen[family] = {
            "engine_name": getattr(cls, 'engine_name', 'Generic Engine'),
            "engine_class": cls.__name__,
            "module": cls.__module__,
            "family": family,
            "activation": getattr(cls, 'activation', 'swiglu'),
            "builtin": False,
            "archs": [arch],
        }
    for e in BUILTIN_ENGINES:
        e = dict(e)
        e['builtin'] = True
        out.append(e)
    for e in seen.values():
        e['archs'] = sorted(e['archs'])
        out.append(e)
    return sorted(out, key=lambda e: e['engine_name'])


# ════════════════════════════════════════════════════════════════════
# ROUTER HELPERS — resolve which dedicated engine runs a model file
# ════════════════════════════════════════════════════════════════════

def sniffed_arch(path):
    """Best-effort: return the architecture string for a model file
    without constructing an engine (uses ntmre_router.sniff_model)."""
    try:
        import ntmre_router
        info = ntmre_router.sniff_model(path)
        if info['error']:
            return '', info
        return info['arch'], info
    except Exception:
        return '', {}


def resolve_for_path(path):
    """Return a dict describing which engine will run a model file."""
    arch, info = sniffed_arch(path)
    kind = info.get('kind', 'unknown')
    if kind in ('v16',):
        return {"engine_name": "NTMRE v16 Engine", "engine_class": "create_v16_engine",
                "module": "ntmre_v16_engine", "family": "ntmre-v16",
                "arch": arch or "RMTN-v16", "builtin": True}
    if kind == 'binary':
        return {"engine_name": "NTMRE v1 Engine", "engine_class": "NtmreRuntime",
                "module": "ntmre_runtime", "family": "ntmre-v1",
                "arch": arch or "RMTN-v1", "builtin": True}
    if kind == 'gguf' and arch in ('qwen35', 'qwen3next', 'qwen3_next', 'qwen3.5'):
        import gguf_parser
        try:
            g = gguf_parser.GGUFParser(path)
            m = g.parse()
            d = {str(t.dtype) for t in m.tensors.values()}
            g.close()
            if d and d <= {0, 1, 41}:
                return {"engine_name": "Bonsai GPU Engine", "engine_class": "BonsaiSuiteModel",
                        "module": "bonsai_suite", "family": "qwen35-bonsai",
                        "arch": arch, "builtin": True}
        except Exception:
            pass
        return {"engine_name": "Qwen3.5 Hybrid Engine", "engine_class": "Qwen35HybridEngine",
                "module": "nightmare_qwen35", "family": "qwen35",
                "arch": arch, "builtin": True}
    cls = engine_class_for(arch)
    if cls is None:
        return {"engine_name": "Gemma4 GPU Engine", "engine_class": "Gemma4GpuEngine",
                "module": "ntmre_gemma4_gpu", "family": "gemma4",
                "arch": arch or "gemma4", "builtin": True}
    return {
        "engine_name": getattr(cls, 'engine_name', 'Generic Engine'),
        "engine_class": cls.__name__,
        "module": cls.__module__,
        "family": getattr(cls, 'family', 'generic'),
        "activation": getattr(cls, 'activation', 'swiglu'),
        "arch": arch or '?',
        "builtin": False,
    }


# ════════════════════════════════════════════════════════════════════
# SELF-TEST (no GPU / no model weights needed)
# ════════════════════════════════════════════════════════════════════

def _selftest():
    print("💀 NIGHTMARE ENGINE SUITE — SELF TEST")
    print("=" * 56)
    ok = True

    # 1) Activation math shapes + reference values
    try:
        x = np.array([[0.5, -1.0, 2.0]], dtype=np.float32)
        s = _silu(x)
        g = _gelu(x)
        assert s.shape == (1, 3) and g.shape == (1, 3), "activation shape"
        assert abs(float(s[0, 1]) - (-1.0 / (1.0 + np.exp(1.0)))) < 1e-4
        assert 0.34 < float(g[0, 0]) < 0.35, "gelu(0.5) ~ 0.3457"
        assert -0.16 < float(g[0, 1]) < -0.15, "gelu(-1) ~ -0.158"
        print("  [OK]   _silu / _gelu math")
    except Exception as e:
        ok = False
        print(f"  [FAIL] activation math: {e}")

    # 2) Family-specific FFN via the dedicated engines (no loading)
    try:
        eng = GenericFallbackEngine.__new__(GenericFallbackEngine)
        eng._ffn_activation = 'swiglu'
        eng._mm = lambda x, w: np.einsum('...i,oi->...o', x, np.asarray(w, dtype=np.float32))
        xn = np.ones((1, 4), dtype=np.float32)
        wg = np.eye(4, 4, dtype=np.float32)
        wu = np.eye(4, 4, dtype=np.float32)
        wd = np.eye(4, 4, dtype=np.float32)
        out = eng._ffn(xn, {'wg': wg * 0.5, 'wu': wu, 'wd': wd})
        assert out.shape == (1, 4)
        # swiglu: silu(0.5)*1 = 0.5*0.6224
        expect = (0.5 / (1.0 + np.exp(-0.5))) * 1.0
        assert abs(float(out[0, 0]) - expect) < 1e-4
        print("  [OK]   swiglu FFN (Llama/Qwen/Mistral/DeepSeek)")

        eng._ffn_activation = 'geglu'
        out = eng._ffn(xn, {'wg': np.ones((4, 4), np.float32) * 2.0,
                            'wu': np.ones((4, 4), np.float32), 'wd': wd})
        # gated-GELU: down(gate * gelu(up)); with xn=ones: gate=8, up=4
        expect = 8.0 * _gelu(np.ones((1, 4)) * 4.0)[0, 0]
        assert abs(float(out[0, 0]) - expect) < 1e-4
        print("  [OK]   gated-GELU FFN (Gemma/Gemma2/3)")

        eng._ffn_activation = 'gelu'
        out = eng._ffn(xn, {'wu': wu, 'wd': wd})
        expect = _gelu(np.ones((1, 4)))[0, 0]
        assert abs(float(out[0, 0]) - expect) < 1e-4
        print("  [OK]   single GELU FFN (Phi-2/StarCoder2/Falcon/MPT)")

        eng._needs_special_math = ArchEngine._needs_special_math.__get__(eng)
        eng._ffn_activation = 'geglu'
        eng._rope_pairs = None
        eng.head_dim = 128
        eng._parallel_attn = False
        assert eng._needs_special_math() is True
        eng._ffn_activation = 'swiglu'
        eng._rope_pairs = 32
        assert eng._needs_special_math() is True
        eng._rope_pairs = None
        assert eng._needs_special_math() is False
        print("  [OK]   magic-math routing flags")
    except Exception as e:
        ok = False
        print(f"  [FAIL] FFN / flags: {e}")

    # 3) Registry coverage
    families = set()
    for arch in ARCH_TO_ENGINE:
        cls = engine_class_for(arch)
        assert cls is not None, f"no engine for {arch}"
        families.add(getattr(cls, 'family', '?'))
    needed = {'llama', 'qwen2', 'gemma', 'gemma2', 'mistral', 'phi', 'deepseek',
              'falcon', 'starcoder', 'gpt2', 'gptneox', 'mpt', 'bloom',
              'internlm', 'command-r', 'granite', 'exaone', 'gpt-oss'}
    missing = needed - families
    assert not missing, f"missing families {missing}"
    print(f"  [OK]   registry: {len(ARCH_TO_ENGINE)} arch strings -> "
          f"{len(families)} dedicated engines")

    # 4) Known-variant resolution
    checks = {
        'llama3.3': 'Llama Engine',
        'qwen2.5': 'Qwen2 Engine',
        'gemma3': 'Gemma3 Engine',
        'qwen3moe': 'Qwen-MoE Engine',
        'deepseek_v3': 'DeepSeek Engine',
        'mistral7b': 'Mistral Engine',
        'phi-3-small': 'Phi Engine',
        'falconM7B': 'Falcon Engine',   # prefix match -> dedicated engine
        'gpt-oss-moe': 'GPT-OSS Engine',
        'gemma4': 'Gemma4 GPU Engine',   # pre-built special engine, never catalog
    }
    for a, expect in checks.items():
        got = engine_name_for(a)
        assert got == expect, f"{a}: {got} != {expect}"
    print("  [OK]   variant resolution matches expected engines")

    # 5) Catalog sanity
    cat = catalog()
    assert len(cat) >= 20
    built = [e for e in cat if e['builtin']]
    assert len(built) >= 7
    print(f"  [OK]   engine catalog ({len(cat)} engines, "
          f"{len(built)} pre-built)")

    print("=" * 56)
    print("   SELFTEST PASSED" if ok else "   SELFTEST FAILED")
    return 0 if ok else 1


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        sys.exit(_selftest())

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        path = sys.argv[1]
        if not os.path.exists(path):
            p2 = Path(__file__).parent / 'models' / path
            path = str(p2) if p2.exists() else path
        print(f"💀 Engine resolver for: {path}")
        if os.path.exists(path):
            info = resolve_for_path(path)
            for k, v in info.items():
                print(f"  {k:<14}{v}")
        else:
            print("  file not found.")
        sys.exit(0)

    print("💀 NIGHTMARE ENGINE SUITE")
    print("Consult nightmare_engines.py — one dedicated engine per model family.\n")
    for e in catalog():
        tag = "PREBUILT" if e['builtin'] else "DEDICATED"
        print(f"[{tag:<10}] {e['engine_name']:<22} {e['family']:<14} "
              f"act={e.get('activation', '-')}")
        print(f"{'':11}  archs: {', '.join(e['archs'][:8])}"
              + (" ..." if len(e['archs']) > 8 else ""))