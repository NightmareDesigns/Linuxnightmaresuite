# -*- coding: utf-8 -*-
"""
bonsai_suite.py — Adapter to run the Bonsai (qwen35 hybrid) GPU engine
inside the Nightmare Suite GUI.

The suite expects a model object exposing:
  * forward(token_id, position) -> logits
  * kv_cache.advance()          (no-op here — BonsaiGPU manages state internally)
  * reset()                     (clear SSM/conv/KV state for a new message)
  * model.name / architecture / metadata
  * dim / n_layers / vocab_size / max_seq_len
  * no_bos = True               (Bonsai has add_bos_token=False)

This wraps bonsai_gpu.BonsaiGPU (the corrected engine) with that interface.
"""
import os
import nightmare_np as np


class _DummyKV:
    """The suite calls kv_cache.advance() after each token; BonsaiGPU tracks
    its own recurrent/attention state, so this is a no-op."""

    def advance(self):
        pass


class BonsaiSuiteModel:
    def __init__(self, filepath, device=1, max_seq_len=2048):
        try:
            import bonsai_gpu_13
            self._bm = bonsai_gpu_13.BonsaiGPU13(filepath, device=device,
                                                 max_seq=max(4096, max_seq_len))
            self._v13 = True
        except Exception as e:
            print('[bonsai_suite] full-GPU engine unavailable (%s) - streaming fallback' % e)
            import bonsai_gpu
            self._bm = bonsai_gpu.BonsaiGPU(filepath, device=device)
            self._v13 = False
        self.model = self._bm.m          # parsed GGUF model (has .metadata/.name/.architecture)
        self.dim = self._bm.dim
        self.n_layers = self._bm.n_layers
        self.vocab_size = self._bm.vocab
        self.max_seq_len = min(max_seq_len, self._bm.m.context_length)
        self.kv_cache = _DummyKV()
        self.loaded = True
        self.no_bos = True               # Bonsai GGUF has add_bos_token=False
        # expose the tokenizer the suite will re-create anyway
        self.tok = self._bm.tok

    @property
    def use_gpu(self):
        """True when the wrapped engine holds a live CUDA/HIP context.

        ntmre_router._model_has_gpu() probes for _fused_ok / _fused_devs /
        _gpu_workers / _gpumm / use_gpu before declaring that a routed engine
        "engaged no accelerator". This wrapper hides all of those behind
        self._bm, so the check used to report

            [PLAN] WARNING: routed to BonsaiSuiteModel but NO accelerator
            was engaged - running on CPU

        for a model that had in fact just reported
            "[bonsai] 3.78GB quantized weights resident in VRAM (0.8s upload)".

        The inner engine keeps its device handle in .cu (a Cuda / HipCuda
        object), so that is the honest signal to surface here.
        """
        return getattr(self._bm, 'cu', None) is not None

    def forward(self, token_id, position):
        if self._v13:
            if position == 0:
                self._bm.reset()
            return self._bm.forward(token_id)
        return self._bm.forward(token_id, position)

    def close(self):
        """Release GPU/RAM when switching models."""
        try:
            if hasattr(self._bm, 'close'):
                self._bm.close()
        except Exception:
            pass
        self._bm = None
        self.tok = None
        self.kv_cache = None
        self.model = None
        import gc
        gc.collect()

    def reset(self):
        """Clear all recurrent/attention state for a fresh message."""
        bm = self._bm
        if self._v13:
            bm.reset()
            return
        bm.ssm_state_mem = [np.zeros((bm.ssm_state, bm.head_v_dim, bm.ssm_dt), np.float32)
                            for _ in range(bm.n_layers)]
        bm.conv_mem = [np.zeros((bm.ssm_conv - 1, bm.conv_dim), np.float32)
                       for _ in range(bm.n_layers)]
        bm.kv = {l: [] for l in range(bm.n_layers)}
