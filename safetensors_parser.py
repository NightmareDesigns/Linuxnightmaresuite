# -*- coding: utf-8 -*-
"""
safetensors_parser.py — SafeTensors model loader for the Nightmare Suite
======================================================================
Parses HuggingFace .safetensors files and presents a GGUF-compatible
interface so the existing GPU-native and streaming engines can load
them with zero changes.

SafeTensors format (very simple):
  8 bytes  : u64 little-endian header length (N)
  N bytes  : JSON header  {"tensor_name": {"dtype","shape","data_offsets"}}
  rest     : raw tensor bytes (no quantization — just F32 / F16 / BF16)

This module is 100% suite-internal code.  No third-party parsers.
Tensor names are mapped from HuggingFace naming (model.layers.0.self_attn.q_proj.weight)
to the suite's internal naming (blk.0.attn_q.weight) so every engine,
converter, and GPU kernel works unchanged.

Multi-file (sharded) models are supported:
  model-00001-of-00005.safetensors + model-00002-of-00005.safetensors + ...

Author: Nightmare Designs 💀
"""

import json
import os
import re
import struct
import mmap as _mmap_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── SafeTensors dtype string → GGUF dtype int ──────────────────────
_ST_DTYPE_MAP: Dict[str, int] = {
    "F32": 0,      # GGUF_TYPE_F32
    "F16": 1,      # GGUF_TYPE_F16
    "BF16": 30,    # GGUF_TYPE_BF16
    "F64": 0,      # treat as F32 (will truncate)
    "I8":  24,     # GGUF_TYPE_I8
    "I16": 25,     # GGUF_TYPE_I16
    "I32": 26,     # GGUF_TYPE_I32
    "I64": 27,     # GGUF_TYPE_I64
    "U8":  24,     # treat as I8
    "BOOL": 0,     # treat as F32
}

# Bytes per element for each SafeTensors dtype
_ST_BYTE_SIZE: Dict[str, int] = {
    "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "BOOL": 1,
}


@dataclass
class STTensorInfo:
    """Mirrors gguf_parser.TensorInfo."""
    name: str           # GGUF-style internal name (blk.0.attn_q.weight)
    n_dims: int
    shape: tuple        # natural (out, in) order — NOT reversed
    dtype: int          # GGUF dtype int
    offset: int         # byte offset within the file's data section
    file_idx: int = 0   # which shard file this tensor lives in
    st_dtype: str = ""  # original SafeTensors dtype string


@dataclass
class SafeTensorsModel:
    """Mirrors gguf_parser.GGUFModel — provides metadata + tensors."""
    metadata: dict = field(default_factory=dict)
    tensors: dict = field(default_factory=dict)   # name -> STTensorInfo
    file_path: str = ""

    def _safe_int(self, value, default=0) -> int:
        if value is None:
            return default
        if isinstance(value, list):
            return max(value) if value else default
        if isinstance(value, float):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_float(self, value, default=0.0) -> float:
        if value is None:
            return default
        if isinstance(value, list):
            return float(value[0]) if value else default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


# ── HuggingFace → suite tensor-name mapping ────────────────────────

# Non-layer tensors
_STATIC_MAP = {
    "model.embed_tokens.weight":          "token_embd.weight",
    "model.embeddings.word_embeddings.weight": "token_embd.weight",
    "transformer.wte.weight":             "token_embd.weight",
    "model.norm.weight":                  "output_norm.weight",
    "model.norm_f.weight":                "output_norm.weight",
    "transformer.ln_f.weight":            "output_norm.weight",
    "model.final_layernorm.weight":       "output_norm.weight",
    "lm_head.weight":                     "output.weight",
    "output.weight":                      "output.weight",
    "transformer.lm_head.weight":         "output.weight",
    "model.lm_head.weight":               "output.weight",
    "model.embed_tokens.norm.weight":     "token_embd_norm.weight",
}

# Per-layer suffix mapping (applied after extracting the layer number)
_LAYER_MAP = {
    # Attention norm
    "input_layernorm.weight":             "attn_norm.weight",
    "ln_1.weight":                        "attn_norm.weight",
    "attention_norm.weight":              "attn_norm.weight",
    "attn_norm.weight":                   "attn_norm.weight",
    # Q / K / V / O projections
    "self_attn.q_proj.weight":             "attn_q.weight",
    "self_attn.k_proj.weight":             "attn_k.weight",
    "self_attn.v_proj.weight":             "attn_v.weight",
    "self_attn.o_proj.weight":            "attn_output.weight",
    "self_attn.Wqkv.weight":              "attn_qkv.weight",   # fused QKV
    "attention.wq.weight":                "attn_q.weight",
    "attention.wk.weight":                "attn_k.weight",
    "attention.wv.weight":                "attn_v.weight",
    "attention.wo.weight":                "attn_output.weight",
    "attn.q.weight":                      "attn_q.weight",
    "attn.k.weight":                      "attn_k.weight",
    "attn.v.weight":                      "attn_v.weight",
    "attn.o.weight":                      "attn_output.weight",
    # QKV biases (qwen2 family)
    "self_attn.q_proj.bias":               "attn_q.bias",
    "self_attn.k_proj.bias":               "attn_k.bias",
    "self_attn.v_proj.bias":               "attn_v.bias",
    "self_attn.o_proj.bias":               "attn_output.bias",
    # FFN norm
    "post_attention_layernorm.weight":    "ffn_norm.weight",
    "ln_2.weight":                        "ffn_norm.weight",
    "ffn_norm.weight":                    "ffn_norm.weight",
    # FFN projections
    "mlp.gate_proj.weight":               "ffn_gate.weight",
    "mlp.up_proj.weight":                 "ffn_up.weight",
    "mlp.down_proj.weight":               "ffn_down.weight",
    "mlp.fc_in.weight":                   "ffn_gate.weight",
    "mlp.fc_out.weight":                  "ffn_down.weight",
    "feed_forward.w1.weight":             "ffn_gate.weight",
    "feed_forward.w2.weight":             "ffn_down.weight",
    "feed_forward.w3.weight":             "ffn_up.weight",
    "mlp.c_fc.weight":                    "ffn_gate.weight",
    "mlp.c_proj.weight":                  "ffn_down.weight",
    # MoE router
    "block_sparse_moe.gate.weight":       "ffn_gate_inp.weight",
    "mlp.gate.weight":                     "ffn_gate_inp.weight",
    # Attention output bias
    "self_attn.o_proj.bias":              "attn_output.bias",
}


def _map_tensor_name(hf_name: str) -> Optional[str]:
    """Map a HuggingFace tensor name to the suite's internal name."""
    if hf_name in _STATIC_MAP:
        return _STATIC_MAP[hf_name]

    # model.layers.{i}.*  (Llama / Mistral / Qwen2 / Gemma / Yi ...)
    m = re.match(r"model\.layers\.(\d+)\.(.+)", hf_name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        if rest in _LAYER_MAP:
            return f"blk.{layer}.{_LAYER_MAP[rest]}"
        # MoE expert weights: block_sparse_moe.experts.{j}.w1.weight
        me = re.match(r"block_sparse_moe\.experts\.(\d+)\.w(\d)\.weight", rest)
        if me:
            exp, wnum = int(me.group(1)), int(me.group(2))
            tag = {1: "ffn_gate", 2: "ffn_down", 3: "ffn_up"}.get(wnum)
            if tag:
                return f"blk.{layer}.experts.{exp}.{tag}.weight"
        return None

    # transformer.h.{i}.*  (GPT-2 / GPT-NeoX family)
    m = re.match(r"transformer\.h\.(\d+)\.(.+)", hf_name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        if rest in _LAYER_MAP:
            return f"blk.{layer}.{_LAYER_MAP[rest]}"
        # GPT-2 fused QKV: attn.c_attn.weight
        if rest == "attn.c_attn.weight":
            return f"blk.{layer}.attn_qkv.weight"
        return None

    return None


class SafeTensorsParser:
    """Parser for .safetensors files — GGUF-compatible interface.

    Usage (identical to gguf_parser.GGUFParser):
        parser = SafeTensorsParser(path)
        model  = parser.parse()
        arr   = parser.load_tensor(model, "blk.0.attn_q.weight")
        raw   = parser.load_tensor_raw(model, "blk.0.attn_q.weight")
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.shape_reversed = False   # SafeTensors shapes are natural (out, in)
        self._mmaps: List[Tuple] = []  # (mmap_obj, file_obj, data_offset)
        self._files: List[Path] = []
        self._model: Optional[SafeTensorsModel] = None

    # ── File discovery (single + sharded) ───────────────────────────

    def _find_shards(self) -> List[Path]:
        """Find all shard files for this model."""
        name = self.file_path.name
        # Sharded: model-00001-of-00005.safetensors
        m = re.match(r"(.+)-(\d+)-of-(\d+)(\.safetensors)", name)
        if m:
            stem, idx, total, ext = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            parent = self.file_path.parent
            shards = []
            for i in range(1, total + 1):
                p = parent / f"{stem}-{i:05d}-of-{total:05d}{ext}"
                if p.exists():
                    shards.append(p)
                else:
                    # try without zero-padding
                    p2 = parent / f"{stem}-{i}-of-{total}{ext}"
                    if p2.exists():
                        shards.append(p2)
            return sorted(shards) if shards else [self.file_path]
        return [self.file_path]

    # ── Header parsing ──────────────────────────────────────────────

    def parse(self) -> SafeTensorsModel:
        shards = self._find_shards()
        self._files = shards
        model = SafeTensorsModel(file_path=str(self.file_path))
        self._model = model

        for file_idx, fpath in enumerate(shards):
            self._parse_one_shard(fpath, file_idx, model)

        # Infer architecture metadata from tensor shapes / config.json
        self._infer_metadata(model)

        # Load tokenizer if available alongside the model
        self._load_tokenizer(model)

        # Print summary
        arch = model.metadata.get("general.architecture", "llama")
        print(f"[SafeTensors] {arch}: {model.metadata.get('llama.embedding_length', '?')}d × "
              f"{model.metadata.get('llama.block_count', '?')}L × "
              f"{model.metadata.get('llama.attention.head_count', '?')}H, "
              f"{len(model.tensors)} tensors, {len(shards)} shard(s)")

        return model

    def _parse_one_shard(self, fpath: Path, file_idx: int, model: SafeTensorsModel):
        """Parse one shard's header and mmap its data."""
        with open(fpath, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"{fpath}: file too small for SafeTensors header")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_json = f.read(header_len).decode("utf-8")

        header = json.loads(header_json)
        data_offset = 8 + header_len

        # Memory-map the file for fast tensor access
        mmap_file = open(fpath, "rb")
        try:
            mm = _mmap_mod.mmap(mmap_file.fileno(), 0, access=_mmap_mod.ACCESS_READ)
        except Exception:
            mm = None
        self._mmaps.append((mm, mmap_file, data_offset))

        # Process each tensor
        for hf_name, info in header.items():
            if hf_name == "__metadata__":
                # Store any custom metadata from the safetensors file
                if isinstance(info, dict):
                    for k, v in info.items():
                        model.metadata[f"safetensors.{k}"] = v
                continue

            gguf_name = _map_tensor_name(hf_name)
            if gguf_name is None:
                continue  # skip unmapped tensors (e.g., rotary emb inv_freq)

            st_dtype = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            data_offsets = info.get("data_offsets", [0, 0])

            gguf_dtype = _ST_DTYPE_MAP.get(st_dtype, 0)
            n_dims = len(shape)
            n_elements = 1
            for d in shape:
                n_elements *= d

            model.tensors[gguf_name] = STTensorInfo(
                name=gguf_name,
                n_dims=n_dims,
                shape=shape,
                dtype=gguf_dtype,
                offset=data_offsets[0],   # relative to data section
                file_idx=file_idx,
                st_dtype=st_dtype,
            )

    # ── Metadata inference ─────────────────────────────────────────

    def _infer_metadata(self, model: SafeTensorsModel):
        """Infer GGUF-style metadata from tensor shapes + optional config.json."""
        meta = model.metadata
        arch = "llama"  # default; SafeTensors doesn't store arch

        # Try to read config.json alongside the model
        config = self._read_config_json()
        if config:
            archs = config.get("architectures", [])
            if archs:
                a0 = str(archs[0]).lower()
                # Map HF class names to our arch strings
                if "qwen2" in a0:
                    arch = "qwen2"
                elif "qwen3" in a0:
                    arch = "qwen2"  # qwen3 uses same layout
                elif "gemma2" in a0:
                    arch = "gemma"
                elif "gemma" in a0:
                    arch = "gemma"
                elif "mistral" in a0:
                    arch = "llama"
                elif "phi" in a0:
                    arch = "phi3"
                elif "gpt2" in a0:
                    arch = "gpt2"
                elif "falcon" in a0:
                    arch = "falcon"
                elif "yi" in a0:
                    arch = "llama"
                elif "mixtral" in a0:
                    arch = "llama"

        meta["general.architecture"] = arch

        # Infer from config.json if available
        if config:
            meta[f"{arch}.embedding_length"] = config.get("hidden_size",
                config.get("n_embd", 4096))
            meta[f"{arch}.block_count"] = config.get("num_hidden_layers",
                config.get("n_layer", 32))
            meta[f"{arch}.attention.head_count"] = config.get("num_attention_heads",
                config.get("n_head", 32))
            meta[f"{arch}.attention.head_count_kv"] = config.get("num_key_value_heads",
                meta[f"{arch}.attention.head_count"])
            meta[f"{arch}.feed_forward_length"] = config.get("intermediate_size",
                config.get("n_inner", 11008))
            meta[f"{arch}.vocabulary_length"] = config.get("vocab_size", 32000)
            meta[f"{arch}.context_length"] = config.get("max_position_embeddings",
                config.get("n_positions", 2048))
            meta[f"{arch}.rope.freq_base"] = config.get("rope_theta", 10000.0)
            eps_key = "rms_norm_eps" if "rms_norm_eps" in config else "layer_norm_eps"
            meta[f"{arch}.attention.layer_norm_rms_epsilon"] = config.get(eps_key, 1e-5)
            # MoE
            if "num_local_experts" in config:
                meta[f"{arch}.expert_count"] = config["num_local_experts"]
                meta[f"{arch}.expert_used_count"] = config.get("num_experts_per_tok", 2)
            return

        # No config.json — infer purely from tensor shapes
        emb = model.tensors.get("token_embd.weight")
        if emb is not None:
            meta[f"{arch}.embedding_length"] = int(emb.shape[1] if len(emb.shape) >= 2 else emb.shape[0])
            meta[f"{arch}.vocabulary_length"] = int(emb.shape[0] if len(emb.shape) >= 2 else 1)

        # Count layers
        layer_nums = set()
        for name in model.tensors:
            m = re.match(r"blk\.(\d+)\.", name)
            if m:
                layer_nums.add(int(m.group(1)))
        meta[f"{arch}.block_count"] = max(layer_nums) + 1 if layer_nums else 32

        # Infer head_dim and n_heads from q_proj shape
        q = model.tensors.get("blk.0.attn_q.weight")
        k = model.tensors.get("blk.0.attn_k.weight")
        if q is not None and len(q.shape) >= 2:
            q_out = int(q.shape[0])  # natural order: [out, in]
            dim = meta.get(f"{arch}.embedding_length", 4096)
            # Try common head_dim values
            for hd in (256, 128, 96, 80, 72, 64, 48, 32):
                if q_out % hd == 0:
                    n_heads = q_out // hd
                    if 1 <= n_heads <= 256:
                        meta[f"{arch}.attention.head_count"] = n_heads
                        meta[f"{arch}.attention.key_length"] = hd
                        break
            else:
                meta[f"{arch}.attention.head_count"] = 32
                meta[f"{arch}.attention.key_length"] = q_out // 32 if q_out % 32 == 0 else 128
        else:
            meta[f"{arch}.attention.head_count"] = 32

        # KV heads
        if k is not None and len(k.shape) >= 2:
            k_out = int(k.shape[0])
            head_dim = meta.get(f"{arch}.attention.key_length", 128)
            if head_dim and k_out % head_dim == 0:
                meta[f"{arch}.attention.head_count_kv"] = k_out // head_dim
            else:
                meta[f"{arch}.attention.head_count_kv"] = meta[f"{arch}.attention.head_count"]
        else:
            meta[f"{arch}.attention.head_count_kv"] = meta[f"{arch}.attention.head_count"]

        # FFN dim
        gate = model.tensors.get("blk.0.ffn_gate.weight")
        if gate is not None and len(gate.shape) >= 2:
            meta[f"{arch}.feed_forward_length"] = int(gate.shape[0])
        else:
            meta[f"{arch}.feed_forward_length"] = 11008

        # Defaults
        meta[f"{arch}.context_length"] = 2048
        meta[f"{arch}.rope.freq_base"] = 10000.0
        meta[f"{arch}.attention.layer_norm_rms_epsilon"] = 1e-5

    def _read_config_json(self) -> Optional[dict]:
        """Look for config.json alongside the model file."""
        for candidate in [
            self.file_path.parent / "config.json",
            self.file_path.parent.parent / "config.json",
        ]:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    # ── Tokenizer loading ───────────────────────────────────────────

    def _load_tokenizer(self, model: SafeTensorsModel):
        """Load tokenizer from tokenizer.json or tokenizer.model alongside model."""
        tok_json = self.file_path.parent / "tokenizer.json"
        tok_cfg = self.file_path.parent / "tokenizer_config.json"

        tokens = []
        merges = []
        bos_id = 1
        eos_id = 2

        if tok_json.exists():
            try:
                with open(tok_json, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                # Build vocab: id → token string
                vocab = {}
                # Model vocab
                model_vocab = tj.get("model", {}).get("vocab", {})
                for tok, idx in model_vocab.items():
                    vocab[int(idx)] = tok
                # Added tokens
                for at in tj.get("added_tokens", []):
                    vocab[int(at["id"])] = at["content"]
                # Merges
                merges = tj.get("model", {}).get("merges", [])

                if vocab:
                    max_id = max(vocab.keys())
                    tokens = [""] * (max_id + 1)
                    for idx, tok in vocab.items():
                        if idx < len(tokens):
                            tokens[idx] = tok

                # BOS / EOS from tokenizer_config.json
                if tok_cfg.exists():
                    try:
                        with open(tok_cfg, "r", encoding="utf-8") as f:
                            tc = json.load(f)
                        for key in ("bos_token", "eos_token"):
                            val = tc.get(key)
                            if isinstance(val, dict):
                                val = val.get("content", "")
                            if val and val in {v: k for k, v in vocab.items()}:
                                tid = {v: k for k, v in vocab.items()}[val]
                                if key == "bos_token":
                                    bos_id = tid
                                else:
                                    eos_id = tid
                    except Exception:
                        pass
            except Exception as e:
                print(f"[SafeTensors] tokenizer.json parse failed: {e}")

        if tokens:
            model.metadata["tokenizer.ggml.tokens"] = tokens
            model.metadata["tokenizer.ggml.merges"] = merges
            model.metadata["tokenizer.ggml.bos_token_id"] = bos_id
            model.metadata["tokenizer.ggml.eos_token_id"] = eos_id
            model.metadata["tokenizer.ggml.model"] = "gpt2"
            print(f"[SafeTensors] Tokenizer loaded: {len(tokens)} tokens, "
                  f"BOS={bos_id}, EOS={eos_id}")
        else:
            print("[SafeTensors] No tokenizer.json found — using basic fallback")

    # ── Tensor loading (GGUF-compatible interface) ─────────────────

    def load_tensor(self, model: SafeTensorsModel, tensor_name: str) -> np.ndarray:
        """Load a tensor and return it as float32 numpy array."""
        if tensor_name not in model.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found")
        info = model.tensors[tensor_name]
        raw = self._read_raw(info)
        return self._to_f32(raw, info).reshape(info.shape)

    def load_tensor_raw(self, model: SafeTensorsModel, tensor_name: str):
        """Load raw tensor bytes without dequantizing.

        Returns: (raw_bytes: np.ndarray[uint8], dtype: int, shape: tuple, n_elements: int)
        """
        if tensor_name not in model.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found")
        info = model.tensors[tensor_name]
        raw = self._read_raw(info)
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        return raw, info.dtype, info.shape, n_elements

    def _read_raw(self, info: STTensorInfo) -> np.ndarray:
        """Read raw bytes for a tensor from the correct shard's mmap."""
        mm, _fobj, data_offset = self._mmaps[info.file_idx]
        byte_size = _ST_BYTE_SIZE.get(info.st_dtype, 4)
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        total_bytes = n_elements * byte_size
        start = data_offset + info.offset

        if mm is not None:
            return np.frombuffer(mm[start:start + total_bytes], dtype=np.uint8).copy()
        else:
            fpath = self._files[info.file_idx]
            with open(fpath, "rb") as f:
                f.seek(start)
                return np.frombuffer(f.read(total_bytes), dtype=np.uint8).copy()

    @staticmethod
    def _to_f32(raw: np.ndarray, info: STTensorInfo) -> np.ndarray:
        """Convert raw bytes to float32 based on the SafeTensors dtype."""
        dt = info.st_dtype
        if dt == "F32":
            return np.frombuffer(raw.tobytes(), dtype=np.float32).copy()
        elif dt == "F16":
            return np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32)
        elif dt == "BF16":
            # BF16 = upper 16 bits of float32 → left-shift 16
            u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16).astype(np.uint32)
            return (u16 << 16).view(np.float32).copy()
        elif dt == "F64":
            return np.frombuffer(raw.tobytes(), dtype=np.float64).astype(np.float32)
        elif dt == "I8" or dt == "U8":
            return np.frombuffer(raw.tobytes(), dtype=np.int8 if dt == "I8" else np.uint8).astype(np.float32)
        elif dt == "I16":
            return np.frombuffer(raw.tobytes(), dtype=np.int16).astype(np.float32)
        elif dt == "I32":
            return np.frombuffer(raw.tobytes(), dtype=np.int32).astype(np.float32)
        elif dt == "I64":
            return np.frombuffer(raw.tobytes(), dtype=np.int64).astype(np.float32)
        elif dt == "BOOL":
            return np.frombuffer(raw.tobytes(), dtype=np.bool_).astype(np.float32)
        else:
            # Unknown — try F32
            return np.frombuffer(raw.tobytes(), dtype=np.float32).copy()

    # ── Cleanup ────────────────────────────────────────────────────

    def close(self):
        """Close all mmap and file handles."""
        for mm, fobj, _ in self._mmaps:
            try:
                if mm is not None:
                    mm.close()
            except Exception:
                pass
            try:
                fobj.close()
            except Exception:
                pass
        self._mmaps.clear()


def quantize_to_q8_34(f32_array: np.ndarray) -> bytes:
    """Quantize a float32 array to NSVD Q8_34 format (34 bytes / 32 elems).

    Layout per 32-element block: fp16 scale (2 bytes) + 32 × int8 (32 bytes).
    This cuts F16/BF16 (2 bytes/elem) to ~1.06 bytes/elem — nearly 2x smaller.
    Used for on-the-fly quantization of SafeTensors models during upload.
    """
    n = len(f32_array)
    if n == 0:
        return b''
    pad = (32 - n % 32) % 32
    if pad:
        f32_array = np.concatenate([f32_array, np.zeros(pad, dtype=np.float32)])
    blocks = f32_array.reshape(-1, 32)  # (n_blocks, 32)
    # Per-block scale = max(|block|) / 127
    scales = np.max(np.abs(blocks), axis=1) / 127.0
    scales = np.where(scales == 0, 1e-10, scales).astype(np.float32)
    # Quantize to int8
    q = np.clip(np.round(blocks / scales[:, None]), -128, 127).astype(np.int8)
    scales_f16 = scales.astype(np.float16)
    # Pack: [fp16_scale (2B), int8×32 (32B)] per block via structured array
    n_blocks = len(blocks)
    out = np.empty(n_blocks, dtype=[('scale', np.float16), ('q', np.int8, (32,))])
    out['scale'] = scales_f16
    out['q'] = q
    return out.tobytes()[:((n + 31) // 32) * 34]
