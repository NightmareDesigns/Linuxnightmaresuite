#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║   NIGHTMARE MODEL CONFIG — HuggingFace Auto-Detection Engine    ║
║                                                                   ║
║   Given a GGUF file OR a HuggingFace model ID, returns the      ║
║   complete inference configuration. No guessing. No manual.      ║
║                                                                   ║
║   Supports: LLaMA, Qwen, Gemma, Mistral, Phi, CodeGemma,       ║
║             StableLM, DeepSeek, Yi, Command-R, InternLM,        ║
║             ChatGLM, Falcon, MPT, OLMo, and more.               ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import json
import struct
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ═══════════════════════════════════════════════════════════════════
# MODEL CONFIG (standardized output)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Universal model configuration — everything the engine needs."""
    # Identity
    name: str = ""
    architecture: str = "llama"     # llama, qwen2, gemma, gemma2, mistral, phi3, etc.
    source: str = ""                # "gguf_metadata", "huggingface", "manual"
    
    # Core dims
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32            # GQA: fewer KV heads than Q heads
    head_dim: int = 128             # dim // n_heads (override if non-standard)
    vocab_size: int = 32000
    
    # FFN
    ffn_dim: int = 11008            # intermediate_size
    ffn_type: str = "swiglu"        # swiglu, gelu, relu
    
    # Context & RoPE
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None  # {"type": "linear", "factor": 4.0} etc.
    
    # Norms
    norm_type: str = "rmsnorm"      # rmsnorm, layernorm
    norm_eps: float = 1e-5
    norm_bias: bool = False
    
    # Attention
    attn_bias: bool = False
    qk_norm: bool = False           # Gemma 2 style per-head Q/K norm
    sliding_window: Optional[int] = None  # Gemma 2, Mistral sliding window
    logit_softcap: Optional[float] = None  # Gemma 2 tanh softcap
    
    # Embeddings
    tie_word_embeddings: bool = True
    embedding_scale: Optional[float] = None  # Gemma: sqrt(dim)
    
    # Special
    has_per_layer_embedding: bool = False  # Gemma 4 PLE
    
    # Tokenizer
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    chat_template: str = ""         # "chatml", "llama3", "gemma", etc.
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE DATABASE
# Known model families and their quirks
# ═══════════════════════════════════════════════════════════════════

ARCH_DEFAULTS = {
    "llama": {
        "ffn_type": "swiglu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "attn_bias": False,
        "tie_word_embeddings": True,
        "chat_template": "llama3",
    },
    "qwen2": {
        "ffn_type": "swiglu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-6,
        "attn_bias": True,
        "tie_word_embeddings": True,
        "rope_theta": 1000000.0,
        "chat_template": "chatml",
    },
    "gemma": {
        "ffn_type": "gelu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-6,
        "attn_bias": False,
        "tie_word_embeddings": True,
        "embedding_scale": True,  # multiply by sqrt(dim)
        "chat_template": "gemma",
    },
    "gemma2": {
        "ffn_type": "gelu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-6,
        "attn_bias": False,
        "tie_word_embeddings": True,
        "embedding_scale": True,
        "qk_norm": True,
        "logit_softcap": 30.0,
        "sliding_window": 4096,
        "chat_template": "gemma",
    },
    "mistral": {
        "ffn_type": "swiglu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "attn_bias": False,
        "tie_word_embeddings": False,
        "sliding_window": 4096,
        "chat_template": "mistral",
    },
    "phi3": {
        "ffn_type": "swiglu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "attn_bias": False,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "chat_template": "phi3",
    },
    "deepseek": {
        "ffn_type": "swiglu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-6,
        "attn_bias": False,
        "tie_word_embeddings": False,
        "chat_template": "deepseek",
    },
    "starcoder2": {
        "ffn_type": "gelu",
        "norm_type": "layernorm",
        "norm_eps": 1e-5,
        "attn_bias": True,
        "tie_word_embeddings": True,
        "chat_template": "starcoder",
    },
    "falcon": {
        "ffn_type": "gelu",
        "norm_type": "layernorm",
        "norm_eps": 1e-5,
        "attn_bias": False,
        "tie_word_embeddings": False,
        "chat_template": "falcon",
    },
    "internlm2": {
        "ffn_type": "swiglu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "attn_bias": False,
        "tie_word_embeddings": False,
        "rope_theta": 1000000.0,
        "chat_template": "internlm",
    },
    "command-r": {
        "ffn_type": "swiglu",
        "norm_type": "layernorm",
        "norm_eps": 1e-5,
        "attn_bias": False,
        "tie_word_embeddings": True,
        "chat_template": "command-r",
    },
}

# Map GGUF architecture strings to our normalized names
GGUF_ARCH_MAP = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen": "qwen2",
    "qwen2.5": "qwen2",
    "qwen3": "qwen2",
    "qwen35": "qwen35",
    "qwen3.5": "qwen35",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma2",
    "gemma4": "gemma2",
    "mistral": "mistral",
    "mixtral": "mistral",
    "phi3": "phi3",
    "phi4": "phi3",
    "phi": "phi3",
    "starcoder2": "starcoder2",
    "starcoder": "starcoder2",
    "falcon": "falcon",
    "deepseek2": "deepseek",
    "deepseek3": "deepseek",
    "deepseek": "deepseek",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseek_r1": "deepseek",
    "internlm2": "internlm2",
    "command-r": "command-r",
    "cohere": "command-r",
    "smollm": "llama",
    "smollm2": "llama",
    "minicpm": "qwen2",
    "minicpm3": "qwen2",
    "nemotron": "llama",
    "mpt": "llama",       # MPT uses same basic structure
    "gpt2": "llama",      # GPT-2 style → treat as basic
    "stablelm": "llama",  # StableLM is LLaMA-like
    "yi": "llama",        # Yi is LLaMA-like
    "olmo": "llama",      # OLMo is LLaMA-like
    "baichuan": "llama",
    "bloom": "llama",
}

# Map HuggingFace model_type to our architecture
HF_ARCH_MAP = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2",
    "qwen3": "qwen2",
    "qwen35": "qwen35",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "mistral": "mistral",
    "mixtral": "mistral",
    "phi3": "phi3",
    "phi4": "phi3",
    "phi": "phi3",
    "starcoder2": "starcoder2",
    "falcon": "falcon",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseek_r1": "deepseek",
    "internlm2": "internlm2",
    "cohere": "command-r",
    "smollm": "llama",
    "smollm2": "llama",
    "minicpm": "qwen2",
    "minicpm3": "qwen2",
    "nemotron": "llama",
    "gpt2": "llama",
    "gpt_neox": "llama",
    "mpt": "llama",
    "stablelm": "llama",
    "olmo": "llama",
    "baichuan": "llama",
    "bloom": "llama",
}


# ═══════════════════════════════════════════════════════════════════
# GGUF METADATA READER (lightweight — just header, no tensor data)
# ═══════════════════════════════════════════════════════════════════

GGUF_MAGIC = 0x46554747

GV_READERS = {
    0: lambda f: struct.unpack('<B', f.read(1))[0],    # uint8
    1: lambda f: struct.unpack('<b', f.read(1))[0],    # int8
    2: lambda f: struct.unpack('<H', f.read(2))[0],    # uint16
    3: lambda f: struct.unpack('<h', f.read(2))[0],    # int16
    4: lambda f: struct.unpack('<I', f.read(4))[0],    # uint32
    5: lambda f: struct.unpack('<i', f.read(4))[0],    # int32
    6: lambda f: struct.unpack('<f', f.read(4))[0],    # float32
    7: lambda f: struct.unpack('<B', f.read(1))[0],    # bool
    10: lambda f: struct.unpack('<Q', f.read(8))[0],   # uint64
    11: lambda f: struct.unpack('<q', f.read(8))[0],   # int64
    12: lambda f: struct.unpack('<d', f.read(8))[0],   # float64
}


def _read_gguf_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def _read_gguf_value(f, vtype):
    if vtype == 8:  # string
        return _read_gguf_string(f)
    elif vtype == 9:  # array
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    elif vtype in GV_READERS:
        return GV_READERS[vtype](f)
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf_metadata(filepath: str) -> Dict[str, Any]:
    """Read all KV metadata from a GGUF file. Returns flat dict."""
    metadata = {}
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {filepath}")
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        metadata['__version'] = version
        metadata['__n_tensors'] = n_tensors
        metadata['__n_kv'] = n_kv
        
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = _read_gguf_value(f, vtype)
            metadata[key] = value
    
    return metadata


# ═══════════════════════════════════════════════════════════════════
# CONFIG FROM GGUF
# ═══════════════════════════════════════════════════════════════════

def config_from_gguf(filepath: str) -> ModelConfig:
    """Extract full ModelConfig from GGUF file metadata."""
    md = read_gguf_metadata(filepath)
    
    # Detect architecture
    raw_arch = md.get('general.architecture', 'llama')
    arch_name = GGUF_ARCH_MAP.get(raw_arch, 'llama')
    defaults = ARCH_DEFAULTS.get(arch_name, ARCH_DEFAULTS['llama'])
    
    # Start with defaults
    cfg = ModelConfig(
        name=md.get('general.name', Path(filepath).stem),
        architecture=arch_name,
        source="gguf_metadata",
    )
    
    # Apply architecture defaults
    for key, val in defaults.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    
    # Extract from GGUF metadata (architecture-prefixed keys)
    prefix = f"{raw_arch}."
    
    # Core dims
    cfg.dim = md.get(f'{prefix}embedding_length', cfg.dim)
    cfg.n_layers = md.get(f'{prefix}block_count', cfg.n_layers)
    cfg.n_heads = md.get(f'{prefix}attention.head_count', cfg.n_heads)
    cfg.n_kv_heads = md.get(f'{prefix}attention.head_count_kv', cfg.n_heads)
    cfg.vocab_size = md.get(f'{prefix}vocab_size', 
                    md.get('tokenizer.ggml.vocab_size', cfg.vocab_size))
    
    # Head dim (compute or override)
    head_dim_key = md.get(f'{prefix}attention.key_length', 0)
    if head_dim_key > 0:
        cfg.head_dim = head_dim_key
    elif cfg.n_heads > 0:
        cfg.head_dim = cfg.dim // cfg.n_heads
    
    # FFN
    cfg.ffn_dim = md.get(f'{prefix}feed_forward_length', cfg.ffn_dim)
    
    # Context
    cfg.max_seq_len = md.get(f'{prefix}context_length', cfg.max_seq_len)
    
    # RoPE
    rope_dim = md.get(f'{prefix}rope.dimension_count', 0)
    cfg.rope_theta = md.get(f'{prefix}rope.freq_base', cfg.rope_theta)
    
    # Rope scaling
    rope_type = md.get(f'{prefix}rope.scaling.type', None)
    if rope_type:
        cfg.rope_scaling = {
            'type': rope_type,
            'factor': md.get(f'{prefix}rope.scaling.factor', 1.0),
        }
    
    # Norms
    norm_eps = md.get(f'{prefix}attention.layer_norm_rms_epsilon',
              md.get(f'{prefix}attention.layer_norm_epsilon', None))
    if norm_eps:
        cfg.norm_eps = norm_eps
    
    # Sliding window
    sw = md.get(f'{prefix}attention.sliding_window', None)
    if sw and sw > 0:
        cfg.sliding_window = sw
    
    # Softcap
    softcap = md.get(f'{prefix}attn_logit_softcapping', None)
    if softcap:
        cfg.logit_softcap = softcap
    
    # Embedding scale (Gemma)
    if arch_name in ('gemma', 'gemma2') or 'gemma' in raw_arch:
        cfg.embedding_scale = cfg.dim ** 0.5
    
    # Tokenizer IDs
    cfg.bos_token_id = md.get('tokenizer.ggml.bos_token_id', cfg.bos_token_id)
    cfg.eos_token_id = md.get('tokenizer.ggml.eos_token_id', cfg.eos_token_id)
    cfg.pad_token_id = md.get('tokenizer.ggml.padding_token_id', cfg.pad_token_id)
    
    # Vocab from tokenizer tokens array length
    tokens = md.get('tokenizer.ggml.tokens', [])
    if tokens and len(tokens) > cfg.vocab_size:
        cfg.vocab_size = len(tokens)
    
    # Tie embeddings check
    cfg.tie_word_embeddings = md.get(f'{prefix}tie_word_embeddings', 
                                     defaults.get('tie_word_embeddings', True))
    
    return cfg


# ═══════════════════════════════════════════════════════════════════
# CONFIG FROM HUGGINGFACE
# ═══════════════════════════════════════════════════════════════════

def config_from_huggingface(model_id: str) -> Optional[ModelConfig]:
    """Fetch config.json from HuggingFace and build ModelConfig."""
    if not HAS_REQUESTS:
        return None
    
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        hf = resp.json()
    except Exception:
        return None
    
    # Detect architecture
    model_type = hf.get('model_type', 'llama')
    arch_name = HF_ARCH_MAP.get(model_type, 'llama')
    defaults = ARCH_DEFAULTS.get(arch_name, ARCH_DEFAULTS['llama'])
    
    cfg = ModelConfig(
        name=model_id.split('/')[-1] if '/' in model_id else model_id,
        architecture=arch_name,
        source="huggingface",
    )
    
    # Apply defaults
    for key, val in defaults.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    
    # Map HF config fields → ModelConfig
    cfg.dim = hf.get('hidden_size', cfg.dim)
    cfg.n_layers = hf.get('num_hidden_layers', cfg.n_layers)
    cfg.n_heads = hf.get('num_attention_heads', cfg.n_heads)
    cfg.n_kv_heads = hf.get('num_key_value_heads', cfg.n_heads)
    cfg.vocab_size = hf.get('vocab_size', cfg.vocab_size)
    cfg.ffn_dim = hf.get('intermediate_size', cfg.ffn_dim)
    cfg.max_seq_len = hf.get('max_position_embeddings', cfg.max_seq_len)
    cfg.rope_theta = hf.get('rope_theta', cfg.rope_theta)
    cfg.norm_eps = hf.get('rms_norm_eps', hf.get('layer_norm_eps', cfg.norm_eps))
    cfg.tie_word_embeddings = hf.get('tie_word_embeddings', cfg.tie_word_embeddings)
    cfg.bos_token_id = hf.get('bos_token_id', cfg.bos_token_id) or 1
    cfg.eos_token_id = hf.get('eos_token_id', cfg.eos_token_id) or 2
    cfg.pad_token_id = hf.get('pad_token_id', cfg.pad_token_id) or 0
    
    # Head dim
    cfg.head_dim = hf.get('head_dim', cfg.dim // max(cfg.n_heads, 1))
    
    # Attention bias
    cfg.attn_bias = hf.get('attention_bias', hf.get('bias', defaults.get('attn_bias', False)))
    
    # Sliding window
    sw = hf.get('sliding_window', None)
    if sw and sw > 0:
        cfg.sliding_window = sw
    
    # Logit softcap (Gemma 2)
    cfg.logit_softcap = hf.get('attn_logit_softcapping', 
                        hf.get('final_logit_softcapping', None))
    
    # QK norm
    cfg.qk_norm = hf.get('query_pre_attn_scalar', None) is not None
    
    # Embedding scale
    if arch_name in ('gemma', 'gemma2'):
        cfg.embedding_scale = cfg.dim ** 0.5
    
    # Rope scaling
    rope_cfg = hf.get('rope_scaling', None)
    if rope_cfg:
        cfg.rope_scaling = rope_cfg
    
    # FFN type from activation
    act = hf.get('hidden_act', hf.get('activation_function', 'silu'))
    if act in ('silu', 'swiglu'):
        cfg.ffn_type = 'swiglu'
    elif act in ('gelu', 'gelu_new', 'gelu_pytorch_tanh'):
        cfg.ffn_type = 'gelu'
    elif act == 'relu':
        cfg.ffn_type = 'relu'
    
    # Chat template detection from model ID
    model_lower = model_id.lower()
    if 'instruct' in model_lower or 'chat' in model_lower or 'it' in model_lower:
        if 'qwen' in model_lower:
            cfg.chat_template = 'chatml'
        elif 'gemma' in model_lower:
            cfg.chat_template = 'gemma'
        elif 'mistral' in model_lower:
            cfg.chat_template = 'mistral'
        elif 'llama' in model_lower:
            cfg.chat_template = 'llama3'
        elif 'phi' in model_lower:
            cfg.chat_template = 'phi3'
    
    return cfg


# ═══════════════════════════════════════════════════════════════════
# AUTO-DETECT (try GGUF first, then HuggingFace, then guess)
# ═══════════════════════════════════════════════════════════════════

def auto_detect_config(path_or_id: str) -> ModelConfig:
    """
    Auto-detect model configuration from any source:
    - Local GGUF file → read metadata
    - HuggingFace model ID (org/model) → fetch config.json
    - Filename with known patterns → guess architecture
    """
    path = Path(path_or_id)
    
    # 1. Local GGUF file
    if path.exists() and path.suffix.lower() == '.gguf':
        return config_from_gguf(str(path))
    
    # 2. HuggingFace model ID
    if '/' in path_or_id and not path.exists():
        cfg = config_from_huggingface(path_or_id)
        if cfg:
            return cfg
    
    # 3. Try to guess from filename
    cfg = _guess_from_filename(path_or_id)
    return cfg


def _guess_from_filename(name: str) -> ModelConfig:
    """Last resort — guess architecture from filename patterns."""
    lower = name.lower()
    cfg = ModelConfig(name=name, source="filename_guess")
    
    # Architecture detection from name
    if 'qwen' in lower:
        cfg.architecture = 'qwen2'
        cfg.rope_theta = 1000000.0
        cfg.norm_eps = 1e-6
        cfg.attn_bias = True
    elif 'gemma' in lower:
        cfg.architecture = 'gemma'
        cfg.norm_eps = 1e-6
    elif 'mistral' in lower or 'mixtral' in lower:
        cfg.architecture = 'mistral'
    elif 'phi' in lower:
        cfg.architecture = 'phi3'
    elif 'deepseek' in lower:
        cfg.architecture = 'deepseek'
    elif 'falcon' in lower:
        cfg.architecture = 'falcon'
    elif 'command' in lower or 'cohere' in lower:
        cfg.architecture = 'command-r'
    else:
        cfg.architecture = 'llama'  # default
    
    # Size detection from name
    size_match = re.search(r'(\d+\.?\d*)[bB]', lower)
    if size_match:
        size_b = float(size_match.group(1))
        if size_b <= 0.5:
            cfg.dim, cfg.n_layers, cfg.n_heads = 512, 8, 8
        elif size_b <= 1:
            cfg.dim, cfg.n_layers, cfg.n_heads = 1024, 16, 16
        elif size_b <= 2:
            cfg.dim, cfg.n_layers, cfg.n_heads = 1536, 28, 12
        elif size_b <= 4:
            cfg.dim, cfg.n_layers, cfg.n_heads = 2048, 32, 16
        elif size_b <= 8:
            cfg.dim, cfg.n_layers, cfg.n_heads = 4096, 32, 32
        elif size_b <= 14:
            cfg.dim, cfg.n_layers, cfg.n_heads = 5120, 40, 40
        elif size_b <= 35:
            cfg.dim, cfg.n_layers, cfg.n_heads = 6656, 60, 52
        else:
            cfg.dim, cfg.n_layers, cfg.n_heads = 8192, 80, 64
        
        cfg.n_kv_heads = max(1, cfg.n_heads // 4)  # assume GQA
        cfg.head_dim = cfg.dim // cfg.n_heads
        cfg.ffn_dim = int(cfg.dim * 2.67)  # typical SwiGLU ratio
    
    # Apply arch defaults
    defaults = ARCH_DEFAULTS.get(cfg.architecture, {})
    for key, val in defaults.items():
        if hasattr(cfg, key) and getattr(cfg, key) == getattr(ModelConfig(), key):
            setattr(cfg, key, val)
    
    return cfg


# ═══════════════════════════════════════════════════════════════════
# CHAT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

CHAT_TEMPLATES = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "stop": ["<|im_end|>"],
    },
    "llama3": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "stop": ["<|eot_id|>"],
    },
    "gemma": {
        "system": "",
        "user": "<start_of_turn>user\n{content}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{content}<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
        "stop": ["<end_of_turn>"],
    },
    "mistral": {
        "system": "",
        "user": "[INST] {content} [/INST]",
        "assistant": "{content}</s>",
        "assistant_start": "",
        "stop": ["</s>"],
    },
    "phi3": {
        "system": "<|system|>\n{content}<|end|>\n",
        "user": "<|user|>\n{content}<|end|>\n",
        "assistant": "<|assistant|>\n{content}<|end|>\n",
        "assistant_start": "<|assistant|>\n",
        "stop": ["<|end|>"],
    },
    "deepseek": {
        "system": "",
        "user": "User: {content}\n\n",
        "assistant": "Assistant: {content}\n\n",
        "assistant_start": "Assistant: ",
        "stop": ["\nUser:"],
    },
    "alpaca": {
        "system": "",
        "user": "### Instruction:\n{content}\n\n### Response:\n",
        "assistant": "{content}\n\n",
        "assistant_start": "### Response:\n",
        "stop": ["### Instruction:"],
    },
}


def format_prompt(cfg: ModelConfig, system: str, user: str) -> str:
    """Format a chat prompt using the model's template."""
    template_name = cfg.chat_template or "chatml"
    template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["chatml"])
    
    parts = []
    if system and template["system"]:
        parts.append(template["system"].format(content=system))
    parts.append(template["user"].format(content=user))
    parts.append(template["assistant_start"])
    
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_config.py <path.gguf | org/model_id>")
        print("\nExample:")
        print("  python model_config.py ./codegemma-7b-it-Q5_K_M.gguf")
        print("  python model_config.py google/gemma-2-9b-it")
        print("  python model_config.py Qwen/Qwen2.5-1.5B")
        sys.exit(0)
    
    cfg = auto_detect_config(sys.argv[1])
    print(f"\n{'='*60}")
    print(f"  Model: {cfg.name}")
    print(f"  Architecture: {cfg.architecture}")
    print(f"  Source: {cfg.source}")
    print(f"{'='*60}")
    print(f"  dim={cfg.dim}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    print(f"  kv_heads={cfg.n_kv_heads}, head_dim={cfg.head_dim}")
    print(f"  vocab={cfg.vocab_size}, ffn={cfg.ffn_dim}")
    print(f"  max_seq={cfg.max_seq_len}, rope_θ={cfg.rope_theta}")
    print(f"  norm={cfg.norm_type} (eps={cfg.norm_eps})")
    print(f"  ffn_type={cfg.ffn_type}, attn_bias={cfg.attn_bias}")
    print(f"  tie_embeddings={cfg.tie_word_embeddings}")
    print(f"  chat_template={cfg.chat_template}")
    if cfg.sliding_window:
        print(f"  sliding_window={cfg.sliding_window}")
    if cfg.logit_softcap:
        print(f"  logit_softcap={cfg.logit_softcap}")
    if cfg.embedding_scale:
        print(f"  embedding_scale={cfg.embedding_scale}")
    print(f"{'='*60}\n")
