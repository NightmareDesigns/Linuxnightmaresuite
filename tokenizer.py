# === FORCE CUDA TO SEE BOTH GPUs ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
_NIGHTMARE_DEBUG = os.environ.get("NIGHTMARE_DEBUG", "1") == "1"

"""
BPE Tokenizer — From Scratch
Reads vocabulary and merge rules from GGUF metadata.
Handles encoding (text -> tokens) and decoding (tokens -> text).

Supports the tokenizer formats embedded in GGUF files:
- SentencePiece (LLaMA, Mistral)
- BPE (GPT-style)

CRITICAL FIX (v5.1):
  - Special tokens (control tokens like <|begin_of_text|>, <start_of_turn>, etc.)
    are now recognized BEFORE byte-level encoding. This prevents them from being
    split into garbage byte sequences.
"""

import re
from typing import Optional



def _bytes_to_unicode():
    """
    GPT-2 byte-to-unicode mapping.
    
    Maps bytes 0-255 to unicode characters used in the BPE vocab.
    This is how GPT-2/LLaMA3 tokenizers represent raw bytes as unicode strings
    in their vocabulary. For example:
      - byte 0x20 (space) → 'Ġ' (U+0120)
      - byte 0x0A (newline) → 'Ċ' (U+010A)
      - printable ASCII stays as-is: 'a' → 'a'
    
    Without this mapping, byte-level BPE won't find tokens in the vocab.
    """
    # Printable bytes map to themselves
    bs = list(range(ord("!"), ord("~") + 1))  # 33-126
    bs += list(range(ord("¡"), ord("¬") + 1))  # 161-172
    bs += list(range(ord("®"), ord("ÿ") + 1))  # 174-255
    
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tokenizer:
    """
    Byte-Pair Encoding tokenizer that reads its vocab from GGUF metadata.
    
    GGUF stores:
      - tokenizer.ggml.tokens: list of token strings
      - tokenizer.ggml.scores: token scores/priorities
      - tokenizer.ggml.token_type: token types (normal, control, unknown, etc.)
      - tokenizer.ggml.merges: BPE merge rules (optional for SPM models)
      - tokenizer.ggml.bos_token_id: beginning-of-sequence token
      - tokenizer.ggml.eos_token_id: end-of-sequence token
    """
    
    # Token types from GGUF spec
    TOKEN_TYPE_NORMAL = 1
    TOKEN_TYPE_UNKNOWN = 2
    TOKEN_TYPE_CONTROL = 3
    TOKEN_TYPE_USER_DEFINED = 4
    TOKEN_TYPE_UNUSED = 5
    TOKEN_TYPE_BYTE = 6
    
    def __init__(self, metadata: dict, model_path: str = None):
        """Initialize tokenizer from GGUF metadata dict.
        If GGUF has no tokenizer, falls back to tokenizer.json beside model file."""
        self.tokens = metadata.get("tokenizer.ggml.tokens", [])
        self.scores = metadata.get("tokenizer.ggml.scores", [])
        self.token_types = metadata.get("tokenizer.ggml.token_type", [])
        self.merges = metadata.get("tokenizer.ggml.merges", [])
        
        # Sanity check: if >90% of tokens are marked CONTROL (type 3),
        # the token_types data is clearly wrong (bad export). Ignore it.
        if self.token_types and self.tokens:
            n_control = sum(1 for t in self.token_types if t == self.TOKEN_TYPE_CONTROL)
            if n_control > len(self.tokens) * 0.9:
                print(f"[Tokenizer] WARNING: {n_control}/{len(self.tokens)} tokens marked CONTROL "
                      f"— token_types data is corrupt, ignoring")
                self.token_types = []  # Clear — fallback to type 1 (NORMAL) for all
            elif n_control > 1000:
                print(f"[Tokenizer] Note: {n_control} control tokens registered")
        
        # Special token IDs
        self.bos_id = metadata.get("tokenizer.ggml.bos_token_id", 1)
        self.eos_id = metadata.get("tokenizer.ggml.eos_token_id", 2)
        self.pad_id = metadata.get("tokenizer.ggml.padding_token_id", 0)
        self.unk_id = metadata.get("tokenizer.ggml.unknown_token_id", 0)
        declared_type = metadata.get("tokenizer.ggml.model", "")

        # Whether a BOS should be prepended is a property of the MODEL, not a
        # free choice. GGUF records it as tokenizer.ggml.add_bos_token.
        #
        # Prepending BOS unconditionally is not a harmless default: on a base
        # model whose flag is false it prepends a stray <|endoftext|> and
        # destroys the next-token distribution. Measured on
        # codeqwen-1_5-7b-q4_0.gguf for the prompt "The capital of France is":
        #
        #   with forced BOS : top-1 logprob -2.6888  (p=0.068)  '``'
        #   honouring flag  : top-1 logprob -0.0198  (p=0.980)  ' '
        #
        # and the honouring-flag run matches llama.cpp's logprobs token for
        # token (-0.0198 vs -0.0169, then -6.11/-6.21, -6.73/-6.86, ...).
        # Absent the key, fall back to True, which is what instruct-tuned
        # llama-family GGUFs overwhelmingly use.
        _ab = metadata.get("tokenizer.ggml.add_bos_token", None)
        if isinstance(_ab, dict):
            _ab = _ab.get("value")
        self.add_bos_token = True if _ab is None else bool(_ab)
        
        # --- NTMRE / RMTN container: the engine already parsed the embedded
        # tokenizer JSON (tokens + id-pair merges + specials).  Nothing
        # populates `metadata` on that path, so without this every container
        # model loaded with 0 tokens, model_type "llama", and encoded to
        # [bos, 0, 0, 0, ...].
        if not self.tokens and model_path:
            try:
                import json as _json
                from ntmre_v1_engine import V1Source as _V1
                _src = _V1(model_path)
                _tb = getattr(_src, "tokenizer_bytes", None)
                if _tb:
                    d = _json.loads(_tb.decode('utf-8'))
                    self.tokens = d.get('tokens', [])
                    self.scores = d.get('scores', [])
                    self.merges = d.get('merges', [])
                    self.bos_id = int(d.get('bos', self.bos_id))
                    self.eos_id = int(d.get('eos', self.eos_id))
                    self.pad_id = int(d.get('pad', self.pad_id))
                    self.unk_id = int(d.get('unk', self.unk_id))
                    declared_type = declared_type or d.get('model', '')
                    print(f"[Tokenizer] Loaded {len(self.tokens)} tokens from "
                          f"NTMRE container")
                _src.close()
            except Exception as e:
                if _NIGHTMARE_DEBUG:
                    print(f"[Tokenizer] NTMRE container load skipped: {e}")

        # --- FALLBACK: Load tokenizer.json if GGUF has no tokenizer data ---
        if not self.tokens and model_path:
            import json
            from pathlib import Path
            tok_dir = Path(model_path).parent
            tok_json = tok_dir / "tokenizer.json"
            if not tok_json.exists():
                # Also check inside the model filename minus extension as folder
                tok_json = tok_dir / "tokenizer.json"
            if tok_json.exists():
                print(f"[Tokenizer] Loading from: {tok_json}")
                with open(tok_json, 'r', encoding='utf-8') as f:
                    tok_data = json.load(f)
                # Extract vocab: model.vocab is {token: id}
                vocab_dict = {}
                if 'model' in tok_data and 'vocab' in tok_data['model']:
                    vocab_dict = tok_data['model']['vocab']
                elif 'vocab' in tok_data:
                    vocab_dict = tok_data['vocab']
                # Sort by ID to get ordered token list
                if vocab_dict:
                    max_id = max(vocab_dict.values())
                    self.tokens = [''] * (max_id + 1)
                    for tok_str, tok_id in vocab_dict.items():
                        self.tokens[tok_id] = tok_str
                # Extract merges
                if 'model' in tok_data and 'merges' in tok_data['model']:
                    self.merges = tok_data['model']['merges']
                # Extract special tokens for BOS/EOS
                if 'added_tokens' in tok_data:
                    for t in tok_data['added_tokens']:
                        if t.get('content') in ('<|endoftext|>', '</s>', '<eos>'):
                            self.eos_id = t['id']
                        if t.get('content') in ('<|startoftext|>', '<s>', '<bos>'):
                            self.bos_id = t['id']
                print(f"[Tokenizer] Loaded {len(self.tokens)} tokens from tokenizer.json")
        
        # Model type — auto-detect from merges presence
        if self.merges:
            self.model_type = "bpe"
        elif declared_type:
            self.model_type = declared_type
        else:
            self.model_type = "llama"  # SPM fallback for Nightmare native models
        
        # Build lookup tables
        self.vocab_size = len(self.tokens)
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}

        # Vocabulary-space detection.
        #
        # The real bug was the branch above: `if self.merges: model_type="bpe"`
        # overrode tokenizer.ggml.model. gemma4 ships 514906 merge rules next
        # to a metaspace vocab ('▁is'), so it was declared "bpe" and the
        # GPT-2 byte-level encoder ran against a metaspace vocab -- where
        # 'Ġis' can never match '▁is'. Every token containing a space became
        # unreachable, " is" came out as bare byte-32 plus bare "is", and
        # every gemma4 prompt reached the model with a spurious space token
        # wedged between each word.
        #
        # The declared type is the only trustworthy signal. Sniffing the vocab
        # is NOT: codeqwen-1_5-7b declares "gpt2", carries 28617 '▁' tokens,
        # and really is byte-level -- its oracle tokenizes "What is the" as
        # [5577, 81858, 1660, 81858] with the space as its own token. Measured
        # against llama.cpp /tokenize on that file:
        #     _encode_bpe  96/96 exact
        #     _encode_spm   0/96
        # Routing it on vocab contents would have broken a working model.
        _decl = str(declared_type or '').lower()
        self.is_metaspace = _decl in ('llama', 'spm', 'gemma', 'gemma2',
                                      'gemma3', 'gemma3n', 'gemma4', 'gemma4n')

        # Gemma ships NO dummy prefix. Verified against llama.cpp's /tokenize:
        # with a leading space prepended the first token of every prompt comes
        # out as a different id ('▁What'=2900 vs 'What'=3689). Llama-family
        # vocabs do want it, so key this off the declared type.
        self.add_dummy_prefix = not _decl.startswith('gemma')
        
        # Build byte-to-token mapping for byte fallback
        self.byte_tokens = {}
        for i, tok in enumerate(self.tokens):
            if i < len(self.token_types) and self.token_types[i] == self.TOKEN_TYPE_BYTE:
                # Byte tokens are like <0x00>, <0x01>, etc.
                if tok.startswith("<0x") and tok.endswith(">"):
                    byte_val = int(tok[3:-1], 16)
                    self.byte_tokens[byte_val] = i
        
        # Build special/control token set for fast lookup during encoding
        # These are tokens that should be matched AS-IS, not decomposed into bytes.
        self.special_tokens = {}
        for i, tok in enumerate(self.tokens):
            if i < len(self.token_types):
                # Control tokens + any token containing special markers
                if self.token_types[i] == self.TOKEN_TYPE_CONTROL:
                    self.special_tokens[tok] = i
            # Also catch tokens that look like special markers (common patterns)
            if (tok.startswith("<|") and tok.endswith("|>")) or\
               (tok.startswith("<") and tok.endswith(">") and "_of_" in tok) or\
               tok in ("<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>",
                       "</s>", "<s>", "<pad>", "<unk>"):
                self.special_tokens[tok] = i
        
        if _NIGHTMARE_DEBUG:
            print(f"[Tokenizer DEBUG] {len(self.special_tokens)} special tokens registered")
        
        # Build merge priority (for BPE encoding)
        #
        # Merge rules come in two shapes and conflating them silently
        # disables BPE entirely:
        #   "tok1 tok2"      - GGUF tokenizer.ggml.merges, token TEXT
        #   [id1, id2]       - NTMRE v13 container, token INDICES
        #
        # _encode_bpe() looks up `pieces[i] + " " + pieces[i+1]`, which is
        # built from byte-encoded text. Stringifying an id pair produced
        # "220 83", which never matches any text pair, so no merge ever
        # fired and every word was emitted one byte at a time
        # ("The capital of France is" -> 25 tokens instead of 6).
        # Resolve id pairs back to token text so both shapes work.
        self.merge_ranks = {}
        _id_pairs = 0
        for rank, merge in enumerate(self.merges):
            if isinstance(merge, (list, tuple)):
                if len(merge) != 2:
                    continue
                a, b = merge
                if isinstance(a, int) and isinstance(b, int):
                    _id_pairs += 1
                    ta = self.tokens[a] if 0 <= a < len(self.tokens) else None
                    tb = self.tokens[b] if 0 <= b < len(self.tokens) else None
                    if ta is None or tb is None:
                        continue
                    merge = ta + " " + tb
                else:
                    merge = ' '.join(str(m) for m in merge)
            if isinstance(merge, str):
                self.merge_ranks[merge] = rank
        if _id_pairs and _NIGHTMARE_DEBUG:
            print(f"[Tokenizer] resolved {_id_pairs} id-pair merges to token text")
        
        # GPT-2 byte encoder/decoder (for BPE tokenizers like LLaMA 3)
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        
        print(f"[Tokenizer] Loaded {self.vocab_size} tokens, model type: {self.model_type}")
        print(f"[Tokenizer] BOS={self.bos_id}, EOS={self.eos_id}")
        if self.merges:
            print(f"[Tokenizer] {len(self.merges)} BPE merge rules loaded")
    
    def encode(self, text: str, add_bos: bool = None) -> list:
        """
        Encode text to token IDs.
        
        For SentencePiece-style (LLaMA):
          - Prepend space to text
          - Use greedy longest-match with scores
          
        For BPE-style:
          - Split by regex pattern
          - Apply BPE merges iteratively
        
        Special tokens embedded in the text (like <|eot_id|>) are matched
        first and preserved as single tokens.

        `add_bos` defaults to the model's own tokenizer.ggml.add_bos_token
        flag rather than always prepending. Pass True/False to override.
        """
        if add_bos is None:
            add_bos = getattr(self, 'add_bos_token', True)
        if self.model_type in ("llama", "spm") or self.is_metaspace:
            return self._encode_spm(text, add_bos)
        else:
            return self._encode_bpe(text, add_bos)
    
    def _split_on_special_tokens(self, text: str) -> list:
        """
        Split text on special tokens, preserving them.
        Returns list of (chunk, is_special) tuples.
        
        Example: "Hello<|eot_id|>World" → [("Hello", False), ("<|eot_id|>", True), ("World", False)]
        """
        if not self.special_tokens:
            return [(text, False)]
        
        # Sort special tokens by length (longest first) for greedy matching
        sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
        
        result = []
        remaining = text
        while remaining:
            found = False
            for special in sorted_specials:
                if remaining.startswith(special):
                    result.append((special, True))
                    remaining = remaining[len(special):]
                    found = True
                    break
            if not found:
                # Find next special token or end of string
                next_pos = len(remaining)
                for special in sorted_specials:
                    pos = remaining.find(special)
                    if pos != -1 and pos < next_pos:
                        next_pos = pos
                result.append((remaining[:next_pos], False))
                remaining = remaining[next_pos:]
        
        return [(chunk, is_sp) for chunk, is_sp in result if chunk]
    
    def _encode_spm(self, text: str, add_bos: bool) -> list:
        """
        SentencePiece encoding (used by LLaMA, Mistral).
        
        Algorithm:
        1. Prepend space (SentencePiece convention: ▁ = word boundary)
        2. Convert to bytes
        3. Greedy tokenization: repeatedly find the highest-score merge
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        
        # Split on special tokens first — encode regular text separately
        chunks = self._split_on_special_tokens(text)
        
        for chunk_idx, (chunk, is_special) in enumerate(chunks):
            if is_special:
                # Direct lookup — don't encode through byte-level
                if chunk in self.special_tokens:
                    tokens.append(self.special_tokens[chunk])
                elif chunk in self.token_to_id:
                    tokens.append(self.token_to_id[chunk])
                if _NIGHTMARE_DEBUG:
                    print(f"[Tokenizer DEBUG] Special token: '{chunk}' → id={tokens[-1]}")
                continue
            
            # Regular text — encode with SentencePiece algorithm
            tokens.extend(self._encode_spm_chunk(
                chunk, prepend_space=(chunk_idx == 0) and self.add_dummy_prefix))
        
        if _NIGHTMARE_DEBUG:
            print(f"[Tokenizer DEBUG] Encoded {len(tokens)} tokens (add_bos={add_bos})")
            if len(tokens) <= 50:
                print(f"[Tokenizer DEBUG] Token IDs: {tokens}")
        
        return tokens
    
    def _encode_spm_chunk(self, text: str, prepend_space: bool = True) -> list:
        """Encode a chunk of regular (non-special) text using SentencePiece algorithm."""
        if not text:
            return []
        
        pieces = []
        
        # SentencePiece prepends a space to the first chunk
        if prepend_space:
            text = " " + text
        
        text_bytes = text.encode("utf-8")
        
        i = 0
        while i < len(text_bytes):
            best_token = None
            best_len = 0
            
            for end in range(min(i + 64, len(text_bytes)), i, -1):
                candidate = text_bytes[i:end]
                candidate_str = candidate.decode("utf-8", errors="replace").replace(" ", "▁")
                
                if candidate_str in self.token_to_id:
                    token_id = self.token_to_id[candidate_str]
                    if best_token is None or end - i > best_len:
                        best_token = token_id
                        best_len = end - i
                    break
            
            if best_token is not None:
                pieces.append(best_token)
                i += best_len
            else:
                byte_val = text_bytes[i]
                if byte_val in self.byte_tokens:
                    pieces.append(self.byte_tokens[byte_val])
                else:
                    pieces.append(self.unk_id)
                i += 1
        
        # Iteratively merge pieces using score-based priority
        while len(pieces) > 1:
            best_merge_idx = -1
            best_score = -float("inf")
            
            for idx in range(len(pieces) - 1):
                # What would the merged token be?
                left = self.tokens[pieces[idx]] if pieces[idx] < len(self.tokens) else ""
                right = self.tokens[pieces[idx + 1]] if pieces[idx + 1] < len(self.tokens) else ""
                merged = left + right
                
                if merged in self.token_to_id:
                    merged_id = self.token_to_id[merged]
                    score = self.scores[merged_id] if merged_id < len(self.scores) else 0
                    if score > best_score:
                        best_score = score
                        best_merge_idx = idx
            
            if best_merge_idx == -1:
                break
            
            left = self.tokens[pieces[best_merge_idx]]
            right = self.tokens[pieces[best_merge_idx + 1]]
            merged_id = self.token_to_id[left + right]
            pieces = pieces[:best_merge_idx] + [merged_id] + pieces[best_merge_idx + 2:]
        
        return pieces
    
    def _encode_bpe(self, text: str, add_bos: bool) -> list:
        """
        Standard BPE encoding (GPT-style).
        
        Algorithm:
        1. Split on special tokens first (preserve <|...|> as single IDs)
        2. For regular text: split into words by regex
        3. For each word, start with byte-level characters
        4. Repeatedly merge the highest-priority pair
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        
        # Split on special tokens first — they must not be byte-encoded
        chunks = self._split_on_special_tokens(text)
        
        for chunk, is_special in chunks:
            if is_special:
                # Direct lookup — single token ID
                if chunk in self.special_tokens:
                    tokens.append(self.special_tokens[chunk])
                elif chunk in self.token_to_id:
                    tokens.append(self.token_to_id[chunk])
                continue
            
            # Regular text — BPE encode
            pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""
            words = re.findall(pattern, chunk)
            
            for word in words:
                # Convert word to bytes and then to byte-level tokens
                word_bytes = word.encode("utf-8")
                pieces = [self._byte_encoder[b] for b in word_bytes]
                
                # Iteratively merge using merge rules
                while len(pieces) > 1:
                    best_pair = None
                    best_rank = float("inf")
                    
                    for i in range(len(pieces) - 1):
                        pair = pieces[i] + " " + pieces[i + 1]
                        if pair in self.merge_ranks and self.merge_ranks[pair] < best_rank:
                            best_rank = self.merge_ranks[pair]
                            best_pair = i
                    
                    if best_pair is None:
                        break
                    
                    # Merge the pair
                    merged = pieces[best_pair] + pieces[best_pair + 1]
                    pieces = pieces[:best_pair] + [merged] + pieces[best_pair + 2:]
                
                # Look up IDs
                for piece in pieces:
                    if piece in self.token_to_id:
                        tokens.append(self.token_to_id[piece])
                    else:
                        tokens.append(self.unk_id)
        
        return tokens
    
    def decode(self, token_ids: list) -> str:
        """
        Decode token IDs back to text.
        
        Handles:
        - Normal tokens (just concatenate)
        - Byte tokens (<0xNN>) -> raw bytes
        - SentencePiece ▁ -> space
        - Skip control tokens (BOS, EOS)
        """
        pieces = []
        
        for token_id in token_ids:
            # Skip special control tokens
            if token_id in (self.bos_id, self.eos_id, self.pad_id):
                continue
            
            if token_id < 0 or token_id >= self.vocab_size:
                continue
            
            token_type = self.token_types[token_id] if token_id < len(self.token_types) else 1
            
            if token_type == self.TOKEN_TYPE_CONTROL:
                continue
            
            token_str = self.tokens[token_id]
            
            if token_type == self.TOKEN_TYPE_BYTE:
                # Byte token: <0xNN> -> raw byte
                if token_str.startswith("<0x") and token_str.endswith(">"):
                    byte_val = int(token_str[3:-1], 16)
                    pieces.append(bytes([byte_val]).decode("utf-8", errors="replace"))
            else:
                # Decode based on tokenizer type
                if self.model_type not in ("llama", "spm") and self._byte_decoder:
                    # BPE models (GPT-2, LLaMA 3, Qwen, etc): reverse byte-to-unicode mapping
                    try:
                        byte_list = [self._byte_decoder[c] for c in token_str]
                        pieces.append(bytes(byte_list).decode("utf-8", errors="replace"))
                    except KeyError:
                        pieces.append(token_str.replace("▁", " "))
                else:
                    # SentencePiece: replace ▁ with space
                    pieces.append(token_str.replace("▁", " "))
        
        result = "".join(pieces)
        
        # Strip leading space (SentencePiece adds one during encoding)
        if result.startswith(" "):
            result = result[1:]
        
        return result
    
    def decode_single(self, token_id: int) -> str:
        """Decode a single token (for streaming output).
        
        Unlike decode(), this does NOT strip the leading space —
        SentencePiece uses ▁ as a word boundary marker that must be
        preserved during streaming to maintain proper spacing.
        """
        if token_id in (self.bos_id, self.eos_id, self.pad_id):
            return ""
        if token_id < 0 or token_id >= self.vocab_size:
            return ""
        
        token_type = self.token_types[token_id] if token_id < len(self.token_types) else 1
        token_str = self.tokens[token_id]
        
        # Turn-control tokens (Gemma, Gemma 2, Gemma 4) need to be visible as
        # literal strings so the streaming stop-string check can catch them.
        # They are stripped from the final output by the stop-string logic.
        if token_str in ("<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>"):
            return token_str
        
        if token_type == self.TOKEN_TYPE_CONTROL:
            return ""
        
        if token_type == self.TOKEN_TYPE_BYTE:
            if token_str.startswith("<0x") and token_str.endswith(">"):
                byte_val = int(token_str[3:-1], 16)
                return bytes([byte_val]).decode("utf-8", errors="replace")
            return ""
        
        # For BPE models (GPT-2/LLaMA 3): reverse byte-to-unicode mapping
        # e.g. 'Ġ' (U+0120) → byte 0x20 → space character
        if self.model_type not in ("llama", "spm") and self._byte_decoder:
            try:
                byte_list = [self._byte_decoder[c] for c in token_str]
                return bytes(byte_list).decode("utf-8", errors="replace")
            except KeyError:
                # Fallback if any char not in byte decoder (shouldn't happen)
                return token_str.replace("▁", " ")
        
        # SentencePiece models: replace ▁ with space
        return token_str.replace("▁", " ")

    def get_token_bytes(self, token_id: int) -> bytes:
        """Return the raw bytes that a token contributes to the stream.

        Mirrors decode_single() but produces bytes instead of text so an
        incremental UTF-8 decoder can reassemble multi-byte characters that
        a byte-level tokenizer splits across two tokens.  Returns b'' for
        control / skipped tokens (BOS, EOS, PAD, control types) and for
        invalid ids.
        """
        if token_id in (self.bos_id, self.eos_id, self.pad_id):
            return b""
        if token_id < 0 or token_id >= self.vocab_size:
            return b""

        token_type = self.token_types[token_id] if token_id < len(self.token_types) else 1
        token_str = self.tokens[token_id]

        # Turn-control tokens (Gemma, Gemma 2, Gemma 4) are emitted literally
        # so the streaming stop-string check can catch them — the same bytes
        # decode_single() would produce as text.
        if token_str in ("<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>"):
            return token_str.encode("utf-8")

        if token_type == self.TOKEN_TYPE_CONTROL:
            return b""

        if token_type == self.TOKEN_TYPE_BYTE:
            if token_str.startswith("<0x") and token_str.endswith(">"):
                return bytes([int(token_str[3:-1], 16)])
            return b""

        # For BPE models (GPT-2/LLaMA 3): reverse byte-to-unicode mapping
        # e.g. 'Ġ' (U+0120) → byte 0x20 → space character
        if self.model_type not in ("llama", "spm") and self._byte_decoder:
            try:
                byte_list = [self._byte_decoder[c] for c in token_str]
                return bytes(byte_list)
            except KeyError:
                # Fallback if any char not in byte decoder (shouldn't happen)
                return token_str.replace("▁", " ").encode("utf-8")

        # SentencePiece models: replace ▁ with space
        return token_str.replace("▁", " ").encode("utf-8")
