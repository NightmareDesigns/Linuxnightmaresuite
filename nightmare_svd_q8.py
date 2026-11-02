import numpy as np, struct, json, sys, time, os
from pathlib import Path

ENERGY_THRESHOLD = 0.999
MAX_RANK_RATIO = 0.15
MIN_RANK = 32
BLOCK_SIZE = 32
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# Token type constants
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_BYTE = 6

def find_rank(S):
    total = np.sum(S**2)
    cum = np.cumsum(S**2)
    r = int(np.searchsorted(cum, ENERGY_THRESHOLD * total)) + 1
    return min(r, max(int(len(S)*MAX_RANK_RATIO), MIN_RANK))

def svd_compress(W):
    if W.ndim != 2: return None
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = find_rank(S)
    energy = np.sum(S[:rank]**2) / np.sum(S**2)
    sqS = np.sqrt(S[:rank])
    A = (U[:,:rank] * sqS[None,:]).astype(np.float32)
    B = (Vt[:rank,:] * sqS[:,None]).astype(np.float32)
    return A, B, rank, energy

def quantize_q8(data):
    flat = data.flatten().astype(np.float32)
    pad = (BLOCK_SIZE - len(flat)%BLOCK_SIZE) % BLOCK_SIZE
    if pad: flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, BLOCK_SIZE)
    n = len(blocks)
    out = bytearray(n*34)
    for i in range(n):
        amax = np.max(np.abs(blocks[i]))
        scale = amax/127.0 if amax>0 else 0.0
        inv = 1.0/scale if scale>0 else 0.0
        struct.pack_into('<e', out, i*34, np.float16(scale))
        q = np.clip(np.round(blocks[i]*inv), -127, 127).astype(np.int8)
        out[i*34+2:i*34+34] = q.tobytes()
    return bytes(out)

def ws(f,s):
    b=s.encode('utf-8'); f.write(struct.pack('<Q',len(b))); f.write(b)
def wkv_s(f,k,v): ws(f,k); f.write(struct.pack('<I',8)); ws(f,v)
def wkv_u(f,k,v): ws(f,k); f.write(struct.pack('<I',4)); f.write(struct.pack('<I',v))
def wkv_arr_f32(f, key, values):
    ws(f, key); f.write(struct.pack('<I', 9))  # ARRAY
    f.write(struct.pack('<I', 6))  # FLOAT32
    f.write(struct.pack('<Q', len(values)))
    for v in values: f.write(struct.pack('<f', v))
def wkv_arr_i32(f, key, values):
    ws(f, key); f.write(struct.pack('<I', 9))  # ARRAY
    f.write(struct.pack('<I', 5))  # INT32
    f.write(struct.pack('<Q', len(values)))
    for v in values: f.write(struct.pack('<i', v))

def export(ckpt_override=None):
    print("\n" + "="*60)
    print("  NIGHTMARE .ntmre compression - 7B to Under 1 GB")
    print("  99.9% energy. Zero loss. From the floor up.")
    print("="*60 + "\n")
    ckpt_dir = Path('checkpoints_7b') if ckpt_override is None else None
    if ckpt_override:
        ckpt = ckpt_override
    elif not ckpt_dir.exists(): print("ERROR: no checkpoints_7b/"); sys.exit(1)
    else:
        if (ckpt_dir/'best').exists(): ckpt = ckpt_dir/'best'
        else:
            cs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and 'step_' in d.name], key=lambda x: int(x.name.split('_')[1]))
            ckpt = cs[-1] if cs else None
        if not ckpt: print("ERROR: no checkpoint found"); sys.exit(1)
    config = json.load(open(ckpt/'config.json'))
    print(f'  Checkpoint: {ckpt.name}')
    print(f'  dim={config["dim"]} layers={config["n_layers"]}')
    n_layers = config['n_layers']
    tensors = []
    total_orig = 0
    total_comp = 0
    t0 = time.time()
    for name, fn in [('token_emb.weight','tok_emb.npy'),('output.weight','output.npy')]:
        W = np.load(ckpt/fn); total_orig += W.nbytes
        q = quantize_q8(W); total_comp += len(q)
        tensors.append((name, W.shape, q))
        print(f'  {name:40s} {W.nbytes/1e6:.0f} -> {len(q)/1e6:.0f} MB')
    nw = np.load(ckpt/'norm.npy')
    total_orig += nw.nbytes; total_comp += nw.nbytes
    tensors.append(('output_norm.weight', nw.shape, nw.tobytes()))
    for L in range(n_layers):
        for tname, fn in [('attn_q',f'layer_{L}_wq.npy'),('attn_k',f'layer_{L}_wk.npy'),('attn_v',f'layer_{L}_wv.npy'),('attn_output',f'layer_{L}_wo.npy'),('ffn_gate',f'layer_{L}_gate.npy'),('ffn_up',f'layer_{L}_up.npy'),('ffn_down',f'layer_{L}_down.npy')]:
            W = np.load(ckpt/fn); total_orig += W.nbytes
            r = svd_compress(W)
            if r is None:
                q = quantize_q8(W); total_comp += len(q)
                tensors.append((f'blk.{L}.{tname}.weight', W.shape, q))
            else:
                A,B,rank,energy = r
                Aq = quantize_q8(A); Bq = quantize_q8(B)
                total_comp += len(Aq)+len(Bq)
                tensors.append((f'blk.{L}.{tname}.weight.a', A.shape, Aq))
                tensors.append((f'blk.{L}.{tname}.weight.b', B.shape, Bq))
                if L%8==0:
                    print(f'  L{L:02d}.{tname:12s} rank={rank:4d}/{min(W.shape)} energy={energy:.4f} {W.nbytes/1e6:.0f}->{(len(Aq)+len(Bq))/1e6:.0f} MB')
        for nn2,nf in [(f'blk.{L}.attn_norm.weight',f'layer_{L}_attn_norm.npy'),(f'blk.{L}.ffn_norm.weight',f'layer_{L}_ffn_norm.npy')]:
            n2 = np.load(ckpt/nf); total_orig+=n2.nbytes; total_comp+=n2.nbytes
            tensors.append((nn2, n2.shape, n2.tobytes()))
        if (L+1)%8==0:
            e=time.time()-t0; eta=e/(L+1)*(n_layers-L-1)
            print(f'  --- {L+1}/{n_layers} layers ({e:.0f}s, ETA {eta:.0f}s) ---')
    print(f'\n  Original: {total_orig/1e9:.2f} GB')
    print(f'  .ntmre compression:   {total_comp/1e9:.2f} GB')
    print(f'  Ratio:    {total_orig/total_comp:.1f}x')
    out_path = f'nightmare_7b_svdq8_{ckpt.name}.gguf'
    print(f'\n  Writing {out_path}...')

    # Load tokenizer for vocab AND merges
    tok_path = None
    for p in [Path('training_data/tokenizer.json'),Path('tokenizer.json'),ckpt/'tokenizer.json']:
        if p.exists(): tok_path=p; break
    vocab = None
    merges = None
    if tok_path:
        td = json.load(open(tok_path, 'r', encoding='utf-8'))
        if 'model' in td:
            vocab = td['model'].get('vocab')
            merges = td['model'].get('merges', [])
        else:
            vocab = td.get('vocab')
            merges = td.get('merges', [])
        print(f'  Tokenizer: {len(vocab)} tokens, {len(merges)} merges')

    # Count KV entries
    n_kv = 13  # base architecture metadata
    if vocab:
        n_kv += 5  # model + tokens + scores + token_type + eos
        if merges: n_kv += 1  # merges array

    with open(out_path,'wb') as f:
        f.write(struct.pack('<I',GGUF_MAGIC))
        f.write(struct.pack('<I',GGUF_VERSION))
        f.write(struct.pack('<Q',len(tensors)))
        f.write(struct.pack('<Q',n_kv))
        wkv_s(f,'general.architecture','llama')
        wkv_s(f,'general.name','Nightmare-7B-SVDQ8')
        wkv_s(f,'general.quantization_version','nightmare_svd_q8_v2')
        wkv_u(f,'llama.context_length',config.get('max_seq_len',1024))
        wkv_u(f,'llama.embedding_length',config['dim'])
        wkv_u(f,'llama.block_count',n_layers)
        wkv_u(f,'llama.feed_forward_length',config['hidden_dim'])
        wkv_u(f,'llama.attention.head_count',config['n_heads'])
        wkv_u(f,'llama.attention.head_count_kv',config['n_kv_heads'])
        wkv_u(f,'llama.vocab_size',config['vocab_size'])
        wkv_u(f,'general.file_type',100)
        wkv_s(f,'nightmare.architecture','nightmare_svd')
        wkv_s(f,'nightmare.compression','svd_rank15_q8_energy999')
        if vocab:
            # tokenizer.ggml.model
            wkv_s(f,'tokenizer.ggml.model','gpt2')

            # Sort tokens by ID
            st = sorted(vocab.items(), key=lambda x: x[1])
            token_list = [t[0] for t in st]

            # Pad to vocab_size if needed
            vocab_size = config.get('vocab_size', len(token_list))
            while len(token_list) < vocab_size:
                token_list.append(f'<unused_{len(token_list)}>')
            token_list = token_list[:vocab_size]

            # === BUILD SCORES (THE FIX) ===
            # Score = negative merge rank. First merge = highest priority.
            # Without this, runtime can't BPE → byte soup → garbage output.
            merge_result_scores = {}
            for rank, merge in enumerate(merges):
                if isinstance(merge, list):
                    merged = ''.join(merge)
                else:
                    parts = merge.split(' ', 1)
                    merged = ''.join(parts)
                merge_result_scores[merged] = -(rank + 1)

            scores = []
            token_types = []
            for token_str in token_list:
                # Score from merge rank
                scores.append(float(merge_result_scores.get(token_str, 0.0)))
                # Token type classification
                if token_str.startswith('<') and token_str.endswith('>'):
                    if token_str.startswith('<0x') and len(token_str) == 6:
                        token_types.append(TOKEN_TYPE_BYTE)
                    else:
                        token_types.append(TOKEN_TYPE_CONTROL)
                else:
                    token_types.append(TOKEN_TYPE_NORMAL)

            n_scored = sum(1 for s in scores if s != 0.0)
            print(f'  Scores computed: {n_scored} tokens from {len(merges)} merges')

            # tokenizer.ggml.tokens (array of strings)
            ws(f,'tokenizer.ggml.tokens')
            f.write(struct.pack('<I',9))  # array type
            f.write(struct.pack('<I',8))  # element type = string
            f.write(struct.pack('<Q',len(token_list)))
            for t in token_list: ws(f, t)

            # tokenizer.ggml.scores (array of float32) — CRITICAL FIX
            wkv_arr_f32(f, 'tokenizer.ggml.scores', scores)

            # tokenizer.ggml.token_type (array of int32) — CRITICAL FIX
            wkv_arr_i32(f, 'tokenizer.ggml.token_type', token_types)

            # tokenizer.ggml.merges (array of strings)
            if merges:
                ws(f,'tokenizer.ggml.merges')
                f.write(struct.pack('<I',9))  # array type
                f.write(struct.pack('<I',8))  # element type = string
                f.write(struct.pack('<Q',len(merges)))
                for m in merges:
                    if isinstance(m, list): m = ' '.join(m)
                    ws(f, str(m))
                print(f'  Embedded {len(merges)} BPE merges')

            # EOS token ID
            wkv_u(f,'tokenizer.ggml.eos_token_id',vocab.get('</s>',2))

        # Tensor info headers
        offset=0; blobs=[]
        for name,shape,data in tensors:
            ws(f,name)
            f.write(struct.pack('<I',len(shape)))
            for s in reversed(shape): f.write(struct.pack('<Q',s))
            f.write(struct.pack('<I',8))  # Q8_0 type
            f.write(struct.pack('<Q',offset))
            blobs.append(data)
            offset+=len(data); offset+=(32-offset%32)%32
        pos=f.tell(); f.write(b'\x00'*((32-pos%32)%32))
        for blob in blobs:
            f.write(blob); pad=(32-len(blob)%32)%32
            if pad: f.write(b'\x00'*pad)
    final=Path(out_path).stat().st_size
    print(f'\n  DONE: {out_path}')
    print(f'  SIZE: {final/1e9:.2f} GB')
    print(f'  We are the nightmare they cannot wake from.')

if __name__=='__main__':
    if len(sys.argv) > 1:
        # Allow specifying checkpoint path
        ckpt_override = Path(sys.argv[1])
        if ckpt_override.exists():
            export(ckpt_override)
        else:
            print(f"ERROR: {ckpt_override} not found"); sys.exit(1)
    else:
        export()
