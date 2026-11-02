import numpy as np


import struct


import json


from pathlib import Path





DIM = 512


N_LAYERS = 20


N_HEADS = 8


N_KV_HEADS = 4


HEAD_DIM = 64


HIDDEN_DIM = 1376


VOCAB_SIZE = 64000


MAX_SEQ = 512





GGUF_MAGIC = 0x46554747


GGUF_VERSION = 3


GGUF_TYPE_UINT32 = 4


GGUF_TYPE_INT32 = 3


GGUF_TYPE_FLOAT32 = 6


GGUF_TYPE_STRING = 8


GGUF_TYPE_ARRAY = 9





def write_string(f, s):


    b = s.encode('utf-8')


    f.write(struct.pack('<Q', len(b)))


    f.write(b)





def write_kv_string(f, key, val):


    write_string(f, key)


    f.write(struct.pack('<I', GGUF_TYPE_STRING))


    write_string(f, val)





def write_kv_uint32(f, key, val):


    write_string(f, key)


    f.write(struct.pack('<I', GGUF_TYPE_UINT32))


    f.write(struct.pack('<I', val))





def quantize_q4_0(data):


    data = data.flatten().astype(np.float32)


    n = len(data)


    if n % 32 != 0:


        data = np.concatenate([data, np.zeros(32 - n % 32, dtype=np.float32)])


        n = len(data)


    blocks = data.reshape(-1, 32)


    n_blocks = len(blocks)


    output = bytearray(n_blocks * 18)


    for i, block in enumerate(blocks):


        amax = np.max(np.abs(block))


        scale = amax / 7.0 if amax != 0 else 0.0


        inv_scale = 1.0 / scale if scale != 0 else 0.0


        scale_f16 = np.float16(scale)


        struct.pack_into('<e', output, i * 18, float(scale_f16))


        quantized = np.clip(np.round(block * inv_scale) + 8, 0, 15).astype(np.uint8)


        for j in range(16):


            low = quantized[j * 2]


            high = quantized[j * 2 + 1]


            output[i * 18 + 2 + j] = (low & 0x0F) | ((high & 0x0F) << 4)


    return bytes(output)





# Find checkpoint


ckpt = None


for p in [Path('checkpoints_90m/step_59617'), Path('checkpoints_90m/best')]:


    if p.exists():


        ckpt = p


        break


if not ckpt:


    ckpt_dir = Path('checkpoints_90m')


    if ckpt_dir.exists():


        candidates = sorted(


            [d for d in ckpt_dir.iterdir() if d.is_dir() and 'step_' in d.name],


            key=lambda x: int(x.name.split('_')[1])


        )


        if candidates:


            ckpt = candidates[-1]


if not ckpt:


    print("ERROR: No checkpoint found!")


    exit(1)





print(f"Exporting: {ckpt}")





# Load tokenizer


tok_path = None


for p in [Path('training_data/tokenizer.json'), Path('tokenizer.json')]:


    if p.exists():


        tok_path = p


        break





vocab = None


merges = None


if tok_path:


    tok_data = json.load(open(tok_path, 'r', encoding='utf-8'))


    if 'model' in tok_data and 'vocab' in tok_data['model']:


        vocab = tok_data['model']['vocab']


        merges = tok_data['model'].get('merges', [])


    else:


        vocab = tok_data.get('vocab', {})


        merges = tok_data.get('merges', [])


    print(f"Tokenizer: {len(vocab)} tokens, {len(merges)} merges")


else:


    print("ERROR: No tokenizer found!")


    exit(1)





# Build token list, scores, and types


sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])


token_list = [t[0] for t in sorted_tokens]


while len(token_list) < VOCAB_SIZE:


    token_list.append(f'<unused_{len(token_list)}>')





# Build merge lookup: which token was formed by which merge rank


# A merged token's score = -(merge_rank) so earlier merges get higher priority


merge_result_to_rank = {}


for rank, m in enumerate(merges):


    if isinstance(m, list):


        merged_str = ''.join(m)


    else:


        parts = m.split(' ', 1)


        merged_str = ''.join(parts)


    if merged_str in vocab:


        merge_result_to_rank[vocab[merged_str]] = rank





# Scores: -(merge_rank) for merged tokens, 0.0 for base tokens


scores = []


for i in range(VOCAB_SIZE):


    if i in merge_result_to_rank:


        scores.append(-float(merge_result_to_rank[i]))


    else:


        scores.append(0.0)





# Token types: 1=NORMAL, 3=CONTROL, 6=BYTE


token_types = []


control_tokens = {'<s>', '</s>', '<pad>', '<unk>'}


for i, tok in enumerate(token_list):


    if tok in control_tokens:


        token_types.append(3)  # CONTROL


    elif tok.startswith('<0x') and tok.endswith('>') and len(tok) == 6:


        token_types.append(6)  # BYTE


    elif tok.startswith('<unused_'):


        token_types.append(5)  # UNUSED


    else:


        token_types.append(1)  # NORMAL





print(f"  Scores computed: {sum(1 for s in scores if s < 0)} merged tokens have rank scores")


print(f"  Token types: {sum(1 for t in token_types if t==1)} normal, {sum(1 for t in token_types if t==6)} byte, {sum(1 for t in token_types if t==3)} control")





# Build tensor map


tensor_map = []


tensor_map.append(('token_emb.weight', ckpt / 'tok_emb.npy', 1, False))


tensor_map.append(('output_norm.weight', ckpt / 'final_norm.npy', 0, False))


layers_dir = ckpt / 'layers'


for i in range(N_LAYERS):


    ld = layers_dir / f'layer_{i:03d}'


    tensor_map.append((f'blk.{i}.attn_q.weight', ld / 'wq.npy', 2, True))


    tensor_map.append((f'blk.{i}.attn_k.weight', ld / 'wk.npy', 2, True))


    tensor_map.append((f'blk.{i}.attn_v.weight', ld / 'wv.npy', 2, True))


    tensor_map.append((f'blk.{i}.attn_output.weight', ld / 'wo.npy', 2, True))


    tensor_map.append((f'blk.{i}.ffn_gate.weight', ld / 'w_gate.npy', 2, True))


    tensor_map.append((f'blk.{i}.ffn_up.weight', ld / 'w_up.npy', 2, True))


    tensor_map.append((f'blk.{i}.ffn_down.weight', ld / 'w_down.npy', 2, True))


    tensor_map.append((f'blk.{i}.attn_norm.weight', ld / 'attn_norm.npy', 0, False))


    tensor_map.append((f'blk.{i}.ffn_norm.weight', ld / 'ffn_norm.npy', 0, False))


tensor_map.append(('output.weight', ckpt / 'tok_emb.npy', 2, False))





# KV count: 12 base + model + tokens + scores + token_type + merges + eos = 18


n_kv = 12 + 6  # base metadata + tokenizer fields





print(f"Tensors: {len(tensor_map)}")


out_path = 'nightmare_90m_q4.gguf'





with open(out_path, 'wb') as f:


    f.write(struct.pack('<I', GGUF_MAGIC))


    f.write(struct.pack('<I', GGUF_VERSION))


    f.write(struct.pack('<Q', len(tensor_map)))


    f.write(struct.pack('<Q', n_kv))


    


    # Model metadata


    write_kv_string(f, 'general.architecture', 'llama')


    write_kv_string(f, 'general.name', 'Nightmare-90M-Chat')


    write_kv_uint32(f, 'llama.context_length', MAX_SEQ)


    write_kv_uint32(f, 'llama.embedding_length', DIM)


    write_kv_uint32(f, 'llama.block_count', N_LAYERS)


    write_kv_uint32(f, 'llama.feed_forward_length', HIDDEN_DIM)


    write_kv_uint32(f, 'llama.attention.head_count', N_HEADS)


    write_kv_uint32(f, 'llama.attention.head_count_kv', N_KV_HEADS)


    write_kv_uint32(f, 'llama.vocab_size', VOCAB_SIZE)


    write_kv_uint32(f, 'general.file_type', 2)


    write_kv_string(f, 'general.quantization_version', '1')


    write_kv_string(f, 'nightmare.architecture', 'nightmare')


    


    # Tokenizer model type


    write_string(f, 'tokenizer.ggml.model')


    f.write(struct.pack('<I', GGUF_TYPE_STRING))


    write_string(f, 'llama')


    


    # Tokenizer tokens (string array)


    write_string(f, 'tokenizer.ggml.tokens')


    f.write(struct.pack('<I', GGUF_TYPE_ARRAY))


    f.write(struct.pack('<I', GGUF_TYPE_STRING))


    f.write(struct.pack('<Q', len(token_list)))


    for tok_str in token_list:


        write_string(f, tok_str)


    


    # Tokenizer scores (float32 array) — THIS WAS MISSING!


    write_string(f, 'tokenizer.ggml.scores')


    f.write(struct.pack('<I', GGUF_TYPE_ARRAY))


    f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))


    f.write(struct.pack('<Q', len(scores)))


    for s in scores:


        f.write(struct.pack('<f', s))


    


    # Tokenizer token types (int32 array) — THIS WAS MISSING!


    write_string(f, 'tokenizer.ggml.token_type')


    f.write(struct.pack('<I', GGUF_TYPE_ARRAY))


    f.write(struct.pack('<I', GGUF_TYPE_INT32))


    f.write(struct.pack('<Q', len(token_types)))


    for t in token_types:


        f.write(struct.pack('<i', t))


    


    # Merges (string array)


    write_string(f, 'tokenizer.ggml.merges')


    f.write(struct.pack('<I', GGUF_TYPE_ARRAY))


    f.write(struct.pack('<I', GGUF_TYPE_STRING))


    f.write(struct.pack('<Q', len(merges)))


    for m in merges:


        if isinstance(m, list):


            m = ' '.join(m)


        write_string(f, m)


    print(f"  Embedded {len(merges)} merges")


    


    # EOS token ID


    write_kv_uint32(f, 'tokenizer.ggml.eos_token_id', vocab.get('</s>', 2))


    


    # Tensor info headers


    tensor_data_list = []


    offset = 0


    for name, npy_path, quant_type, needs_transpose in tensor_map:


        data = np.load(npy_path).astype(np.float32)


        if needs_transpose and data.ndim == 2:


            data = data.T


        shape = data.shape


        if quant_type == 2:


            qdata = quantize_q4_0(data)


        elif quant_type == 1:


            qdata = data.astype(np.float16).tobytes()


        else:


            qdata = data.tobytes()


        size = len(qdata)


        tensor_data_list.append(qdata)


        write_string(f, name)


        f.write(struct.pack('<I', len(shape)))


        for s in reversed(shape):


            f.write(struct.pack('<Q', s))


        f.write(struct.pack('<I', quant_type))


        f.write(struct.pack('<Q', offset))


        offset += size


        if offset % 32 != 0:


            offset += 32 - (offset % 32)


        print(f'  {name:40s} {str(shape):20s} {"Q4" if quant_type==2 else "F16" if quant_type==1 else "F32":4s} {size/1e6:.1f} MB')


    


    # Align to 32 bytes before tensor data


    pos = f.tell()


    if pos % 32 != 0:


        f.write(b'\x00' * (32 - pos % 32))


    


    # Write tensor data


    for qdata in tensor_data_list:


        f.write(qdata)


        if len(qdata) % 32 != 0:


            f.write(b'\x00' * (32 - len(qdata) % 32))





import os


final_size = os.path.getsize(out_path)


print(f"\nDone! {out_path} ({final_size/1e6:.1f} MB)")


