# -*- coding: utf-8 -*-
"""
bonsai_gpu.py  —  Bonsai-27B (qwen35 hybrid) GPU streaming runner
================================================================
Uses the user's OWN NVRTC kernels (no nvcc, no torch):
  * NVRTC matmul kernel (from engine/bonsai_pc_app.py)
  * NVRTC Q1_0 dequant kernel (keeps weights quantized in RAM)
Weights stay Q1_0 in RAM; each layer is dequantized on the GPU and
matmul'd on the GPU, then freed.  SSM state update stays in numpy
(small).  Math follows llama.cpp qwen35.cpp exactly.

Correctness fixes vs older copies:
  * SiLU applied after the SSM conv1d
  * FFN uses post_attention_norm, residual added to pre-norm x
  * RoPE = standard RoPE on first 64 dims (freq = base^(-2i/64)),
    matching llama.cpp MRoPE for a text-only model
"""
import sys, os, ctypes, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf_parser
from engine import bonsai_pc_app as b

DEQ_KERNEL = r"""
extern "C" __device__ float h2f(unsigned short h){
    unsigned sign=(h&0x8000)?1:0; unsigned exp=(h>>10)&0x1f; unsigned frac=h&0x3ff; unsigned f;
    if(exp==0){ if(frac==0) f=sign<<31; else { exp=127-15+1; while((frac&0x400)==0){frac<<=1;exp--;} frac&=0x3ff; f=(sign<<31)|(exp<<23)|(frac<<13);} }
    else if(exp==31) f=(sign<<31)|0x7f800000|(frac<<13);
    else f=(sign<<31)|((exp+127-15)<<23)|(frac<<13);
    return __uint_as_float(f);
}
extern "C" __global__ void dequant_q1_0_t(const unsigned char* in, float* out, int n_blocks, int out_cols){
    int b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=n_blocks) return;
    const unsigned char* blk=in+(size_t)b*18;
    unsigned short sb=(unsigned short)blk[0] | ((unsigned short)blk[1]<<8);
    float scale=h2f(sb);
    const unsigned char* packed=blk+2;
    for(int j=0;j<128;j++){
        int idx=b*128+j;
        int i=idx/out_cols;
        int o=idx%out_cols;
        unsigned char byte=packed[j>>3];
        float v=((byte>>(j&7))&1)? scale : -scale;
        out[o* (n_blocks*128/out_cols) + i] = v;
    }
}
// Fused Q1_0 dequant+matvec: out[n] = sum_k x[k]*W[k,n], weights read
// straight from 1-bit blocks in VRAM — never dequantized to floats.
// Raw layout: row n (output) = bpr contiguous 18-byte blocks, 128 inputs each.
extern "C" __global__ void q1_0_matvec(const unsigned char* w, const float* x, float* out, int K, int N){
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=N) return;
    int bpr=K>>7;
    const unsigned char* row=w+(size_t)n*bpr*18;
    float acc=0.f;
    for(int b=0;b<bpr;b++){
        const unsigned char* blk=row+(size_t)b*18;
        unsigned short sb=(unsigned short)blk[0] | ((unsigned short)blk[1]<<8);
        float scale=h2f(sb);
        const unsigned char* packed=blk+2;
        const float* xb=x+(b<<7);
        float sp=0.f, sm=0.f;
        #pragma unroll
        for(int j=0;j<16;j++){
            unsigned char byte=packed[j];
            #pragma unroll
            for(int t=0;t<8;t++){
                float xv=xb[(j<<3)+t];
                if((byte>>t)&1) sp+=xv; else sm+=xv;
            }
        }
        acc+=scale*(sp-sm);
    }
    out[n]=acc;
}
"""


def _rmsnorm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w.astype(np.float32)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30, 30)))


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _rope(x, position, n_dims, base):
    """Standard RoPE on first n_dims dims, pairs (i, i+n_dims/2)."""
    out = x.copy()
    half = n_dims // 2
    freqs = base ** (-2.0 * np.arange(half) / n_dims)
    ang = position * freqs
    c = np.cos(ang); s = np.sin(ang)
    x0 = out[..., :half].copy()
    x1 = out[..., half:n_dims].copy()
    out[..., :half] = x0 * c - x1 * s
    out[..., half:n_dims] = x0 * s + x1 * c
    return out


class BonsaiGPU:
    def __init__(self, path, device=1):
        self.g = gguf_parser.GGUFParser(path)
        self.m = self.g.parse()
        self.dim = self.m.embedding_length
        self.n_layers = self.m.block_count
        self.vocab = self.m.vocab_size
        self.n_heads = self.m.head_count
        self.n_kv = self.m.head_count_kv_scalar
        self.head_dim = int(self.m.metadata.get('qwen35.attention.key_length', 256))
        self.ffn_dim = self.m.feed_forward_length
        self.norm_eps = float(self.m.metadata.get('qwen35.attention.layer_norm_rms_epsilon', 1e-6))
        self.rope_base = float(self.m.metadata.get('qwen35.rope.freq_base', 1e7))
        self.rope_dim = int(self.m.metadata.get('qwen35.rope.dimension_count', 64))
        self.full_interval = int(self.m.metadata.get('qwen35.full_attention_interval', 4))
        # SSM dims
        self.ssm_state = int(self.m.metadata.get('qwen35.ssm.state_size', 128))
        self.ssm_group = int(self.m.metadata.get('qwen35.ssm.group_count', 16))
        self.ssm_dt = int(self.m.metadata.get('qwen35.ssm.time_step_rank', 48))
        self.ssm_inner = int(self.m.metadata.get('qwen35.ssm.inner_size', 6144))
        self.ssm_conv = int(self.m.metadata.get('qwen35.ssm.conv_kernel', 4))
        self.head_v_dim = self.ssm_inner // self.ssm_dt
        self.key_dim = self.ssm_state * self.ssm_group
        self.value_dim = self.head_v_dim * self.ssm_dt
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # GPU
        self.cu = b.Cuda(device=device)
        self._compile_dequant()
        self._scratch = None
        self._scratch_nbytes = 0

        # Load raw Q1_0 for all 2D weights + embd + output (kept quantized)
        self.raw = {}
        self.small = {}
        for name, info in self.m.tensors.items():
            if info.dtype == 41:  # Q1_0
                raw, _, shape, ne = self.g.load_tensor_raw(self.m, name)
                self.raw[name] = (raw, tuple(shape), ne)
            else:
                # 1D small weights (norms, ssm_a, ssm_dt, conv1d)
                self.small[name] = self.g.load_tensor(self.m, name).astype(np.float32)

        # VRAM-resident quantized weights (one upload at load; per-token
        # compute reads them at compressed size via the fused matvec kernel)
        self.vram = None
        self._xbuf = None
        self._xbuf_nbytes = 0
        self._obufs = {}
        self._vram_upload_all()

        # recurrent states
        self.ssm_state_mem = [np.zeros((self.ssm_state, self.head_v_dim, self.ssm_dt), np.float32)
                              for _ in range(self.n_layers)]
        self.conv_mem = [np.zeros((self.ssm_conv - 1, self.conv_dim), np.float32)
                         for _ in range(self.n_layers)]
        self.kv = {l: [] for l in range(self.n_layers)}

        # tokenizer
        from tokenizer import Tokenizer
        self.tok = Tokenizer(self.m.metadata, path)

    # ---- GPU helpers ----
    def _free_vram(self):
        free = ctypes.c_size_t(); tot = ctypes.c_size_t()
        self.cu.cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(tot))
        return free.value

    def _vram_upload_all(self):
        """Upload all raw Q1_0 weights to VRAM once. Falls back to
        per-token streaming if they don't fit on either GPU."""
        total = sum(len(v[0]) for v in self.raw.values())
        for attempt in range(2):
            free = self._free_vram()
            if total <= int(free * 0.85):
                break
            if attempt == 0:
                other = 1 - self.cu.dev.value
                print('[bonsai] %.2fGB weights vs %.2fGB free on dev %d - trying dev %d'
                      % (total / 1e9, free / 1e9, self.cu.dev.value, other))
                try:
                    self.cu = b.Cuda(device=other)
                    self._compile_dequant()
                    self._scratch = None
                    self._scratch_nbytes = 0
                except Exception as e:
                    print('[bonsai] dev %d unavailable: %s' % (other, e))
                    break
            else:
                print('[bonsai] weights do not fit in VRAM - streaming fallback (slow)')
                self.vram = None
                return
        self.vram = {}
        t0 = time.time()
        for name, (raw, shape, ne) in self.raw.items():
            arr = np.frombuffer(raw, dtype=np.uint8)
            ptr = self.cu.alloc(arr.nbytes)
            r = self.cu.cu.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes)
            if r != 0:
                raise RuntimeError('HtoD %d' % r)
            self.vram[name] = (ptr, shape, ne)
        print('[bonsai] %.2fGB quantized weights resident in VRAM (%.1fs upload)'
              % (total / 1e9, time.time() - t0))

    def _mmv(self, xv, name):
        """Fused Q1_0 matvec on VRAM-resident weights. xv: (K,) float32."""
        ptr, shape, ne = self.vram[name]
        K, N = int(shape[0]), int(shape[1])
        if self._xbuf is None or self._xbuf_nbytes < xv.nbytes:
            if self._xbuf is not None:
                self.cu.free(self._xbuf)
            self._xbuf = self.cu.alloc(xv.nbytes)
            self._xbuf_nbytes = xv.nbytes
        xv = np.ascontiguousarray(xv, dtype=np.float32)
        r = self.cu.cu.cuMemcpyHtoD_v2(self._xbuf, ctypes.c_void_p(xv.ctypes.data), xv.nbytes)
        if r != 0:
            raise RuntimeError('HtoD x %d' % r)
        op = self._obufs.get(N)
        if op is None:
            op = self.cu.alloc(N * 4)
            self._obufs[N] = op
        self.cu.launch(self._matvec_fn, ((N + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(ptr.value), ctypes.c_ulonglong(self._xbuf.value),
                        ctypes.c_ulonglong(op.value), K, N])
        return self.cu.from_gpu(op, (N,))

    def _compile_dequant(self):
        cu = self.cu
        prog = ctypes.c_void_p()
        cu.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), DEQ_KERNEL.encode(), b"d.cu", 0, None, None)
        opts = (ctypes.c_char_p * 2)(b"--std=c++17", b"-use_fast_math")
        r = cu.nvrtc.nvrtcCompileProgram(prog, 2, opts)
        if r != 0:
            sz = ctypes.c_size_t(); cu.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            buf = ctypes.create_string_buffer(sz.value); cu.nvrtc.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError("dequant compile failed:\n" + buf.value.decode())
        psz = ctypes.c_size_t(); cu.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(psz))
        ptx = ctypes.create_string_buffer(psz.value); cu.nvrtc.nvrtcGetPTX(prog, ptx)
        mod = ctypes.c_void_p(); cu.cu.cuModuleLoadDataEx(ctypes.byref(mod), ptx, 0, None, None)
        fn = ctypes.c_void_p(); cu.cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"dequant_q1_0_t")
        self._deq_fn = fn
        mv = ctypes.c_void_p(); cu.cu.cuModuleGetFunction(ctypes.byref(mv), mod, b"q1_0_matvec")
        self._matvec_fn = mv

    def _raw_upload(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        ptr = self.cu.alloc(arr.nbytes)
        r = self.cu.cu.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes)
        if r != 0:
            raise RuntimeError("HtoD %d" % r)
        return ptr

    def _dequant_gpu(self, name):
        """Dequant weight (name) into a persistent GPU scratch, return (ptr, shape).
        Kernel writes out[(idx%in)*out + idx//in] so out_cols must be the INPUT dim
        (shape[0]) to match load_tensor's (in, out) layout."""
        raw, shape, ne = self.raw[name]
        n_blocks = ne // 128
        out_cols = shape[0]  # input dim (stride = out dim)
        nbytes = ne * 4
        if self._scratch is None or self._scratch_nbytes < nbytes:
            if self._scratch is not None:
                self.cu.free(self._scratch)
            self._scratch = self.cu.alloc(nbytes)
            self._scratch_nbytes = nbytes
        inp = np.frombuffer(raw, dtype=np.uint8)
        ip = self._raw_upload(inp)
        self.cu.launch(self._deq_fn, ((n_blocks + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(ip.value), ctypes.c_ulonglong(self._scratch.value),
                        n_blocks, out_cols])
        self.cu.free(ip)
        return self._scratch, (shape[0], shape[1])

    def _mm(self, x, name):
        """y = x @ W(name). x: (seq, in). Returns (seq, out)."""
        seq, k = x.shape
        if self.vram is not None and seq == 1:
            return self._mmv(x[0], name)[None, :]
        wp, (m, n) = self._dequant_gpu(name)  # W: (in=m, out=n)
        xp, _ = self.cu.to_gpu(x)
        cp = self.cu.alloc(seq * n * 4)
        self.cu.launch(self.cu.f_matmul, ((n + 15) // 16, (seq + 15) // 16), (16, 16),
                       [ctypes.c_ulonglong(xp.value), ctypes.c_ulonglong(wp.value),
                        ctypes.c_ulonglong(cp.value), seq, k, n])
        out = self.cu.from_gpu(cp, (seq, n))
        self.cu.free(xp); self.cu.free(cp)
        return out

    def _embed(self, token):
        """Return embedding vector for token (numpy, dequant just that row)."""
        raw, shape, ne = self.raw['token_embd.weight']
        dim = self.dim
        row = np.frombuffer(raw, dtype=np.uint8)[token*dim//128*18 : (token+1)*dim//128*18]
        blocks = row.reshape(-1, 18)
        scales = blocks[:, 0:2].copy().view(np.uint16).view(np.float16).astype(np.float32).ravel()
        bits = np.unpackbits(blocks[:, 2:18], axis=1, bitorder='little')
        vals = np.where(bits == 1, np.float32(1.0), np.float32(-1.0)) * scales[:, None]
        return vals.ravel()[:dim].astype(np.float32)

    # ---- layers ----
    def _ssm_layer(self, x, i):
        """Gated DeltaNet linear attention. x: (1, dim)."""
        h = _rmsnorm(x, self.small['blk.%d.attn_norm.weight' % i], self.norm_eps)
        h2 = h[None, :]
        qkv = self._mm(h2, 'blk.%d.attn_qkv.weight' % i)[0]
        z = self._mm(h2, 'blk.%d.attn_gate.weight' % i)[0]
        beta = self._mm(h2, 'blk.%d.ssm_beta.weight' % i)[0]
        alpha = self._mm(h2, 'blk.%d.ssm_alpha.weight' % i)[0]
        dt = self.small['blk.%d.ssm_dt.bias' % i]
        a = self.small['blk.%d.ssm_a' % i]
        gate = _softplus(alpha + dt) * a
        decay = np.exp(np.clip(gate, -30, 0))
        beta_sig = 1.0 / (1.0 + np.exp(-beta))

        conv_w = self.small['blk.%d.ssm_conv1d.weight' % i]  # (kernel, conv_dim)
        if conv_w.shape[0] != self.ssm_conv:
            conv_w = conv_w.T  # GGUF stores (conv_dim, kernel); need tap-major
        conv_in = np.concatenate([self.conv_mem[i], qkv[None, :]], axis=0)
        self.conv_mem[i] = conv_in[-(self.ssm_conv - 1):].copy()
        conv_out = np.zeros_like(qkv)
        for k in range(self.ssm_conv):
            conv_out += conv_w[k] * conv_in[k]  # current token weighted by last conv weight
        conv_out = _silu(conv_out)

        hk, nk, nv = self.ssm_state, self.ssm_group, self.ssm_dt
        hv = self.head_v_dim
        # llama.cpp views conv out as (head_k_dim, num_k_heads) [i][j]=dim i of head j,
        # so q,k as (num_heads, head_k_dim) after transpose; v as (num_heads, head_v_dim)
        q = conv_out[:hk*nk].reshape(nk, hk)
        k = conv_out[hk*nk:2*hk*nk].reshape(nk, hk)
        v = conv_out[2*hk*nk:].reshape(nv, hv)
        # GQA: repeat key heads to match value heads. llama.cpp uses ggml_repeat_4d
        # which TILES (head v uses head v % nk), not np.repeat (which would use v//nk).
        rep = nv // nk
        if rep > 1:
            q = np.tile(q, (rep, 1))
            k = np.tile(k, (rep, 1))
        q = q / (np.sqrt(np.sum(q*q, axis=1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k*k, axis=1, keepdims=True)) + 1e-6)
        q = q * (1.0 / np.sqrt(hv))

        state = self.ssm_state_mem[i]          # (hk, hv, nv)
        state = state * decay[None, None, :]
        kv_mem = np.einsum('khv,vk->hv', state, k)      # (hv, nv)
        delta = (v.T - kv_mem) * beta_sig[None, :]      # (hv, nv)
        state = state + np.einsum('vk,hv->khv', k, delta)
        self.ssm_state_mem[i] = state
        core = np.einsum('khv,vk->hv', state, q)        # (hv, nv)

        core = core.reshape(hv, nv)
        rms = np.sqrt(np.mean(core*core, axis=0, keepdims=True) + self.norm_eps)
        sn = self.small['blk.%d.ssm_norm.weight' % i]
        core_norm = (core / rms) * sn[:, None]
        # llama.cpp stores z as [head_v_dim, num_v_heads] (z[i + j*head_v_dim]);
        # reshape accordingly so the gate lines up with the core.
        zz = z.reshape(nv, hv).T
        core_gated = core_norm * _silu(zz)
        # llama.cpp flattens final_output as [i + j*head_v_dim] (head_v_dim fastest),
        # so transpose before the ssm_out projection to match its storage layout.
        out = self._mm(core_gated.T.reshape(1, hv*nv), 'blk.%d.ssm_out.weight' % i)[0]
        return out

    def _full_attention(self, x, i, position):
        """Gated full attention. x: (1, dim)."""
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv
        h = _rmsnorm(x, self.small['blk.%d.attn_norm.weight' % i], self.norm_eps)
        h2 = h[None, :]
        q_full = self._mm(h2, 'blk.%d.attn_q.weight' % i)[0].reshape(nh, hd*2)
        query = q_full[:, :hd]
        gate = q_full[:, hd:]
        key = self._mm(h2, 'blk.%d.attn_k.weight' % i)[0].reshape(nkv, hd)
        value = self._mm(h2, 'blk.%d.attn_v.weight' % i)[0].reshape(nkv, hd)
        query = _rmsnorm(query, self.small['blk.%d.attn_q_norm.weight' % i], self.norm_eps)
        key = _rmsnorm(key, self.small['blk.%d.attn_k_norm.weight' % i], self.norm_eps)
        query = _rope(query, position, self.rope_dim, self.rope_base)
        key = _rope(key, position, self.rope_dim, self.rope_base)
        self.kv[i].append((key.copy(), value.copy()))
        K_all = np.array([kv[0] for kv in self.kv[i]])
        V_all = np.array([kv[1] for kv in self.kv[i]])
        scale = 1.0 / np.sqrt(hd)
        out = np.zeros((nh, hd), np.float32)
        for hh in range(nh):
            # GQA: query head hh maps to KV head hh // (nh // nkv)
            kvh = hh // (nh // nkv)
            scores = (K_all[:, kvh] @ query[hh]) * scale
            w = _softmax(scores, axis=0)
            out[hh] = np.sum(w[:, None] * V_all[:, kvh], axis=0)
        out = out * (1.0 / (1.0 + np.exp(-gate)))
        # llama.cpp flattens (head_dim, n_head) in head_dim-major order
        return self._mm(out.reshape(1, hd*nh), 'blk.%d.attn_output.weight' % i)[0]

    def forward(self, token, position):
        x = self._embed(token)
        for i in range(self.n_layers):
            if i % self.full_interval == (self.full_interval - 1):
                attn_out = self._full_attention(x, i, position)
            else:
                attn_out = self._ssm_layer(x, i)
            x = x + attn_out
            h = _rmsnorm(x, self.small['blk.%d.post_attention_norm.weight' % i], self.norm_eps)
            gate = _silu(self._mm(h[None, :], 'blk.%d.ffn_gate.weight' % i)[0])
            up = self._mm(h[None, :], 'blk.%d.ffn_up.weight' % i)[0]
            down = self._mm((gate * up)[None, :], 'blk.%d.ffn_down.weight' % i)[0]
            x = x + down
        x = _rmsnorm(x, self.small['output_norm.weight'], self.norm_eps)
        return self._output_logits(x)

    def _output_logits(self, x):
        """logits = x @ output.weight. output.weight stored natural (input, output);
        llama.cpp reads it as (vocab, input) -> logits[o] = sum_i x[i]*data[o*input+i].
        Chunk by output tokens (each token's input-dim row is contiguous in storage)."""
        if self.vram is not None:
            return self._mmv(x, 'output.weight')
        raw, shape, ne = self.raw['output.weight']
        in_dim = shape[0]      # 5120
        vocab = shape[1]       # 248320
        blocks_per_row = in_dim // 128   # 5120/128 = 40 blocks per output token
        chunk_tok = 4096
        logits = np.zeros(vocab, np.float32)
        inp = np.frombuffer(raw, dtype=np.uint8)
        for c0 in range(0, vocab, chunk_tok):
            c1 = min(c0 + chunk_tok, vocab)
            b0 = c0 * blocks_per_row
            b1 = c1 * blocks_per_row
            seg = inp[b0*18 : b1*18]
            nblocks_c = b1 - b0
            ne_chunk = (c1 - c0) * in_dim
            nbytes = ne_chunk * 4
            if self._scratch is None or self._scratch_nbytes < nbytes:
                if self._scratch is not None:
                    self.cu.free(self._scratch)
                self._scratch = self.cu.alloc(nbytes)
                self._scratch_nbytes = nbytes
            ip = self._raw_upload(seg)
            self.cu.launch(self._deq_fn, ((nblocks_c + 255) // 256, 1), (256, 1),
                           [ctypes.c_ulonglong(ip.value), ctypes.c_ulonglong(self._scratch.value),
                            nblocks_c, ne_chunk])
            self.cu.free(ip)
            wchunk = self.cu.from_gpu(self._scratch, (c1 - c0, in_dim))
            logits[c0:c1] = wchunk @ x
        return logits

    def generate(self, prompt, max_new=32, temperature=0.7, top_k=40, top_p=0.9):
        # Bonsai has add_bos_token=False in its GGUF metadata; do NOT prepend BOS
        ids = self.tok.encode(prompt, add_bos=False)
        if not ids:
            ids = [getattr(self.tok, 'bos_id', 1)]
        out_ids = list(ids)
        t0 = time.time()
        for pos in range(len(out_ids) - 1):
            self.forward(out_ids[pos], pos)
        for step in range(max_new):
            logits = self.forward(out_ids[-1], len(out_ids) - 1)
            probs = _softmax(logits / temperature)
            if top_k and top_k < len(probs):
                idx = np.argsort(probs)[-top_k:]
                m = np.zeros_like(probs); m[idx] = probs[idx]; probs = m / m.sum()
            nxt = int(np.random.choice(len(probs), p=probs))
            out_ids.append(nxt)
            if nxt in (getattr(self.tok, 'eos_id', 2), 0):
                break
        dt = time.time() - t0
        print('[bonsai] %d tokens in %.1fs = %.2f tok/s' % (len(out_ids) - len(ids), dt, (len(out_ids) - len(ids)) / max(dt, 1e-9)))
        return self.tok.decode(out_ids), out_ids


if __name__ == '__main__':
    p = r'C:\Users\User\.lmstudio\models\lmstudio-community\Bonsai-27B-GGUF\Bonsai-27B-Q1_0.gguf'
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    bm = BonsaiGPU(p, device=dev)
    print('model ready', bm.dim, 'layers', bm.n_layers, 'vocab', bm.vocab)
    text, ids = bm.generate('Hello', max_new=20)
    print('OUT:', text)
