# -*- coding: utf-8 -*-
"""
qwen35_gpu.py - NVRTC GPU accelerator for Qwen35HybridEngine
============================================================
Suite-internal (no third-party code): reuses the Cuda wrapper from
engine/bonsai_pc_app.py and adds vector-matrix kernels for the weight
formats the hybrid engine meets:

  F32            raw f32, direct matvec
  Q1_0 (NSVD)    18 B / 128 elems: fp16 scale + 16 packed bytes
  Q8  (NSVD)     34 B / 32 elems:  fp16 scale + 32 x int8
  Q8  (v16)      36 B / 32 elems:  f32  scale + 32 x int8
  .ntmre compression (NSVD)  rank u32 + U Q8 + S f32 + Vt Q8  -> y=(x@Vt.T*S)@U.T
  .ntmre compression pair (v16) separate .A (out,rank) / .B (rank,in) Q8 tensors

All matvecs are y[o] = sum_k x[k] * W[o*K + k]  (W stored (out, in)
row-major), which matches the engine's universal W_pt rule.

Author: Nightmare Designs 💀
"""
import ctypes
import os

import nightmare_np as np

# Reliable GPU memory queries (nvidia-smi / NVML fallback)
import gpu_info

KERNELS = r"""
extern "C" __device__ float h2f(unsigned short h){
    unsigned sign=(h&0x8000)?1:0; unsigned exp=(h>>10)&0x1f; unsigned frac=h&0x3ff; unsigned f;
    if(exp==0){ if(frac==0) f=sign<<31; else { exp=127-15+1; while((frac&0x400)==0){frac<<=1;exp--;} frac&=0x3ff; f=(sign<<31)|(exp<<23)|(frac<<13);} }
    else if(exp==31) f=(sign<<31)|0x7f800000|(frac<<13);
    else f=(sign<<31)|((exp+127-15)<<23)|(frac<<13);
    return __uint_as_float(f);
}

// y[o] = sum_k x[k] * W[o*K + k]      W: (N, K) row-major f32
extern "C" __global__ void matvec_t(const float* x, const float* W, float* y, int K, int N){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= N) return;
    const float* w = W + (size_t)o * K;
    float s = 0.0f;
    for(int k = 0; k < K; k++) s += x[k] * w[k];
    y[o] = s;
}

// === FUSED dequant+matvec kernels (one pass, no F32 scratch) ===
// Each thread computes one output element o.  It reads the quantized
// weight row once, dequants on-the-fly into registers, and dot-products
// with x - avoiding the full F32 materialization that the two-pass
// dequant-then-matvec approach requires (~8x less bandwidth).

// Q8_34: 34 B / 32 elems (fp16 scale + 32 x int8)  [GGML Q8_0 / NSVD]
extern "C" __global__ void matvec_q8_34(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 34;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 34;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const signed char* qs = (const signed char*)(blk + 2);
        for(int j = 0; j < 32; j++) acc += x[(b << 5) + j] * d * (float)qs[j];
    }
    y[o] = acc;
}

// Q8_36: 36 B / 32 elems (f32 scale + 32 x int8)  [v16]
extern "C" __global__ void matvec_q8_36(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 36;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 36;
        float d = __uint_as_float(*(const unsigned int*)blk);
        const signed char* qs = (const signed char*)(blk + 4);
        for(int j = 0; j < 32; j++) acc += x[(b << 5) + j] * d * (float)qs[j];
    }
    y[o] = acc;
}

// Q4_0: 18 B / 32 elems (fp16 d + 16 B qs)
extern "C" __global__ void matvec_q4_0(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 18;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 18;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 16; j++){
            unsigned char byte = blk[2 + j];
            acc += x[(b << 5) + j]      * d * (float)((byte & 0xF) - 8);
            acc += x[(b << 5) + 16 + j] * d * (float)((byte >> 4) - 8);
        }
    }
    y[o] = acc;
}

// Q5_0: 22 B / 32 elems (fp16 d + 4 B qh + 16 B qs)
extern "C" __global__ void matvec_q5_0(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 22;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 22;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[6 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
            acc += x[(b << 5) + j] * d * (float)(q4 + high * 16 - 16);
        }
    }
    y[o] = acc;
}

// Q6_0: 26 B / 32 elems (16 B ql + 8 B qh + fp16 d @24)
extern "C" __global__ void matvec_q6_0(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 26;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 26;
        float d = h2f((unsigned short)blk[24] | ((unsigned short)blk[25] << 8));
        for(int j = 0; j < 32; j++){
            unsigned char lb = blk[j & 15];
            int q4 = (j < 16) ? (lb & 0xF) : (lb >> 4);
            int high = (blk[16 + ((2 * j) >> 3)] >> ((2 * j) & 7)) & 3;
            acc += x[(b << 5) + j] * d * (float)(q4 + high * 16 - 32);
        }
    }
    y[o] = acc;
}

// Q5_1: 24 B / 32 elems (fp16 d @0, fp16 m @2, qh u32 @4, 16 B qs nibbles @8)
// y = d*q + m  where q = low4 | (highbit<<4)
extern "C" __global__ void matvec_q5_1(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 24;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 24;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float m = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        unsigned int qh = (unsigned int)blk[4] | ((unsigned int)blk[5] << 8)
                        | ((unsigned int)blk[6] << 16) | ((unsigned int)blk[7] << 24);
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[8 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (int)((qh >> j) & 1u);
            acc += x[(b << 5) + j] * (d * (float)(q4 | (high << 4)) + m);
        }
    }
    y[o] = acc;
}

// Q4_K: 144 B / 256 elems (d, dmin, scales[12], qs[128])
extern "C" __global__ void matvec_q4_k(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 144;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 144;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        float sc_f[8], mn_f[8];
        for(int s = 0; s < 8; s++){
            int sc, mn;
            if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
            else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
                   mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
            sc_f[s] = d * (float)sc;  mn_f[s] = dmin * (float)mn;
        }
        for(int i = 0; i < 256; i++){
            int s = i >> 5, j = i & 31, group = s >> 1;
            unsigned char byte = blk[16 + group * 32 + j];
            int q = (s & 1) ? (byte >> 4) : (byte & 0xF);
            acc += x[(b << 8) + i] * (sc_f[s] * (float)q - mn_f[s]);
        }
    }
    y[o] = acc;
}

// Q5_K: 176 B / 256 elems (d, dmin, scales[12], qh[32], qs[128])
extern "C" __global__ void matvec_q5_k(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 176;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 176;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        float sc_f[8], mn_f[8];
        for(int s = 0; s < 8; s++){
            int sc, mn;
            if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
            else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
                   mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
            sc_f[s] = d * (float)sc;  mn_f[s] = dmin * (float)mn;
        }
        for(int i = 0; i < 256; i++){
            int s = i >> 5, j = i & 31, group = s >> 1;
            unsigned char byte = blk[48 + group * 32 + j];
            int q4 = (s & 1) ? (byte >> 4) : (byte & 0xF);
            int bit = 2 * group + (s & 1);
            int high = (blk[16 + j] >> bit) & 1;
            acc += x[(b << 8) + i] * (sc_f[s] * (float)(q4 + high * 16) - mn_f[s]);
        }
    }
    y[o] = acc;
}

// Q6_K: 210 B / 256 elems (ql[128], qh[64], scales[16] i8, d fp16 @208)
extern "C" __global__ void matvec_q6_k(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 210;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 210;
        float d = h2f((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
        for(int i = 0; i < 256; i++){
            int n2 = i >> 7, h = i & 127, quarter = h >> 5, j = h & 31;
            int qlb = blk[n2 * 64 + (quarter & 1) * 32 + j];
            int nib = (quarter < 2) ? (qlb & 0xF) : (qlb >> 4);
            int hi = (blk[128 + n2 * 32 + j] >> (2 * quarter)) & 3;
            int q = (nib | (hi << 4)) - 32;
            int sc = (int)(signed char)blk[192 + n2 * 8 + quarter * 2 + (j >> 4)];
            acc += x[(b << 8) + i] * d * (float)sc * (float)q;
        }
    }
    y[o] = acc;
}

// F16 fused matvec (2 B / elem)
extern "C" __global__ void matvec_f16(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    const unsigned short* row = (const unsigned short*)(W + (size_t)o * in_dim * 2);
    float acc = 0.0f;
    for(int k = 0; k < in_dim; k++) acc += x[k] * h2f(row[k]);
    y[o] = acc;
}

// BF16 fused matvec (2 B / elem)
extern "C" __global__ void matvec_bf16(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    const unsigned short* row = (const unsigned short*)(W + (size_t)o * in_dim * 2);
    float acc = 0.0f;
    for(int k = 0; k < in_dim; k++){
        unsigned int f = (unsigned int)row[k] << 16;
        acc += x[k] * __uint_as_float(f);
    }
    y[o] = acc;
}

// === OPTIMIZED shared-memory matvec kernels ===
// Load x into shared memory once per block (cooperative), then each
// thread reads sx[] instead of x[] - avoids L1 cache thrashing by
// weight data.  Launch with 128 threads/block for better SM occupancy.
// Pass in_dim*4 as shared_mem bytes.

extern "C" __global__ void matvec_q4_k_smem(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int o = blockIdx.x * blockDim.x + tid;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 144;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 144;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        float sc_f[8], mn_f[8];
        for(int s = 0; s < 8; s++){
            int sc, mn;
            if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
            else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
                   mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
            sc_f[s] = d * (float)sc;  mn_f[s] = dmin * (float)mn;
        }
        for(int i = 0; i < 256; i++){
            int s = i >> 5, j = i & 31, group = s >> 1;
            unsigned char byte = blk[16 + group * 32 + j];
            int q = (s & 1) ? (byte >> 4) : (byte & 0xF);
            acc += sx[(b << 8) + i] * (sc_f[s] * (float)q - mn_f[s]);
        }
    }
    y[o] = acc;
}

extern "C" __global__ void matvec_q4_0_smem(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int o = blockIdx.x * blockDim.x + tid;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 18;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 18;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 16; j++){
            unsigned char byte = blk[2 + j];
            acc += sx[(b << 5) + j]      * d * (float)((byte & 0xF) - 8);
            acc += sx[(b << 5) + 16 + j] * d * (float)((byte >> 4) - 8);
        }
    }
    y[o] = acc;
}

extern "C" __global__ void matvec_q5_0_smem(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int o = blockIdx.x * blockDim.x + tid;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 22;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 22;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[6 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
            acc += sx[(b << 5) + j] * d * (float)(q4 + high * 16 - 16);
        }
    }
    y[o] = acc;
}

extern "C" __global__ void matvec_q5_1_smem(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int o = blockIdx.x * blockDim.x + tid;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 24;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 24;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float m = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        unsigned int qh = (unsigned int)blk[4] | ((unsigned int)blk[5] << 8)
                        | ((unsigned int)blk[6] << 16) | ((unsigned int)blk[7] << 24);
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[8 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (int)((qh >> j) & 1u);
            acc += sx[(b << 5) + j] * (d * (float)(q4 | (high << 4)) + m);
        }
    }
    y[o] = acc;
}

extern "C" __global__ void matvec_q6_k_smem(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int o = blockIdx.x * blockDim.x + tid;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 210;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 210;
        float d = h2f((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
        for(int i = 0; i < 256; i++){
            int n2 = i >> 7, h = i & 127, quarter = h >> 5, j = h & 31;
            int qlb = blk[n2 * 64 + (quarter & 1) * 32 + j];
            int nib = (quarter < 2) ? (qlb & 0xF) : (qlb >> 4);
            int hi = (blk[128 + n2 * 32 + j] >> (2 * quarter)) & 3;
            int q = (nib | (hi << 4)) - 32;
            int sc = (int)(signed char)blk[192 + n2 * 8 + quarter * 2 + (j >> 4)];
            acc += sx[(b << 8) + i] * d * (float)sc * (float)q;
        }
    }
    y[o] = acc;
}

extern "C" __global__ void matvec_f16_smem(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int o = blockIdx.x * blockDim.x + tid;
    if(o >= out_dim) return;
    const unsigned short* row = (const unsigned short*)(W + (size_t)o * in_dim * 2);
    float acc = 0.0f;
    for(int k = 0; k < in_dim; k++) acc += sx[k] * h2f(row[k]);
    y[o] = acc;
}

// === SPLIT-K matvec kernels (high-occupancy for small out_dim) ===
// One block per output element, 32 threads (1 warp).  Each thread
// processes a subset of the quantized blocks, then warp-reduce via
// __shfl_down_sync.  This gives out_dim blocks -> full SM occupancy
// even when out_dim is small (e.g. 896 -> 896 blocks vs 7 blocks
// with the 128-thread smem kernel).  x is loaded into shared memory.

extern "C" __global__ void matvec_q5_0_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 22;
    float acc = 0.0f;
    for(int b = tid; b < nblk; b += blockDim.x){
        const unsigned char* blk = row + (size_t)b * 22;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[6 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
            acc += sx[(b << 5) + j] * d * (float)(q4 + high * 16 - 16);
        }
    }
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

extern "C" __global__ void matvec_q4_0_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 18;
    float acc = 0.0f;
    for(int b = tid; b < nblk; b += blockDim.x){
        const unsigned char* blk = row + (size_t)b * 18;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 16; j++){
            unsigned char byte = blk[2 + j];
            acc += sx[(b << 5) + j]      * d * (float)((byte & 0xF) - 8);
            acc += sx[(b << 5) + 16 + j] * d * (float)((byte >> 4) - 8);
        }
    }
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

extern "C" __global__ void matvec_q4_k_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 144;
    float acc = 0.0f;
    for(int b = tid; b < nblk; b += blockDim.x){
        const unsigned char* blk = row + (size_t)b * 144;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        float sc_f[8], mn_f[8];
        for(int s = 0; s < 8; s++){
            int sc, mn;
            if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
            else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
                   mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
            sc_f[s] = d * (float)sc;  mn_f[s] = dmin * (float)mn;
        }
        for(int i = 0; i < 256; i++){
            int s = i >> 5, j = i & 31, group = s >> 1;
            unsigned char byte = blk[16 + group * 32 + j];
            int q = (s & 1) ? (byte >> 4) : (byte & 0xF);
            acc += sx[(b << 8) + i] * (sc_f[s] * (float)q - mn_f[s]);
        }
    }
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

extern "C" __global__ void matvec_q5_1_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 24;
    float acc = 0.0f;
    for(int b = tid; b < nblk; b += blockDim.x){
        const unsigned char* blk = row + (size_t)b * 24;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float m = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        unsigned int qh = (unsigned int)blk[4] | ((unsigned int)blk[5] << 8)
                        | ((unsigned int)blk[6] << 16) | ((unsigned int)blk[7] << 24);
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[8 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (int)((qh >> j) & 1u);
            acc += sx[(b << 5) + j] * (d * (float)(q4 | (high << 4)) + m);
        }
    }
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

extern "C" __global__ void matvec_q6_k_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 210;
    float acc = 0.0f;
    for(int b = tid; b < nblk; b += blockDim.x){
        const unsigned char* blk = row + (size_t)b * 210;
        float d = h2f((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
        for(int i = 0; i < 256; i++){
            int n2 = i >> 7, h = i & 127, quarter = h >> 5, j = h & 31;
            int qlb = blk[n2 * 64 + (quarter & 1) * 32 + j];
            int nib = (quarter < 2) ? (qlb & 0xF) : (qlb >> 4);
            int hi = (blk[128 + n2 * 32 + j] >> (2 * quarter)) & 3;
            int q = (nib | (hi << 4)) - 32;
            int sc = (int)(signed char)blk[192 + n2 * 8 + quarter * 2 + (j >> 4)];
            acc += sx[(b << 8) + i] * d * (float)sc * (float)q;
        }
    }
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

extern "C" __global__ void matvec_f16_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const unsigned short* row = (const unsigned short*)(W + (size_t)o * in_dim * 2);
    float acc = 0.0f;
    for(int k = tid; k < in_dim; k += blockDim.x) acc += sx[k] * h2f(row[k]);
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

extern "C" __global__ void matvec_q8_34_splitk(const unsigned char* W, const float* x, float* y, int in_dim, int out_dim){
    extern __shared__ float sx[];
    int o = blockIdx.x;
    if(o >= out_dim) return;
    int tid = threadIdx.x;
    for(int i = tid; i < in_dim; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 34;
    float acc = 0.0f;
    for(int b = tid; b < nblk; b += blockDim.x){
        const unsigned char* blk = row + (size_t)b * 34;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const signed char* qs = (const signed char*)(blk + 2);
        for(int j = 0; j < 32; j++) acc += sx[(b << 5) + j] * d * (float)qs[j];
    }
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if(tid == 0) y[o] = acc;
}

// Q1_0: 18 B / 128 elems (fp16 scale + 16 packed bytes), natural order
extern "C" __global__ void dequant_q1_0(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 7, j = idx & 127;
    const unsigned char* blk = in + (size_t)b * 18;
    float scale = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    unsigned char byte = blk[2 + (j >> 3)];
    out[idx] = ((byte >> (j & 7)) & 1) ? scale : -scale;
}

// NSVD Q8: 34 B / 32 elems (fp16 scale + 32 x int8)
extern "C" __global__ void dequant_q8_34(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 5, j = idx & 31;
    const unsigned char* blk = in + (size_t)b * 34;
    float scale = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    out[idx] = scale * (float)(signed char)blk[2 + j];
}

// v16 Q8: 36 B / 32 elems (f32 scale + 32 x int8)
extern "C" __global__ void dequant_q8_36(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 5, j = idx & 31;
    const unsigned char* blk = in + (size_t)b * 36;
    float scale = __uint_as_float(*(const unsigned int*)blk);
    out[idx] = scale * (float)(signed char)blk[4 + j];
}

// F16: raw half -> float
extern "C" __global__ void dequant_f16(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    const unsigned short* h = (const unsigned short*)in;
    out[idx] = h2f(h[idx]);
}

// Q4_0: 18 B / 32 elems (fp16 d + 16 B qs), ggml split nibbles
extern "C" __global__ void dequant_q4_0(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 5, j = idx & 31;
    const unsigned char* blk = in + (size_t)b * 18;
    float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    unsigned char byte = blk[2 + (j & 15)];
    int q = (j < 16 ? (byte & 0xF) : (byte >> 4)) - 8;
    out[idx] = d * (float)q;
}

// Q5_0: 22 B / 32 elems (fp16 d + 4 B qh + 16 B qs)
extern "C" __global__ void dequant_q5_0(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 5, j = idx & 31;
    const unsigned char* blk = in + (size_t)b * 22;
    float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    unsigned char byte = blk[6 + (j & 15)];
    int q4 = (j < 16 ? (byte & 0xF) : (byte >> 4));
    int high = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
    out[idx] = d * (float)(q4 + high * 16 - 16);
}

// Q6_0: 26 B / 32 elems (16 B ql + 8 B qh + fp16 d @24)
extern "C" __global__ void dequant_q6_0(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 5, j = idx & 31;
    const unsigned char* blk = in + (size_t)b * 26;
    unsigned char lb = blk[j & 15];
    int q4 = (j < 16 ? (lb & 0xF) : (lb >> 4));
    int high = (blk[16 + ((2 * j) >> 3)] >> ((2 * j) & 7)) & 3;
    float d = h2f((unsigned short)blk[24] | ((unsigned short)blk[25] << 8));
    out[idx] = d * (float)(q4 + high * 16 - 32);
}

// Q5_1: 24 B / 32 elems (fp16 d @0, fp16 m @2, qh u32 @4, nibbles @8)
extern "C" __global__ void dequant_q5_1(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 5, j = idx & 31;
    const unsigned char* blk = in + (size_t)b * 24;
    float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    float m = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
    unsigned int qh = (unsigned int)blk[4] | ((unsigned int)blk[5] << 8)
                    | ((unsigned int)blk[6] << 16) | ((unsigned int)blk[7] << 24);
    unsigned char byte = blk[8 + (j & 15)];
    int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
    int high = (int)((qh >> j) & 1u);
    out[idx] = d * (float)(q4 | (high << 4)) + m;
}

// Q4_K: 144 B / 256 elems (d, dmin, scales[12], qs[128])
extern "C" __global__ void dequant_q4_k(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 8, i = idx & 255;
    const unsigned char* blk = in + (size_t)b * 144;
    int s = i >> 5, j = i & 31, group = s >> 1;
    int sc, mn;
    if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
    else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
           mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
    float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
    unsigned char byte = blk[16 + group * 32 + j];
    int q = (s & 1) ? (byte >> 4) : (byte & 0xF);
    out[idx] = d * (float)sc * (float)q - dmin * (float)mn;
}

// Q5_K: 176 B / 256 elems (d, dmin, scales[12], qh[32], qs[128])
extern "C" __global__ void dequant_q5_k(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 8, i = idx & 255;
    const unsigned char* blk = in + (size_t)b * 176;
    int s = i >> 5, j = i & 31, group = s >> 1;
    int sc, mn;
    if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
    else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
           mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
    float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
    unsigned char byte = blk[48 + group * 32 + j];
    int q4 = (s & 1) ? (byte >> 4) : (byte & 0xF);
    int bit = 2 * group + (s & 1);
    int high = (blk[16 + j] >> bit) & 1;
    out[idx] = d * (float)sc * (float)(q4 + high * 16) - dmin * (float)mn;
}

// Q6_K: 210 B / 256 elems (ql[128], qh[64], scales[16] i8, d fp16 @208)
extern "C" __global__ void dequant_q6_k(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    int b = idx >> 8, i = idx & 255;
    const unsigned char* blk = in + (size_t)b * 210;
    int n2 = i >> 7, h = i & 127, quarter = h >> 5, j = h & 31;
    int qlb = blk[n2 * 64 + (quarter & 1) * 32 + j];
    int nib = (quarter < 2) ? (qlb & 0xF) : (qlb >> 4);
    int hi = (blk[128 + n2 * 32 + j] >> (2 * quarter)) & 3;
    int q = (nib | (hi << 4)) - 32;
    int sc = (int)(signed char)blk[192 + n2 * 8 + quarter * 2 + (j >> 4)];
    float d = h2f((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
    out[idx] = d * (float)sc * (float)q;
}

// y[i] = x[i] * s[i]   (.ntmre compression singular-value scaling)
extern "C" __global__ void scale_vec(float* x, const float* s, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) x[i] *= s[i];
}

// U[a*rank + r] *= S[r]   (fold singular values into U columns)
extern "C" __global__ void scale_cols(float* U, const float* s, int total, int rank){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < total) U[i] *= s[i % rank];
}

// --- BF16 dequant: brain-float 16 -> f32 (shift left 16 bits) ---
extern "C" __global__ void dequant_bf16(const unsigned char* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    unsigned short h = ((const unsigned short*)in)[idx];
    unsigned int f = (unsigned int)h << 16;
    out[idx] = __uint_as_float(f);
}

// --- RMSNorm: out = x * rsqrt(mean(x^2)+eps) * weight ---
extern "C" __global__ void kernel_rmsnorm(float* out, const float* x, const float* weight, int dim, float eps){
    __shared__ float ssq[1];
    __shared__ float shared[256];
    int tid = threadIdx.x;
    float s = 0.0f;
    for(int i = tid; i < dim; i += blockDim.x) s += x[i] * x[i];
    shared[tid] = s;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if(tid == 0) ssq[0] = shared[0];
    __syncthreads();
    float rms = sqrtf(ssq[0] / (float)dim + eps);
    for(int i = tid; i < dim; i += blockDim.x)
        out[i] = x[i] / rms * weight[i];
}

// --- LayerNorm (with mean subtraction) ---
extern "C" __global__ void kernel_layernorm(float* out, const float* x, const float* weight, int dim, float eps){
    __shared__ float shared[256];
    __shared__ float smean[1];
    __shared__ float svar[1];
    int tid = threadIdx.x;
    float s = 0.0f;
    for(int i = tid; i < dim; i += blockDim.x) s += x[i];
    shared[tid] = s;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if(tid == 0) smean[0] = shared[0] / (float)dim;
    __syncthreads();
    float mean = smean[0];
    s = 0.0f;
    for(int i = tid; i < dim; i += blockDim.x){ float d = x[i] - mean; s += d * d; }
    shared[tid] = s;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if(tid == 0) svar[0] = shared[0] / (float)dim;
    __syncthreads();
    float inv_std = 1.0f / sqrtf(svar[0] + eps);
    for(int i = tid; i < dim; i += blockDim.x)
        out[i] = (x[i] - mean) * inv_std * weight[i];
}

// --- RoPE: apply rotary position embedding to Q and K in-place ---
extern "C" __global__ void kernel_rope(float* q, float* k,
        int n_heads, int n_kv_heads, int head_dim, int pos, float rope_base){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total_q = n_heads * half;
    int total_k = n_kv_heads * half;
    if(tid < total_q){
        int h = tid / half, i = tid % half;
        float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        int base = h * head_dim + i;
        float q0 = q[base], q1 = q[base + half];
        q[base] = q0 * c - q1 * s;
        q[base + half] = q0 * s + q1 * c;
    }
    if(tid < total_k){
        int h = tid / half, i = tid % half;
        float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        int base = h * head_dim + i;
        float k0 = k[base], k1 = k[base + half];
        k[base] = k0 * c - k1 * s;
        k[base + half] = k0 * s + k1 * c;
    }
}

// --- Attention: one block per head, thread per head_dim element ---
// KV cache layout: k_cache[seq_len][n_kv_heads * head_dim], same for v_cache
extern "C" __global__ void kernel_attention(float* out,
        const float* q, const float* k_cache, const float* v_cache,
        int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_stride){
    int head = blockIdx.x;
    if(head >= n_heads) return;
    int kv_h = head * n_kv_heads / n_heads;
    int d = threadIdx.x;
    if(d >= head_dim) return;
    const float* qh = q + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    __shared__ float scores[4096];
    __shared__ float sred[256];
    // Phase 1: compute all scores into shared memory (no race)
    for(int p = d; p < seq_len; p += blockDim.x){
        float dot = 0.0f;
        for(int i = 0; i < head_dim; i++)
            dot += qh[i] * k_cache[p * kv_stride + kv_h * head_dim + i];
        scores[p] = dot * scale;
    }
    __syncthreads();
    // Phase 2: reduce max via tree reduction (deterministic, no race)
    float my_max = -1e30f;
    for(int p = d; p < seq_len; p += blockDim.x)
        if(scores[p] > my_max) my_max = scores[p];
    sred[d] = my_max;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] = fmaxf(sred[d], sred[d + stride]);
        __syncthreads();
    }
    float smax = sred[0];
    // Phase 3: exp and reduce sum via tree reduction (no race)
    float my_sum = 0.0f;
    for(int p = d; p < seq_len; p += blockDim.x){
        float e = expf(scores[p] - smax);
        scores[p] = e;
        my_sum += e;
    }
    sred[d] = my_sum;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] += sred[d + stride];
        __syncthreads();
    }
    float ssum = sred[0];
    // Phase 4: weighted sum
    float out_d = 0.0f;
    for(int p = 0; p < seq_len; p++){
        float w = scores[p] / ssum;
        out_d += w * v_cache[p * kv_stride + kv_h * head_dim + d];
    }
    out[head * head_dim + d] = out_d;
}

// --- SiLU * up: out[i] = (gate[i] / (1+exp(-gate[i]))) * up[i] ---
extern "C" __global__ void kernel_silu_mul(float* out, const float* gate, const float* up, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= dim) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = g * sig * up[i];
}

// --- Residual add: a[i] += b[i] ---
extern "C" __global__ void kernel_add(float* a, const float* b, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= dim) return;
    a[i] += b[i];
}

// --- Scaled accumulate: a[i] += s * b[i]  (MoE expert combine) ---
extern "C" __global__ void kernel_add_scaled(float* a, const float* b, float s, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= dim) return;
    a[i] += s * b[i];
}

// --- In-place scalar scale: x[i] *= s  (layer_output_scale) ---
extern "C" __global__ void kernel_scale_const(float* x, float s, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) x[i] *= s;
}

// --- GeGLU-tanh (gemma4 experts): 0.5*g*(1+tanh(sqrt(2/pi)*(g+0.044715 g^3)))*u ---
extern "C" __global__ void kernel_geglu_mul(float* out, const float* gate, const float* up, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= dim) return;
    float g = gate[i];
    float a = 0.7978845608028654f * (g + 0.044715f * g * g * g);  // sqrt(2/pi)
    out[i] = 0.5f * g * (1.0f + tanhf(a)) * up[i];
}

// =================================================================
// BATCHED PREFILL KERNELS  (process M tokens at once)
// Y[m][o] = sum_k X[m][k] * dequant(W[o][k])  - weight read once for all M
// Grid: (ceil(out_dim/256), M), block: (256, 1)
// =================================================================

extern "C" __global__ void matmul_q8_34(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 34;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 34;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const signed char* qs = (const signed char*)(blk + 2);
        for(int j = 0; j < 32; j++) acc += xrow[(b << 5) + j] * d * (float)qs[j];
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q8_36(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 36;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 36;
        float d = __uint_as_float(*(const unsigned int*)blk);
        const signed char* qs = (const signed char*)(blk + 4);
        for(int j = 0; j < 32; j++) acc += xrow[(b << 5) + j] * d * (float)qs[j];
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q4_0(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 18;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 18;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 16; j++){
            unsigned char byte = blk[2 + j];
            acc += xrow[(b << 5) + j]      * d * (float)((byte & 0xF) - 8);
            acc += xrow[(b << 5) + 16 + j] * d * (float)((byte >> 4) - 8);
        }
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q5_0(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 22;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 22;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        for(int j = 0; j < 32; j++){
            unsigned char byte = blk[6 + (j & 15)];
            int q4 = (j < 16) ? (byte & 0xF) : (byte >> 4);
            int high = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
            acc += xrow[(b << 5) + j] * d * (float)(q4 + high * 16 - 16);
        }
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q6_0(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 5;
    const unsigned char* row = W + (size_t)o * nblk * 26;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 26;
        float d = h2f((unsigned short)blk[24] | ((unsigned short)blk[25] << 8));
        for(int j = 0; j < 32; j++){
            unsigned char lb = blk[j & 15];
            int q4 = (j < 16) ? (lb & 0xF) : (lb >> 4);
            int high = (blk[16 + ((2 * j) >> 3)] >> ((2 * j) & 7)) & 3;
            acc += xrow[(b << 5) + j] * d * (float)(q4 + high * 16 - 32);
        }
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q4_k(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 144;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 144;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        float sc_f[8], mn_f[8];
        for(int s = 0; s < 8; s++){
            int sc, mn;
            if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
            else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
                   mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
            sc_f[s] = d * (float)sc;  mn_f[s] = dmin * (float)mn;
        }
        for(int i = 0; i < 256; i++){
            int s = i >> 5, j = i & 31, group = s >> 1;
            unsigned char byte = blk[16 + group * 32 + j];
            int q = (s & 1) ? (byte >> 4) : (byte & 0xF);
            acc += xrow[(b << 8) + i] * (sc_f[s] * (float)q - mn_f[s]);
        }
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q5_k(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 176;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 176;
        float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        float dmin = h2f((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
        float sc_f[8], mn_f[8];
        for(int s = 0; s < 8; s++){
            int sc, mn;
            if(s < 4){ sc = blk[4 + s] & 63; mn = blk[8 + s] & 63; }
            else { sc = (blk[8 + s] & 0xF) | ((blk[s] >> 6) << 4);
                   mn = (blk[8 + s] >> 4) | ((blk[4 + s] >> 6) << 4); }
            sc_f[s] = d * (float)sc;  mn_f[s] = dmin * (float)mn;
        }
        for(int i = 0; i < 256; i++){
            int s = i >> 5, j = i & 31, group = s >> 1;
            unsigned char byte = blk[48 + group * 32 + j];
            int q4 = (s & 1) ? (byte >> 4) : (byte & 0xF);
            int bit = 2 * group + (s & 1);
            int high = (blk[16 + j] >> bit) & 1;
            acc += xrow[(b << 8) + i] * (sc_f[s] * (float)(q4 + high * 16) - mn_f[s]);
        }
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_q6_k(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    int nblk = in_dim >> 8;
    const unsigned char* row = W + (size_t)o * nblk * 210;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int b = 0; b < nblk; b++){
        const unsigned char* blk = row + (size_t)b * 210;
        float d = h2f((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
        for(int i = 0; i < 256; i++){
            int n2 = i >> 7, h = i & 127, quarter = h >> 5, j = h & 31;
            int qlb = blk[n2 * 64 + (quarter & 1) * 32 + j];
            int nib = (quarter < 2) ? (qlb & 0xF) : (qlb >> 4);
            int hi = (blk[128 + n2 * 32 + j] >> (2 * quarter)) & 3;
            int q = (nib | (hi << 4)) - 32;
            int sc = (int)(signed char)blk[192 + n2 * 8 + quarter * 2 + (j >> 4)];
            acc += xrow[(b << 8) + i] * d * (float)sc * (float)q;
        }
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_f16(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    const unsigned short* row = (const unsigned short*)(W + (size_t)o * in_dim * 2);
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int k = 0; k < in_dim; k++) acc += xrow[k] * h2f(row[k]);
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_bf16(const unsigned char* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    const unsigned short* row = (const unsigned short*)(W + (size_t)o * in_dim * 2);
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int k = 0; k < in_dim; k++){
        unsigned int f = (unsigned int)row[k] << 16;
        acc += xrow[k] * __uint_as_float(f);
    }
    Y[(size_t)m * out_dim + o] = acc;
}

extern "C" __global__ void matmul_f32(const float* W, const float* X, float* Y, int M, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if(o >= out_dim) return;
    const float* row = W + (size_t)o * in_dim;
    const float* xrow = X + (size_t)m * in_dim;
    float acc = 0.0f;
    for(int k = 0; k < in_dim; k++) acc += xrow[k] * row[k];
    Y[(size_t)m * out_dim + o] = acc;
}

// --- Batched RMSNorm: one block per token row (m) ---
extern "C" __global__ void kernel_batch_rmsnorm(float* out, const float* x, const float* weight, int dim, float eps, int M){
    int m = blockIdx.x;
    if(m >= M) return;
    const float* xrow = x + (size_t)m * dim;
    float* orow = out + (size_t)m * dim;
    __shared__ float shared[256];
    int tid = threadIdx.x;
    float s = 0.0f;
    for(int i = tid; i < dim; i += blockDim.x) s += xrow[i] * xrow[i];
    shared[tid] = s;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    float inv = rsqrtf(shared[0] / (float)dim + eps);
    for(int i = tid; i < dim; i += blockDim.x) orow[i] = xrow[i] * inv * weight[i];
}

// --- Batched RoPE: one block per token (m), position = start_pos + m ---
extern "C" __global__ void kernel_batch_rope(float* q, float* k,
        int n_heads, int n_kv_heads, int head_dim, int start_pos, int M, float rope_base){
    int m = blockIdx.x;
    if(m >= M) return;
    int half = head_dim / 2;
    int tid = threadIdx.x;
    int total_q = n_heads * half;
    int total_k = n_kv_heads * half;
    int pos = start_pos + m;
    float* qrow = q + (size_t)m * n_heads * head_dim;
    float* krow = k + (size_t)m * n_kv_heads * head_dim;
    for(int idx = tid; idx < total_q; idx += blockDim.x){
        int h = idx / half, i = idx % half;
        float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        int base = h * head_dim + i;
        float q0 = qrow[base], q1 = qrow[base + half];
        qrow[base] = q0 * c - q1 * s;
        qrow[base + half] = q0 * s + q1 * c;
    }
    for(int idx = tid; idx < total_k; idx += blockDim.x){
        int h = idx / half, i = idx % half;
        float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        int base = h * head_dim + i;
        float k0 = krow[base], k1 = krow[base + half];
        krow[base] = k0 * c - k1 * s;
        krow[base + half] = k0 * s + k1 * c;
    }
}

// --- Batched SiLU * up: out[m][i] = silu(gate[m][i]) * up[m][i] ---
extern "C" __global__ void kernel_batch_silu_mul(float* out, const float* gate, const float* up, int dim, int M){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * M;
    if(idx >= total) return;
    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = g * sig * up[idx];
}

// --- Batched residual add: a[m][i] += b[m][i] ---
extern "C" __global__ void kernel_batch_add(float* a, const float* b, int dim, int M){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * M;
    if(idx >= total) return;
    a[idx] += b[idx];
}

// --- Float-to-half conversion (for F16 KV cache) ---
__device__ unsigned short f2h(float f){
    unsigned x = __float_as_uint(f);
    unsigned sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xff) - 127 + 15;
    unsigned frac = x & 0x7fffff;
    if(exp <= 0){
        if(exp < -10) return (unsigned short)sign;
        frac |= 0x800000;
        int shift = 14 - exp;
        frac >>= shift;
        return (unsigned short)(sign | (frac >> 13));
    } else if(exp >= 31){
        return (unsigned short)(sign | 0x7c00);
    }
    return (unsigned short)(sign | (exp << 10) | (frac >> 13));
}

extern "C" __global__ void kernel_f32_to_f16(const float* in, unsigned short* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    out[idx] = f2h(in[idx]);
}

// --- Attention with F16 KV cache (half the VRAM for cache) ---
extern "C" __global__ void kernel_attention_f16(float* out,
        const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
        int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_stride){
    int head = blockIdx.x;
    if(head >= n_heads) return;
    int kv_h = head * n_kv_heads / n_heads;
    int d = threadIdx.x;
    if(d >= head_dim) return;
    const float* qh = q + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    __shared__ float scores[4096];
    __shared__ float sred[256];
    // Phase 1: compute all scores into shared memory (no race)
    for(int p = d; p < seq_len; p += blockDim.x){
        float dot = 0.0f;
        for(int i = 0; i < head_dim; i++)
            dot += qh[i] * h2f(k_cache[p * kv_stride + kv_h * head_dim + i]);
        scores[p] = dot * scale;
    }
    __syncthreads();
    // Phase 2: reduce max via tree reduction (deterministic, no race)
    float my_max = -1e30f;
    for(int p = d; p < seq_len; p += blockDim.x)
        if(scores[p] > my_max) my_max = scores[p];
    sred[d] = my_max;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] = fmaxf(sred[d], sred[d + stride]);
        __syncthreads();
    }
    float smax = sred[0];
    // Phase 3: exp and reduce sum via tree reduction (no race)
    float my_sum = 0.0f;
    for(int p = d; p < seq_len; p += blockDim.x){
        float e = expf(scores[p] - smax);
        scores[p] = e;
        my_sum += e;
    }
    sred[d] = my_sum;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] += sred[d + stride];
        __syncthreads();
    }
    float ssum = sred[0];
    // Phase 4: weighted sum
    float out_d = 0.0f;
    for(int p = 0; p < seq_len; p++){
        float w = scores[p] / ssum;
        out_d += w * h2f(v_cache[p * kv_stride + kv_h * head_dim + d]);
    }
    out[head * head_dim + d] = out_d;
}

// --- Batched matvec for prefill: y[b*out_dim+o] = sum_k x[b*in_dim+k] * W[o*in_dim+k] ---
extern "C" __global__ void kernel_batch_matvec(
        const float* x, const float* W, float* y,
        int batch, int in_dim, int out_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o >= out_dim) return;
    const float* w = W + (size_t)o * in_dim;
    for(int b = 0; b < batch; b++){
        const float* xb = x + (size_t)b * in_dim;
        float s = 0.0f;
        for(int k = 0; k < in_dim; k++) s += xb[k] * w[k];
        y[(size_t)b * out_dim + o] = s;
    }
}

// --- Fused QKV matvec: 3 outputs from one input, one kernel launch ---
// wq (q_dim x in_dim), wk (k_dim x in_dim), wv (v_dim x in_dim)
extern "C" __global__ void kernel_fused_qkv(
        const float* x, const float* wq, const float* wk, const float* wv,
        float* q, float* k, float* v,
        int in_dim, int q_dim, int k_dim, int v_dim){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int total = q_dim + k_dim + v_dim;
    if(o >= total) return;
    const float* w;
    float* out;
    int out_dim;
    if(o < q_dim){
        w = wq + (size_t)o * in_dim;
        out = q;
        out_dim = q_dim;
    } else if(o < q_dim + k_dim){
        int oo = o - q_dim;
        w = wk + (size_t)oo * in_dim;
        out = k;
        out_dim = k_dim;
    } else {
        int oo = o - q_dim - k_dim;
        w = wv + (size_t)oo * in_dim;
        out = v;
        out_dim = v_dim;
    }
    float s = 0.0f;
    for(int i = 0; i < in_dim; i++) s += x[i] * w[i];
    out[o < q_dim ? o : (o < q_dim + k_dim ? o - q_dim : o - q_dim - k_dim)] = s;
}

// --- Flash-style attention: tiled, no 4096 context limit ---
// Uses online softmax: process keys in tiles, accumulate running max + sum
extern "C" __global__ void kernel_flash_attention(float* out,
        const float* q, const float* k_cache, const float* v_cache,
        int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_stride){
    int head = blockIdx.x;
    if(head >= n_heads) return;
    int kv_h = head * n_kv_heads / n_heads;
    int d = threadIdx.x;
    if(d >= head_dim) return;
    const float* qh = q + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    // Online softmax: running max, running sum, running weighted output
    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;
    // Process keys in tiles of blockDim.x
    for(int tile_start = 0; tile_start < seq_len; tile_start += blockDim.x){
        int p = tile_start + d;
        // Phase 1: compute score for this position (if valid)
        float sc = -1e30f;
        if(p < seq_len){
            float dot = 0.0f;
            for(int i = 0; i < head_dim; i++)
                dot += qh[i] * k_cache[p * kv_stride + kv_h * head_dim + i];
            sc = dot * scale;
        }
        // Phase 2: find tile max via reduction
        __shared__ float tile_max[256];
        __shared__ float tile_sum[256];
        tile_max[d] = sc;
        __syncthreads();
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(d < stride && tile_max[d + stride] > tile_max[d])
                tile_max[d] = tile_max[d + stride];
            __syncthreads();
        }
        float new_max = fmaxf(running_max, tile_max[0]);
        // Phase 3: rescale running values
        float rescale = expf(running_max - new_max);
        running_sum *= rescale;
        running_out *= rescale;
        // Phase 4: compute exp and sum for this tile
        float e = (p < seq_len) ? expf(sc - new_max) : 0.0f;
        tile_sum[d] = e;
        __syncthreads();
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(d < stride) tile_sum[d] += tile_sum[d + stride];
            __syncthreads();
        }
        running_sum += tile_sum[0];
        running_max = new_max;
        // Phase 5: accumulate weighted value
        if(p < seq_len){
            float w = e;
            running_out += w * v_cache[p * kv_stride + kv_h * head_dim + d];
        }
        __syncthreads();
    }
    out[head * head_dim + d] = running_out / running_sum;
}

// --- Batched causal attention (F32 KV cache) ---
// Grid: (n_heads, batch). Each block: one (head, token) pair.
// Token b attends to positions 0..b (causal mask).
extern "C" __global__ void kernel_batch_attention(float* out, const float* q_batch, const float* k_cache, const float* v_cache, int n_heads, int n_kv_heads, int head_dim, int batch, int kv_stride){
    int head = blockIdx.x;
    int b = blockIdx.y;
    if(head >= n_heads || b >= batch) return;
    int kv_h = head * n_kv_heads / n_heads;
    int d = threadIdx.x;
    if(d >= head_dim) return;
    const float* qh = q_batch + (size_t)b * n_heads * head_dim + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    __shared__ float scores[4096];
    __shared__ float sred[256];
    int seq_len = b + 1;
    // Phase 1: compute all scores into shared memory (no race)
    for(int p = d; p < seq_len; p += blockDim.x){
        float dot = 0.0f;
        for(int i = 0; i < head_dim; i++)
            dot += qh[i] * k_cache[(size_t)p * kv_stride + kv_h * head_dim + i];
        scores[p] = dot * scale;
    }
    __syncthreads();
    // Phase 2: reduce max via tree reduction (deterministic, no race)
    float my_max = -1e30f;
    for(int p = d; p < seq_len; p += blockDim.x)
        if(scores[p] > my_max) my_max = scores[p];
    sred[d] = my_max;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] = fmaxf(sred[d], sred[d + stride]);
        __syncthreads();
    }
    float smax = sred[0];
    // Phase 3: exp and reduce sum via tree reduction (no race)
    float my_sum = 0.0f;
    for(int p = d; p < seq_len; p += blockDim.x){
        float e = expf(scores[p] - smax);
        scores[p] = e;
        my_sum += e;
    }
    sred[d] = my_sum;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] += sred[d + stride];
        __syncthreads();
    }
    float ssum = sred[0];
    // Phase 4: weighted sum
    float out_d = 0.0f;
    for(int p = 0; p < seq_len; p++){
        float wgt = scores[p] / ssum;
        out_d += wgt * v_cache[(size_t)p * kv_stride + kv_h * head_dim + d];
    }
    out[(size_t)b * n_heads * head_dim + head * head_dim + d] = out_d;
}

// --- Batched causal attention (F16 KV cache) ---
extern "C" __global__ void kernel_batch_attention_f16(float* out, const float* q_batch, const unsigned short* k_cache, const unsigned short* v_cache, int n_heads, int n_kv_heads, int head_dim, int batch, int kv_stride){
    int head = blockIdx.x;
    int b = blockIdx.y;
    if(head >= n_heads || b >= batch) return;
    int kv_h = head * n_kv_heads / n_heads;
    int d = threadIdx.x;
    if(d >= head_dim) return;
    const float* qh = q_batch + (size_t)b * n_heads * head_dim + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    __shared__ float scores[4096];
    __shared__ float sred[256];
    int seq_len = b + 1;
    // Phase 1: compute all scores into shared memory (no race)
    for(int p = d; p < seq_len; p += blockDim.x){
        float dot = 0.0f;
        for(int i = 0; i < head_dim; i++)
            dot += qh[i] * h2f(k_cache[(size_t)p * kv_stride + kv_h * head_dim + i]);
        scores[p] = dot * scale;
    }
    __syncthreads();
    // Phase 2: reduce max via tree reduction (deterministic, no race)
    float my_max = -1e30f;
    for(int p = d; p < seq_len; p += blockDim.x)
        if(scores[p] > my_max) my_max = scores[p];
    sred[d] = my_max;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] = fmaxf(sred[d], sred[d + stride]);
        __syncthreads();
    }
    float smax = sred[0];
    // Phase 3: exp and reduce sum via tree reduction (no race)
    float my_sum = 0.0f;
    for(int p = d; p < seq_len; p += blockDim.x){
        float e = expf(scores[p] - smax);
        scores[p] = e;
        my_sum += e;
    }
    sred[d] = my_sum;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(d < stride) sred[d] += sred[d + stride];
        __syncthreads();
    }
    float ssum = sred[0];
    // Phase 4: weighted sum
    float out_d = 0.0f;
    for(int p = 0; p < seq_len; p++){
        float wgt = scores[p] / ssum;
        out_d += wgt * h2f(v_cache[(size_t)p * kv_stride + kv_h * head_dim + d]);
    }
    out[(size_t)b * n_heads * head_dim + head * head_dim + d] = out_d;
}

// --- Flash-style attention with F16 KV cache (no 4096 limit) ---
extern "C" __global__ void kernel_flash_attention_f16(float* out, const float* q, const unsigned short* k_cache, const unsigned short* v_cache, int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_stride){
    int head = blockIdx.x;
    if(head >= n_heads) return;
    int kv_h = head * n_kv_heads / n_heads;
    int d = threadIdx.x;
    if(d >= head_dim) return;
    const float* qh = q + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;
    for(int tile_start = 0; tile_start < seq_len; tile_start += blockDim.x){
        int p = tile_start + d;
        float sc = -1e30f;
        if(p < seq_len){
            float dot = 0.0f;
            for(int i = 0; i < head_dim; i++)
                dot += qh[i] * h2f(k_cache[(size_t)p * kv_stride + kv_h * head_dim + i]);
            sc = dot * scale;
        }
        __shared__ float tile_max[256];
        __shared__ float tile_sum[256];
        tile_max[d] = sc;
        __syncthreads();
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(d < stride && tile_max[d + stride] > tile_max[d])
                tile_max[d] = tile_max[d + stride];
            __syncthreads();
        }
        float new_max = fmaxf(running_max, tile_max[0]);
        float rescale = expf(running_max - new_max);
        running_sum *= rescale;
        running_out *= rescale;
        float e = (p < seq_len) ? expf(sc - new_max) : 0.0f;
        tile_sum[d] = e;
        __syncthreads();
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(d < stride) tile_sum[d] += tile_sum[d + stride];
            __syncthreads();
        }
        running_sum += tile_sum[0];
        running_max = new_max;
        if(p < seq_len){
            float w = e;
            running_out += w * h2f(v_cache[(size_t)p * kv_stride + kv_h * head_dim + d]);
        }
        __syncthreads();
    }
    out[head * head_dim + d] = running_out / running_sum;
}
"""

# weight-type tags understood by the accelerator
G_F32 = 'f32'
G_F16 = 'f16'        # raw half precision
G_Q1_0 = 'q1_0'      # NSVD 18B/128
G_Q8_34 = 'q8_34'    # NSVD 34B/32 (fp16 scale)
G_Q8_36 = 'q8_36'    # v16  36B/32 (f32 scale)
G_Q4_0 = 'q4_0'      # ggml 18B/32
G_Q5_0 = 'q5_0'      # ggml 22B/32
G_Q5_1 = 'q5_1'      # ggml 24B/32 (fp16 scale + fp16 min)
G_Q6_0 = 'q6_0'      # ggml 26B/32
G_Q4_K = 'q4_k'      # ggml 144B/256
G_Q5_K = 'q5_k'      # ggml 176B/256
G_Q6_K = 'q6_k'      # ggml 210B/256
G_BF16 = 'bf16'      # brain-float 16 (2 bytes, shift-left)

# ggml block geometry: tag -> (elems per block, bytes per block)
_GGML_BLOCKS = {
    G_Q4_0: (32, 18), G_Q5_0: (32, 22), G_Q5_1: (32, 24), G_Q6_0: (32, 26),
    G_Q4_K: (256, 144), G_Q5_K: (256, 176), G_Q6_K: (256, 210),
}

# gtype tag -> GpuMM attribute holding the NVRTC dequant kernel handle
_DEQUANT_FNS = {
    G_F16: 'f_f16', G_Q4_0: 'f_q4_0', G_Q5_0: 'f_q5_0', G_Q5_1: 'f_q5_1',
    G_Q6_0: 'f_q6_0', G_Q4_K: 'f_q4_k', G_Q5_K: 'f_q5_k', G_Q6_K: 'f_q6_k',
    G_BF16: 'f_bf16',
}

# gtype tag -> GpuMM attribute holding the FUSED dequant+matvec kernel.
# When a gtype is present here, matvec_gpu / matvec use the single-pass fused
# kernel instead of the two-pass dequant-then-matvec (saves the full F32
# scratch materialization - ~8x less memory bandwidth).
_FUSED_MV = {
    G_Q8_34: 'f_mv_q8_34', G_Q8_36: 'f_mv_q8_36',
    G_Q4_0: 'f_mv_q4_0', G_Q5_0: 'f_mv_q5_0', G_Q5_1: 'f_mv_q5_1',
    G_Q6_0: 'f_mv_q6_0',
    G_Q4_K: 'f_mv_q4_k', G_Q5_K: 'f_mv_q5_k', G_Q6_K: 'f_mv_q6_k',
    G_F16: 'f_mv_f16', G_BF16: 'f_mv_bf16',
}

# block size (elems) each fused kernel expects in_dim to be a multiple of.
# F16/BF16 are element-wise so any in_dim works (block = 1).
_FUSED_MV_BLOCK = {
    G_Q8_34: 32, G_Q8_36: 32, G_Q4_0: 32, G_Q5_0: 32, G_Q5_1: 32, G_Q6_0: 32,
    G_Q4_K: 256, G_Q5_K: 256, G_Q6_K: 256,
    G_F16: 1, G_BF16: 1,
}

# Shared-memory fused matvec kernels (x loaded into smem once per block).
# Used when the weight is resident and in_dim fits in shared memory (< 48 KB).
_SMEM_MV = {
    G_Q4_K: 'f_mv_q4_k_smem', G_Q4_0: 'f_mv_q4_0_smem',
    G_Q5_0: 'f_mv_q5_0_smem', G_Q5_1: 'f_mv_q5_1_smem',
    G_Q6_K: 'f_mv_q6_k_smem',
    G_F16: 'f_mv_f16_smem',
}
_SMEM_MAX_DIM = 12288  # 48 KB / 4 bytes - max in_dim for shared-mem path

# Split-K high-occupancy kernels: 1 block per output element, 32 threads,
# warp-level reduction.  Used when out_dim is small (< _SPLITK_THRESHOLD)
# so the 128-thread smem kernel would underutilise the GPU (too few blocks).
_SPLITK_MV = {
    G_Q4_K: 'f_mv_q4_k_splitk', G_Q4_0: 'f_mv_q4_0_splitk',
    G_Q5_0: 'f_mv_q5_0_splitk', G_Q5_1: 'f_mv_q5_1_splitk',
    G_Q6_K: 'f_mv_q6_k_splitk',
    G_F16: 'f_mv_f16_splitk', G_Q8_34: 'f_mv_q8_34_splitk',
}
# Below this out_dim, split-K (1 block per output element) gives more
# blocks than the 128-thread smem kernel -> full SM occupancy.  Set high
# enough to cover FFN gate/up (hidden_dim) and Q/K/O projections which
# would otherwise underutilise the GPU (e.g. 38 blocks on 48 SMs).
# Vocab-size output projections (32k+) stay on the regular kernel.
_SPLITK_THRESHOLD = 8192

def _fused_ok(gtype, in_dim):
    """True if the fused kernel can handle this gtype + in_dim alignment."""
    blk = _FUSED_MV_BLOCK.get(gtype)
    return blk is not None and (in_dim % blk == 0)

# gtype tag -> GpuMM attribute holding the BATCHED fused dequant+matmul kernel.
# Same block-alignment constraint as the single-token fused matvec kernels.

_CTYPES_OK = False


def _ensure_ctypes():
    """Workaround: some sealed exe runtimes drop ctypes from module globals."""
    global _CTYPES_OK
    if _CTYPES_OK:
        return
    if 'ctypes' not in globals():
        import ctypes as _ct
        globals()['ctypes'] = _ct
    _CTYPES_OK = True

_BATCH_MV = {
    G_Q8_34: 'f_mm_q8_34', G_Q8_36: 'f_mm_q8_36',
    G_Q4_0: 'f_mm_q4_0', G_Q5_0: 'f_mm_q5_0', G_Q6_0: 'f_mm_q6_0',
    G_Q4_K: 'f_mm_q4_k', G_Q5_K: 'f_mm_q5_k', G_Q6_K: 'f_mm_q6_k',
    G_F16: 'f_mm_f16', G_BF16: 'f_mm_bf16',
}


class GpuMM:
    """Keeps raw quantized weights on the GPU; dequants into a scratch
    buffer and matvecs per token.  Falls back gracefully: construct via
    GpuMM.try_create() which returns None when CUDA is unavailable."""

    def __init__(self, device=None):
        from engine.bonsai_pc_app import Cuda
        if device is None:
            # Default to device 1 (the 12 GB card on the user's dual-GPU
            # rig, same convention as bonsai_gpu.py); overridable.
            device = int(os.environ.get('NTMRE_GPU_DEVICE', '1'))
        self.cu = Cuda(device=device)
        self.device = device
        mod = self.cu.compile(KERNELS)
        self.f_matvec = self.cu.func(mod, "matvec_t")
        # Fused dequant+matvec kernels (one pass, no F32 scratch)
        self.f_mv_q8_34 = self.cu.func(mod, "matvec_q8_34")
        self.f_mv_q8_36 = self.cu.func(mod, "matvec_q8_36")
        self.f_mv_q4_0  = self.cu.func(mod, "matvec_q4_0")
        self.f_mv_q5_0  = self.cu.func(mod, "matvec_q5_0")
        self.f_mv_q5_1  = self.cu.func(mod, "matvec_q5_1")
        self.f_mv_q6_0  = self.cu.func(mod, "matvec_q6_0")
        self.f_mv_q4_k  = self.cu.func(mod, "matvec_q4_k")
        self.f_mv_q5_k  = self.cu.func(mod, "matvec_q5_k")
        self.f_mv_q6_k  = self.cu.func(mod, "matvec_q6_k")
        self.f_mv_f16   = self.cu.func(mod, "matvec_f16")
        self.f_mv_bf16  = self.cu.func(mod, "matvec_bf16")
        # Optimized shared-memory fused matvec kernels (x in smem, 128 thd/blk)
        self.f_mv_q4_k_smem = self.cu.func(mod, "matvec_q4_k_smem")
        self.f_mv_q4_0_smem = self.cu.func(mod, "matvec_q4_0_smem")
        self.f_mv_q5_0_smem = self.cu.func(mod, "matvec_q5_0_smem")
        self.f_mv_q5_1_smem = self.cu.func(mod, "matvec_q5_1_smem")
        self.f_mv_q6_k_smem = self.cu.func(mod, "matvec_q6_k_smem")
        self.f_mv_f16_smem  = self.cu.func(mod, "matvec_f16_smem")
        # Split-K high-occupancy kernels (1 block/output, 32 thd, warp-reduce)
        self.f_mv_q5_0_splitk = self.cu.func(mod, "matvec_q5_0_splitk")
        self.f_mv_q5_1_splitk = self.cu.func(mod, "matvec_q5_1_splitk")
        self.f_mv_q4_0_splitk = self.cu.func(mod, "matvec_q4_0_splitk")
        self.f_mv_q4_k_splitk = self.cu.func(mod, "matvec_q4_k_splitk")
        self.f_mv_q6_k_splitk = self.cu.func(mod, "matvec_q6_k_splitk")
        self.f_mv_f16_splitk  = self.cu.func(mod, "matvec_f16_splitk")
        self.f_mv_q8_34_splitk = self.cu.func(mod, "matvec_q8_34_splitk")
        # Batched (prefill) fused dequant+matmul kernels - M tokens at once
        self.f_mm_q8_34 = self.cu.func(mod, "matmul_q8_34")
        self.f_mm_q8_36 = self.cu.func(mod, "matmul_q8_36")
        self.f_mm_q4_0  = self.cu.func(mod, "matmul_q4_0")
        self.f_mm_q5_0  = self.cu.func(mod, "matmul_q5_0")
        self.f_mm_q6_0  = self.cu.func(mod, "matmul_q6_0")
        self.f_mm_q4_k  = self.cu.func(mod, "matmul_q4_k")
        self.f_mm_q5_k  = self.cu.func(mod, "matmul_q5_k")
        self.f_mm_q6_k  = self.cu.func(mod, "matmul_q6_k")
        self.f_mm_f16   = self.cu.func(mod, "matmul_f16")
        self.f_mm_bf16  = self.cu.func(mod, "matmul_bf16")
        self.f_mm_f32   = self.cu.func(mod, "matmul_f32")
        self.f_q1_0 = self.cu.func(mod, "dequant_q1_0")
        self.f_q8_34 = self.cu.func(mod, "dequant_q8_34")
        self.f_q8_36 = self.cu.func(mod, "dequant_q8_36")
        self.f_scale = self.cu.func(mod, "scale_vec")
        self.f_scols = self.cu.func(mod, "scale_cols")
        self.f_f16 = self.cu.func(mod, "dequant_f16")
        self.f_q4_0 = self.cu.func(mod, "dequant_q4_0")
        self.f_q5_0 = self.cu.func(mod, "dequant_q5_0")
        self.f_q5_1 = self.cu.func(mod, "dequant_q5_1")
        self.f_q6_0 = self.cu.func(mod, "dequant_q6_0")
        self.f_q4_k = self.cu.func(mod, "dequant_q4_k")
        self.f_q5_k = self.cu.func(mod, "dequant_q5_k")
        self.f_q6_k = self.cu.func(mod, "dequant_q6_k")
        self.f_bf16 = self.cu.func(mod, "dequant_bf16")
        self.f_rmsnorm = self.cu.func(mod, "kernel_rmsnorm")
        self.f_layernorm = self.cu.func(mod, "kernel_layernorm")
        self.f_rope = self.cu.func(mod, "kernel_rope")
        self.f_attn = self.cu.func(mod, "kernel_attention")
        self.f_silu = self.cu.func(mod, "kernel_silu_mul")
        self.f_geglu = self.cu.func(mod, "kernel_geglu_mul")
        self.f_add = self.cu.func(mod, "kernel_add")
        self.f_add_scaled = self.cu.func(mod, "kernel_add_scaled")
        self.f_scale_const = self.cu.func(mod, "kernel_scale_const")
        self.f_f32_to_f16 = self.cu.func(mod, "kernel_f32_to_f16")
        self.f_attn_f16 = self.cu.func(mod, "kernel_attention_f16")
        self.f_batch_matvec = self.cu.func(mod, "kernel_batch_matvec")
        self.f_fused_qkv = self.cu.func(mod, "kernel_fused_qkv")
        self.f_flash_attn = self.cu.func(mod, "kernel_flash_attention")
        self.f_flash_attn_f16 = self.cu.func(mod, "kernel_flash_attention_f16")
        self.f_batch_rmsnorm = self.cu.func(mod, "kernel_batch_rmsnorm")
        self.f_batch_rope = self.cu.func(mod, "kernel_batch_rope")
        self.f_batch_attn = self.cu.func(mod, "kernel_batch_attention")
        self.f_batch_attn_f16 = self.cu.func(mod, "kernel_batch_attention_f16")
        self.f_batch_silu = self.cu.func(mod, "kernel_batch_silu_mul")
        self.f_batch_add = self.cu.func(mod, "kernel_batch_add")
        self._raw = {}          # name -> (ptr, gtype, out_dim, in_dim)
        self._scratch = None    # dequant target / matvec weight
        self._scratch_n = 0
        self._scratch2 = None   # .ntmre compression factor U
        self._scratch2_n = 0
        self._scratch3 = None   # .ntmre compression materialized N_c
        self._scratch3_n = 0
        self._x = None          # persistent input vector buffer
        self._x_n = 0
        # VRAM budget for RESIDENT weights - use 99% of free VRAM (was 97%).
        # The extra 2% lets the streaming engine keep more layers resident
        # before falling back to RAM, which is the biggest speed win for
        # large models that don't fit entirely in VRAM.
        _frac = float(os.environ.get('NTMRE_RESIDENT_FRACTION', '0.99'))
        try:
            free = gpu_info.get_free_vram(device)
            if free > 0:
                self._budget = int(free * _frac)
                print(f"[Qwen35-GPU] CUDA device {device}, "
                      f"{free / 1e9:.1f} GB free -> budget {self._budget / 1e9:.1f} GB")
            else:
                raise RuntimeError("nvidia-smi returned zero free VRAM")
        except Exception as _e:
            self._budget = int(os.environ.get('NTMRE_GPU_BUDGET_MB', '2200')) * 1024 * 1024
            print(f"[Qwen35-GPU] CUDA device {device}, "
                  f"resident budget {self._budget / 1e9:.1f} GB + streaming ({_e})")

    @staticmethod
    def try_create(device=None):
        if os.environ.get('NIGHTMARE_FORCE_CPU') == '1':
            return None
        try:
            from nightmare_gpu_discovery import has_gpu_accelerator
            if not has_gpu_accelerator():
                return None
        except Exception:
            pass
        try:
            return GpuMM(device=device)
        except Exception as e:
            return None

    # ── buffer management ─────────────────────────────────────────
    def _bind(self):
        """Make this GpuMM's CUDA context current on the calling thread.

        CUDA contexts are thread-local: the suite constructs the engine on
        the GUI thread but runs tokens on a worker (generate_thread).  A
        GpuMM created on one thread leaves its context current only there, so
        any later op on another thread hits CUDA_ERROR_INVALID_CONTEXT (201).
        Every public op must bind explicitly because the raw cuMemcpyHtoD /
        cuMemsetD32 calls below do not go through the Cuda wrapper that would
        otherwise bind for us."""
        try:
            self.cu._ensure_ctx()
        except Exception:
            pass

    def _ensure_scratch(self, n_elems):
        self._bind()
        if self._scratch_n < n_elems:
            if self._scratch is not None:
                self.cu.free(self._scratch)
            self._scratch = self.cu.alloc(n_elems * 4)
            self._scratch_n = n_elems

    def _ensure_x(self, n_elems):
        self._bind()
        if self._x_n < n_elems:
            if self._x is not None:
                self.cu.free(self._x)
            self._x = self.cu.alloc(n_elems * 4)
            self._x_n = n_elems

    def _ensure_scratch2(self, n_elems):
        if self._scratch2_n < n_elems:
            if self._scratch2 is not None:
                self.cu.free(self._scratch2)
            self._scratch2 = self.cu.alloc(n_elems * 4)
            self._scratch2_n = n_elems

    def _ensure_scratch3(self, n_elems):
        if self._scratch3_n < n_elems:
            if self._scratch3 is not None:
                self.cu.free(self._scratch3)
            self._scratch3 = self.cu.alloc(n_elems * 4)
            self._scratch3_n = n_elems

    def _refresh_budget(self, fraction=0.90, headroom=256 * 1024 * 1024):
        """Re-query free VRAM and set the resident weight budget to a fraction
        of what is actually free right now (minus headroom).  The budget set
        in __init__ goes stale once the host engine allocates its own buffers
        and KV cache, so callers should invoke this after those allocations."""
        try:
            free = gpu_info.get_free_vram(self.device)
            if free > 0:
                self._budget = max(0, int(free * fraction) - int(headroom))
                return self._budget
        except Exception:
            pass
        return self._budget

    def upload_raw(self, name, raw, gtype, out_dim, in_dim):
        self._bind()
        """Upload raw quantized/f32 bytes once; kept on device.  When the
        resident budget is exhausted the tensor is marked for streaming:
        the engine re-supplies the raw bytes on every matvec."""
        if name in self._raw:
            return
        nbytes = len(raw)
        # Small weights (biases, norms - typically < 1 MB) bypass the
        # budget pre-check: there is always a little VRAM headroom for them
        # (the budget is 99 % of free, leaving ~1 %).  The layer code assumes
        # 1-D weights stay resident and doesn't handle 'stream' for them.
        _SMALL = 1024 * 1024
        if nbytes > self._budget and nbytes >= _SMALL:
            self._raw[name] = ('stream', gtype, int(out_dim), int(in_dim))
            return
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = np.ascontiguousarray(arr)
        try:
            ptr = self.cu.alloc(arr.nbytes)
        except RuntimeError:
            # Budget was stale - the host engine allocated its own buffers/KV
            # after GpuMM was created.  Re-query free VRAM, leave headroom, and
            # retry once so a single failed alloc doesn't force every later
            # weight to stream (the old cliff-edge nuked the whole budget).
            self._refresh_budget(0.99, 16 * 1024 * 1024)
            if nbytes > self._budget:
                self._raw[name] = ('stream', gtype, int(out_dim), int(in_dim))
                return
            try:
                ptr = self.cu.alloc(arr.nbytes)
            except RuntimeError:
                self._budget = 0   # card is genuinely full: stream the rest
                self._raw[name] = ('stream', gtype, int(out_dim), int(in_dim))
                return
        self._budget -= nbytes
        r = self.cu.cu.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes)
        if r != 0:
            raise RuntimeError("HtoD %d" % r)
        self._raw[name] = (ptr, gtype, int(out_dim), int(in_dim))

    @staticmethod
    def _weight_bytes(gtype, out_dim, in_dim):
        """Approximate byte size of a weight tensor from its gtype & dims."""
        n = out_dim * in_dim
        if gtype == G_F32 or gtype == G_F16 or gtype == G_BF16:
            return n * (4 if gtype == G_F32 else 2)
        if gtype == G_Q1_0:
            return (n // 128) * 18
        if gtype in (G_Q8_34, G_Q8_36):
            return (n // 32) * (34 if gtype == G_Q8_34 else 36)
        try:
            eb, bb = _GGML_BLOCKS[gtype]
            return (n // eb) * bb
        except Exception:
            return n * 4

    def _evict_resident(self, min_bytes=256 * 1024):
        self._bind()
        """Free large VRAM-resident weights so streamed weights can allocate
        temp buffers.  Only triggered when VRAM is genuinely full (small
        GPUs running large models).  Small weights (biases, norms) are kept
        resident so the layer code that assumes they stay resident isn't
        broken.  Evicted weights are re-supplied from RAM on next use.
        On GPUs with adequate VRAM this is a no-op."""
        for wname, entry in list(self._raw.items()):
            ptr = entry[0]
            if ptr == 'stream':
                continue
            gtype, out_dim, in_dim = entry[1], entry[2], entry[3]
            if self._weight_bytes(gtype, out_dim, in_dim) < min_bytes:
                continue  # keep small weights (biases, norms) resident
            try:
                self.cu.free(ptr)
            except Exception:
                pass
            self._raw[wname] = ('stream', gtype, out_dim, in_dim)
        self._refresh_budget(0.99, 16 * 1024 * 1024)

    def _weight_ptr(self, name, raw):
        self._bind()
        """(device ptr, gtype, out_dim, in_dim, is_temp) for a weight.
        Streamed weights get a private per-call device buffer (the caller
        frees it) so concurrent factors never share a staging buffer."""
        ptr, gtype, out_dim, in_dim = self._raw[name]
        if ptr != 'stream':
            return ptr, gtype, out_dim, in_dim, False
        if raw is None:
            raise ValueError(f"{name}: streamed weight needs raw bytes")
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = np.ascontiguousarray(arr)
        try:
            p = self.cu.alloc(len(raw))
        except RuntimeError:
            # VRAM full from resident weights - evict them and retry.
            # This only happens on VRAM-constrained GPUs; on adequate
            # hardware the first alloc always succeeds.
            self._evict_resident()
            p = self.cu.alloc(len(raw))
        r = self.cu.cu.cuMemcpyHtoD_v2(p, ctypes.c_void_p(arr.ctypes.data), len(raw))
        if r != 0:
            self.cu.free(p)
            raise RuntimeError("HtoD stream %d" % r)
        return p, gtype, out_dim, in_dim, True

    def has(self, name):
        return name in self._raw

    def resident_ptr(self, name):
        """Device pointer for `name`, or None if it is not resident.

        `_raw` is a heterogeneous dict.  Resident entries are
        (ptr, gtype, out_dim, in_dim); entries the VRAM budget could not
        hold are ('stream', gtype, out_dim, in_dim).  `has()` is True for
        both, so callers that dereference [0] must use this instead.
        """
        e = self._raw.get(name)
        if e is None or isinstance(e[0], str):
            return None
        return e[0]

    def ensure_f32(self, name, arr, out_dim, in_dim):
        """Force `name` resident as f32 and return its device pointer.

        upload_raw() early-returns when the name already exists, so a
        'stream' placeholder could never be upgraded.  Drop the placeholder
        first.  Returns None only if the card genuinely cannot take it.
        """
        if self.resident_ptr(name) is None:
            self._raw.pop(name, None)
            self.upload_f32(name, arr, out_dim, in_dim)
        return self.resident_ptr(name)

    # ── compute ───────────────────────────────────────────────────
    def _dequant(self, ptr, gtype, n):
        self._ensure_scratch(n)
        grid = ((n + 255) // 256, 1)
        if gtype == G_Q1_0:
            self.cu.launch(self.f_q1_0, grid, (256, 1),
                           [ctypes.c_ulonglong(ptr.value), ctypes.c_ulonglong(self._scratch.value), n])
        elif gtype == G_Q8_34:
            self.cu.launch(self.f_q8_34, grid, (256, 1),
                           [ctypes.c_ulonglong(ptr.value), ctypes.c_ulonglong(self._scratch.value), n])
        elif gtype == G_Q8_36:
            self.cu.launch(self.f_q8_36, grid, (256, 1),
                           [ctypes.c_ulonglong(ptr.value), ctypes.c_ulonglong(self._scratch.value), n])
        elif gtype in _DEQUANT_FNS:
            fn = getattr(self, _DEQUANT_FNS[gtype])
            self.cu.launch(fn, grid, (256, 1),
                           [ctypes.c_ulonglong(ptr.value), ctypes.c_ulonglong(self._scratch.value), n])
        else:
            raise ValueError(gtype)
        return self._scratch

    def _matvec_dev(self, x_ptr, w_ptr, k, n, out=None):
        self._bind()
        if out is None:
            out = self.cu.alloc(n * 4)
        self.cu.launch(self.f_matvec, ((n + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(x_ptr.value), ctypes.c_ulonglong(w_ptr.value),
                        ctypes.c_ulonglong(out.value), k, n])
        return out

    def matvec(self, x, name, raw=None):
        """y = x @ W.T   x: (in,) f32 ndarray -> (out,) f32 ndarray.
        Uses the fused dequant+matvec kernel for quantized weights."""
        _ensure_ctypes()
        self._bind()
        ptr, gtype, out_dim, in_dim, temp = self._weight_ptr(name, raw)
        try:
            x = np.ascontiguousarray(x, dtype=np.float32).ravel()
            assert x.size == in_dim, f"{name}: x {x.size} != in {in_dim}"
            self._ensure_x(in_dim)
            r = self.cu.cu.cuMemcpyHtoD_v2(self._x, ctypes.c_void_p(x.ctypes.data), x.nbytes)
            if r != 0:
                raise RuntimeError("HtoD x %d" % r)
            yp = self.cu.alloc(out_dim * 4)
            if gtype == G_F32:
                self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(self._x.value),
                                ctypes.c_ulonglong(ptr.value),
                                ctypes.c_ulonglong(yp.value),
                                in_dim, out_dim])
            elif _fused_ok(gtype, in_dim):
                splitk_attr = _SPLITK_MV.get(gtype)
                if (splitk_attr is not None and out_dim < _SPLITK_THRESHOLD
                        and in_dim <= _SMEM_MAX_DIM):
                    fn = getattr(self, splitk_attr)
                    self.cu.launch(fn, (out_dim, 1), (32, 1),
                                   [ctypes.c_ulonglong(ptr.value),
                                    ctypes.c_ulonglong(self._x.value),
                                    ctypes.c_ulonglong(yp.value),
                                    in_dim, out_dim],
                                   shared_mem=in_dim * 4)
                else:
                    smem_attr = _SMEM_MV.get(gtype)
                    if smem_attr is not None and in_dim <= _SMEM_MAX_DIM:
                        fn = getattr(self, smem_attr)
                        self.cu.launch(fn, ((out_dim + 127) // 128, 1), (128, 1),
                                       [ctypes.c_ulonglong(ptr.value),
                                        ctypes.c_ulonglong(self._x.value),
                                        ctypes.c_ulonglong(yp.value),
                                        in_dim, out_dim],
                                       shared_mem=in_dim * 4)
                    else:
                        fn = getattr(self, _FUSED_MV[gtype])
                        self.cu.launch(fn, ((out_dim + 255) // 256, 1), (256, 1),
                                       [ctypes.c_ulonglong(ptr.value),
                                        ctypes.c_ulonglong(self._x.value),
                                        ctypes.c_ulonglong(yp.value),
                                        in_dim, out_dim])
            else:
                w = self._dequant(ptr, gtype, out_dim * in_dim)
                self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(self._x.value),
                                ctypes.c_ulonglong(w.value),
                                ctypes.c_ulonglong(yp.value),
                                in_dim, out_dim])
            y = self.cu.from_gpu(yp, (out_dim,))
            self.cu.free(yp)
            return y
        finally:
            if temp:
                self.cu.free(ptr)

    # ── GPU-resident operations (no CPU download between ops) ────────
    def matvec_gpu(self, x_ptr, name, out_ptr=None, raw=None):
        """matvec but keeps result on GPU (returns device ptr, no download).
        x_ptr is a cuMemAlloc_v2 ptr; result written to out_ptr (allocated if None).
        Uses the fused dequant+matvec kernel for quantized weights (one pass,
        no F32 scratch) and the plain F32 matvec for float weights."""
        _ensure_ctypes()
        self._bind()
        ptr, gtype, out_dim, in_dim, temp = self._weight_ptr(name, raw)
        try:
            if out_ptr is None:
                out_ptr = self.cu.alloc(out_dim * 4)
            if gtype == G_F32:
                self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(x_ptr.value),
                                ctypes.c_ulonglong(ptr.value),
                                ctypes.c_ulonglong(out_ptr.value),
                                in_dim, out_dim])
            elif _fused_ok(gtype, in_dim):
                splitk_attr = _SPLITK_MV.get(gtype)
                if (splitk_attr is not None and out_dim < _SPLITK_THRESHOLD
                        and in_dim <= _SMEM_MAX_DIM):
                    # Split-K: 1 block per output element, 32 threads,
                    # warp-reduce.  Full SM occupancy even for small out_dim
                    # (e.g. 896 -> 896 blocks vs 7 with 128-thread smem).
                    fn = getattr(self, splitk_attr)
                    self.cu.launch(fn, (out_dim, 1), (32, 1),
                                   [ctypes.c_ulonglong(ptr.value),
                                    ctypes.c_ulonglong(x_ptr.value),
                                    ctypes.c_ulonglong(out_ptr.value),
                                    in_dim, out_dim],
                                   shared_mem=in_dim * 4)
                else:
                    smem_attr = _SMEM_MV.get(gtype)
                    if smem_attr is not None and in_dim <= _SMEM_MAX_DIM:
                        # Shared-memory path: x in smem (loaded once per block),
                        # 128 threads/block for better SM occupancy.
                        fn = getattr(self, smem_attr)
                        self.cu.launch(fn, ((out_dim + 127) // 128, 1), (128, 1),
                                       [ctypes.c_ulonglong(ptr.value),
                                        ctypes.c_ulonglong(x_ptr.value),
                                        ctypes.c_ulonglong(out_ptr.value),
                                        in_dim, out_dim],
                                       shared_mem=in_dim * 4)
                    else:
                        fn = getattr(self, _FUSED_MV[gtype])
                        self.cu.launch(fn, ((out_dim + 255) // 256, 1), (256, 1),
                                       [ctypes.c_ulonglong(ptr.value),
                                        ctypes.c_ulonglong(x_ptr.value),
                                        ctypes.c_ulonglong(out_ptr.value),
                                        in_dim, out_dim])
            else:
                # No fused kernel / bad alignment - fall back to two-pass
                w = self._dequant(ptr, gtype, out_dim * in_dim)
                self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(x_ptr.value),
                                ctypes.c_ulonglong(w.value),
                                ctypes.c_ulonglong(out_ptr.value),
                                in_dim, out_dim])
            return out_ptr, out_dim
        finally:
            if temp:
                self.cu.free(ptr)

    # ── Batched (prefill) GPU operations ─────────────────────────────
    def matmul_batch_gpu(self, x_ptr, name, out_ptr, M, raw=None):
        self._bind()
        """Batched fused dequant+matmul: Y[M,out] = X[M,in] @ W[out,in].T.
        Keeps result on GPU.  Uses the batched fused kernel when the gtype
        is supported and in_dim is block-aligned; otherwise falls back to
        per-token matvec_gpu calls."""
        _ensure_ctypes()
        ptr, gtype, out_dim, in_dim, temp = self._weight_ptr(name, raw)
        try:
            grid = ((out_dim + 255) // 256, M)
            block = (256, 1)
            if gtype == G_F32:
                self.cu.launch(self.f_mm_f32, grid, block,
                               [ctypes.c_ulonglong(ptr.value),
                                ctypes.c_ulonglong(x_ptr.value),
                                ctypes.c_ulonglong(out_ptr.value),
                                M, in_dim, out_dim])
            elif _fused_ok(gtype, in_dim) and gtype in _BATCH_MV:
                fn = getattr(self, _BATCH_MV[gtype])
                self.cu.launch(fn, grid, block,
                               [ctypes.c_ulonglong(ptr.value),
                                ctypes.c_ulonglong(x_ptr.value),
                                ctypes.c_ulonglong(out_ptr.value),
                                M, in_dim, out_dim])
            else:
                # Fall back: M individual matvec_gpu calls
                for m in range(M):
                    xrow = self.cu.alloc(in_dim * 4)
                    self.cu.cu.cuMemcpyDtoD_v2(xrow,
                        ctypes.c_ulonglong(x_ptr.value + m * in_dim * 4), in_dim * 4)
                    yrow = self.cu.alloc(out_dim * 4)
                    self.matvec_gpu(xrow, name, yrow, raw=raw)
                    self.cu.cu.cuMemcpyDtoD_v2(
                        ctypes.c_ulonglong(out_ptr.value + m * out_dim * 4),
                        yrow, out_dim * 4)
                    self.cu.free(xrow); self.cu.free(yrow)
            return out_ptr, out_dim
        finally:
            if temp:
                self.cu.free(ptr)

    def batch_rmsnorm_gpu(self, x_ptr, w_ptr, dim, M, eps=1e-5, out_ptr=None):
        """Batched RMSNorm over M rows.  Grid (M,), block (min(dim,256),)."""
        _ensure_ctypes()
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * M * 4)
        block = min(dim, 256)
        self.cu.launch(self.f_batch_rmsnorm, (M, 1), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(x_ptr.value),
                        ctypes.c_ulonglong(w_ptr.value),
                        dim, eps, M])
        return out_ptr

    def batch_rope_gpu(self, q_ptr, k_ptr, n_heads, n_kv_heads,
                       head_dim, start_pos, M, rope_base):
        """Batched RoPE: apply position (start_pos + m) to each token m."""
        _ensure_ctypes()
        total = n_heads * head_dim // 2
        if n_kv_heads * head_dim // 2 > total:
            total = n_kv_heads * head_dim // 2
        block = min(total, 256)
        self.cu.launch(self.f_batch_rope, (M, 1), (block, 1),
                       [ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_ptr.value),
                        n_heads, n_kv_heads, head_dim, start_pos, M, rope_base])

    def batch_silu_mul_gpu(self, gate_ptr, up_ptr, dim, M, out_ptr=None):
        """Batched SiLU * up over M rows."""
        _ensure_ctypes()
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * M * 4)
        total = dim * M
        self.cu.launch(self.f_batch_silu, ((total + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(gate_ptr.value),
                        ctypes.c_ulonglong(up_ptr.value),
                        dim, M])
        return out_ptr

    def batch_add_gpu(self, a_ptr, b_ptr, dim, M):
        """Batched residual add: a[m] += b[m] over M rows."""
        _ensure_ctypes()
        total = dim * M
        self.cu.launch(self.f_batch_add, ((total + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(a_ptr.value),
                        ctypes.c_ulonglong(b_ptr.value),
                        dim, M])

    def rmsnorm_gpu(self, x_ptr, w_ptr, dim, eps=1e-5, out_ptr=None):
        """RMSNorm on GPU. x_ptr, w_ptr are device ptrs. Returns out_ptr."""
        _ensure_ctypes()
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        block = min(dim, 256)
        self.cu.launch(self.f_rmsnorm, (1, 1), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(x_ptr.value),
                        ctypes.c_ulonglong(w_ptr.value),
                        dim, eps])
        return out_ptr

    def layernorm_gpu(self, x_ptr, w_ptr, dim, eps=1e-5, out_ptr=None):
        """LayerNorm on GPU."""
        _ensure_ctypes()
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        block = min(dim, 256)
        self.cu.launch(self.f_layernorm, (1, 1), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(x_ptr.value),
                        ctypes.c_ulonglong(w_ptr.value),
                        dim, eps])
        return out_ptr

    def rope_gpu(self, q_ptr, k_ptr, n_heads, n_kv_heads, head_dim, pos, rope_base=10000.0):
        """Apply RoPE to Q and K in-place on GPU."""
        _ensure_ctypes()
        total = max(n_heads, n_kv_heads) * (head_dim // 2)
        grid = ((total + 255) // 256, 1)
        self.cu.launch(self.f_rope, grid, (256, 1),
                       [ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_ptr.value),
                        n_heads, n_kv_heads, head_dim, pos, rope_base])

    def attention_gpu(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                      n_heads, n_kv_heads, head_dim, seq_len, kv_stride):
        """Compute attention on GPU. One block per head.
        k_cache/v_cache: (seq_len, kv_stride) device ptrs.
        out_ptr: (n_heads * head_dim,) device ptr.
        Uses flash attention (no 4096 limit) for long contexts."""
        _ensure_ctypes()
        block = min(head_dim, 256)
        # During CUDA graph capture, always use flash attention (no 4096
        # limit) - see attention_f16_gpu for the rationale.
        _capturing = getattr(self.cu, '_capturing', False)
        _seq_val = seq_len.value if hasattr(seq_len, 'value') else seq_len
        if _seq_val > 4096 or _capturing:
            self.cu.launch(self.f_flash_attn, (n_heads, 1), (block, 1),
                           [ctypes.c_ulonglong(out_ptr.value),
                            ctypes.c_ulonglong(q_ptr.value),
                            ctypes.c_ulonglong(k_cache_ptr.value),
                            ctypes.c_ulonglong(v_cache_ptr.value),
                            n_heads, n_kv_heads, head_dim, seq_len, kv_stride])
        else:
            self.cu.launch(self.f_attn, (n_heads, 1), (block, 1),
                           [ctypes.c_ulonglong(out_ptr.value),
                            ctypes.c_ulonglong(q_ptr.value),
                            ctypes.c_ulonglong(k_cache_ptr.value),
                            ctypes.c_ulonglong(v_cache_ptr.value),
                            n_heads, n_kv_heads, head_dim, seq_len, kv_stride])
        return out_ptr

    def silu_mul_gpu(self, gate_ptr, up_ptr, dim, out_ptr=None):
        """SiLU(gate) * up on GPU. Returns out_ptr."""
        _ensure_ctypes()
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        self.cu.launch(self.f_silu, ((dim + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(gate_ptr.value),
                        ctypes.c_ulonglong(up_ptr.value), dim])
        return out_ptr

    def add_gpu(self, a_ptr, b_ptr, dim):
        """a += b on GPU (in-place)."""
        _ensure_ctypes()
        self.cu.launch(self.f_add, ((dim + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(a_ptr.value),
                        ctypes.c_ulonglong(b_ptr.value), dim])
        return a_ptr

    def add_scaled_gpu(self, a_ptr, b_ptr, s, dim):
        """a += s * b on GPU (in-place).  s is a python float."""
        _ensure_ctypes()
        self.cu.launch(self.f_add_scaled, ((dim + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(a_ptr.value),
                        ctypes.c_ulonglong(b_ptr.value), float(s), dim])
        return a_ptr

    def geglu_mul_gpu(self, gate_ptr, up_ptr, dim, out_ptr=None):
        """GeGLU-tanh(gate) * up on GPU (gemma4 experts). Returns out_ptr."""
        _ensure_ctypes()
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        self.cu.launch(self.f_geglu, ((dim + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(gate_ptr.value),
                        ctypes.c_ulonglong(up_ptr.value), dim])
        return out_ptr

    def scale_const_gpu(self, x_ptr, s, dim):
        """x *= s in place on GPU. Returns x_ptr."""
        _ensure_ctypes()
        self.cu.launch(self.f_scale_const, ((dim + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(x_ptr.value), float(s), dim])
        return x_ptr

    def store_f16_gpu(self, f32_ptr, f16_ptr, n_elems):
        """Convert F32 array to F16 on GPU (for F16 KV cache).
        Passes c_ulonglong args directly (not copies) so that persistent
        graph-mode args (e.g. _g_k_dst) are updated in place by replay."""
        _ensure_ctypes()
        a = f32_ptr if isinstance(f32_ptr, ctypes.c_ulonglong) else ctypes.c_ulonglong(f32_ptr)
        b = f16_ptr if isinstance(f16_ptr, ctypes.c_ulonglong) else ctypes.c_ulonglong(f16_ptr)
        self.cu.launch(self.f_f32_to_f16, ((n_elems + 255) // 256, 1), (256, 1),
                       [a, b, n_elems])
        return f16_ptr

    def attention_f16_gpu(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                          n_heads, n_kv_heads, head_dim, seq_len, kv_stride):
        """Attention with F16 KV cache - half the VRAM for cache.
        Uses the flash-attention kernel (online softmax, no 4096 context
        limit) when seq_len exceeds the shared-memory limit of the
        classic kernel; otherwise uses the classic kernel which is
        slightly faster for short contexts (fewer tiles)."""
        _ensure_ctypes()
        block = min(head_dim, 256)
        # During CUDA graph capture, always use flash attention (no 4096
        # limit) - the graph is captured at position 0 (seq_len=1) and
        # replayed as the conversation grows.  The classic kernel would
        # read out of bounds past 4096 tokens.
        _capturing = getattr(self.cu, '_capturing', False)
        _seq_val = seq_len.value if hasattr(seq_len, 'value') else seq_len
        if _seq_val > 4096 or _capturing:
            # Flash attention: tiled online softmax, no context limit.
            self.cu.launch(self.f_flash_attn_f16, (n_heads, 1), (block, 1),
                           [ctypes.c_ulonglong(out_ptr.value),
                            ctypes.c_ulonglong(q_ptr.value),
                            ctypes.c_ulonglong(k_cache_ptr.value),
                            ctypes.c_ulonglong(v_cache_ptr.value),
                            n_heads, n_kv_heads, head_dim, seq_len, kv_stride])
        else:
            self.cu.launch(self.f_attn_f16, (n_heads, 1), (block, 1),
                           [ctypes.c_ulonglong(out_ptr.value),
                            ctypes.c_ulonglong(q_ptr.value),
                            ctypes.c_ulonglong(k_cache_ptr.value),
                            ctypes.c_ulonglong(v_cache_ptr.value),
                            n_heads, n_kv_heads, head_dim, seq_len, kv_stride])
        return out_ptr

    def flash_attention_gpu(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                             n_heads, n_kv_heads, head_dim, seq_len, kv_stride):
        """Flash-style attention - tiled, no 4096 context limit."""
        _ensure_ctypes()
        block = min(head_dim, 256)
        self.cu.launch(self.f_flash_attn, (n_heads, 1), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_cache_ptr.value),
                        ctypes.c_ulonglong(v_cache_ptr.value),
                        n_heads, n_kv_heads, head_dim, seq_len, kv_stride])
        return out_ptr

    def flash_attention_f16_gpu(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                                n_heads, n_kv_heads, head_dim, seq_len, kv_stride):
        """Flash-style attention with F16 KV cache - tiled, no 4096 limit."""
        _ensure_ctypes()
        block = min(head_dim, 256)
        self.cu.launch(self.f_flash_attn_f16, (n_heads, 1), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_cache_ptr.value),
                        ctypes.c_ulonglong(v_cache_ptr.value),
                        n_heads, n_kv_heads, head_dim, seq_len, kv_stride])
        return out_ptr

    def batch_attention_gpu(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                            n_heads, n_kv_heads, head_dim, batch, kv_stride):
        """Batched causal attention (F32 KV cache). Grid: (n_heads, batch)."""
        _ensure_ctypes()
        block = min(head_dim, 256)
        self.cu.launch(self.f_batch_attn, (n_heads, batch), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_cache_ptr.value),
                        ctypes.c_ulonglong(v_cache_ptr.value),
                        n_heads, n_kv_heads, head_dim, batch, kv_stride])
        return out_ptr

    def batch_attention_f16_gpu(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                                n_heads, n_kv_heads, head_dim, batch, kv_stride):
        """Batched causal attention (F16 KV cache). Grid: (n_heads, batch)."""
        _ensure_ctypes()
        block = min(head_dim, 256)
        self.cu.launch(self.f_batch_attn_f16, (n_heads, batch), (block, 1),
                       [ctypes.c_ulonglong(out_ptr.value),
                        ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_cache_ptr.value),
                        ctypes.c_ulonglong(v_cache_ptr.value),
                        n_heads, n_kv_heads, head_dim, batch, kv_stride])
        return out_ptr

    def batch_matvec_gpu(self, x_ptr, name, out_ptr, batch, in_dim, out_dim):
        self._bind()
        """Batched matvec for prefill: (batch, in_dim) x (out_dim, in_dim) -> (batch, out_dim).
        Keeps result on GPU. x_ptr and out_ptr are device ptrs."""
        _ensure_ctypes()
        ptr, gtype, od, id_, temp = self._weight_ptr(name, None)
        try:
            if gtype == G_F32:
                w = ptr
            else:
                w = self._dequant(ptr, gtype, out_dim * in_dim)
            self.cu.launch(self.f_batch_matvec,
                           ((out_dim + 255) // 256, 1), (256, 1),
                           [ctypes.c_ulonglong(x_ptr.value),
                            ctypes.c_ulonglong(w.value),
                            ctypes.c_ulonglong(out_ptr.value),
                            batch, in_dim, out_dim])
            return out_ptr, out_dim
        finally:
            if temp:
                self.cu.free(ptr)

    def fused_qkv_gpu(self, x_ptr, wq_name, wk_name, wv_name,
                      q_ptr, k_ptr, v_ptr, in_dim, q_dim, k_dim, v_dim):
        self._bind()
        """Fused QKV matvec - one kernel launch instead of three."""
        _ensure_ctypes()
        # Get weight pointers (must be F32 or dequantized first)
        pq, gq, _, _, tq = self._weight_ptr(wq_name, None)
        pk, gk, _, _, tk = self._weight_ptr(wk_name, None)
        pv, gv, _, _, tv = self._weight_ptr(wv_name, None)
        try:
            wq = pq if gq == G_F32 else self._dequant(pq, gq, q_dim * in_dim)
            wk = pk if gk == G_F32 else self._dequant(pk, gk, k_dim * in_dim)
            wv = pv if gv == G_F32 else self._dequant(pv, gv, v_dim * in_dim)
            total = q_dim + k_dim + v_dim
            self.cu.launch(self.f_fused_qkv,
                           ((total + 255) // 256, 1), (256, 1),
                           [ctypes.c_ulonglong(x_ptr.value),
                            ctypes.c_ulonglong(wq.value),
                            ctypes.c_ulonglong(wk.value),
                            ctypes.c_ulonglong(wv.value),
                            ctypes.c_ulonglong(q_ptr.value),
                            ctypes.c_ulonglong(k_ptr.value),
                            ctypes.c_ulonglong(v_ptr.value),
                            in_dim, q_dim, k_dim, v_dim])
            return q_ptr, k_ptr, v_ptr
        finally:
            if tq: self.cu.free(pq)
            if tk: self.cu.free(pk)
            if tv: self.cu.free(pv)

    def upload_f32(self, name, arr, out_dim, in_dim):
        """Upload an f32 weight (norms, biases) to GPU as G_F32."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        raw = arr.tobytes()
        self.upload_raw(name, raw, G_F32, out_dim, in_dim)

    def to_gpu_ptr(self, arr):
        self._bind()
        """Upload a numpy array to GPU, return device ptr."""
        _ensure_ctypes()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ptr = self.cu.alloc(arr.nbytes)
        r = self.cu.cu.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes)
        if r != 0:
            raise RuntimeError("HtoD %d" % r)
        return ptr

    def from_gpu_ptr(self, ptr, n_elems):
        self._bind()
        """Download n_elems float32 from GPU to numpy."""
        _ensure_ctypes()
        return self.cu.from_gpu(ptr, (n_elems,))

    # ── GPU-resident .ntmre compression matvec (no CPU download) ───────────────────
    def matvec_svd_gpu(self, x_ptr, name_a, name_b, out_ptr=None, s_name=None,
                       raw_a=None, raw_b=None):
        self._bind()
        """.ntmre compression matvec keeping result on GPU. x_ptr is a device ptr.
        Returns (out_ptr, out_dim)."""
        pb, gb, rank, in_dim, tb = self._weight_ptr(name_b, raw_b)
        pa, ga, out_dim, rank_a, ta = self._weight_ptr(name_a, raw_a)
        assert rank == rank_a
        try:
            # h = x @ Q.T  (rank,)
            wq = pb if gb == G_F32 else self._dequant(pb, gb, rank * in_dim)
            hp = self._matvec_dev(x_ptr, wq, in_dim, rank)
            # h *= S
            if s_name is not None:
                sp, _sg, _n, _m = self._raw[s_name]
                self.cu.launch(self.f_scale, ((rank + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(hp.value),
                                ctypes.c_ulonglong(sp.value), rank])
            # y = h @ P.T  (out,)
            wp = pa if ga == G_F32 else self._dequant(pa, ga, out_dim * rank)
            if out_ptr is None:
                out_ptr = self.cu.alloc(out_dim * 4)
            self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                           [ctypes.c_ulonglong(hp.value),
                            ctypes.c_ulonglong(wp.value),
                            ctypes.c_ulonglong(out_ptr.value),
                            rank, out_dim])
            self.cu.free(hp)
            return out_ptr, out_dim
        finally:
            if ta:
                self.cu.free(pa)
            if tb:
                self.cu.free(pb)

    def matvec_svd_reinterpret_gpu(self, x_ptr, name, out_dim, in_dim,
                                   out_ptr=None, blob=None):
        self._bind()
        """NSVD .ntmre compression matvec keeping result on GPU. x_ptr is a device ptr.
        Returns (out_ptr, out_dim)."""
        tag, m, n, rank, apply_s = self._raw[name]
        temps = []
        sptr = None
        if tag == 'svd_stream':
            if blob is None:
                raise ValueError(f"{name}: streamed .ntmre compression needs the blob")
            _rank, u_len, s_len, v_off, apply_s = self._svd_blob_layout(blob, m, n)
            parts = [blob[4:4 + u_len]]
            if apply_s:
                parts.append(blob[4 + u_len:4 + u_len + s_len])
            parts.append(blob[v_off:])
            for raw_part in parts:
                arr = np.ascontiguousarray(np.frombuffer(raw_part, dtype=np.uint8))
                p = self.cu.alloc(len(raw_part))
                r = self.cu.cu.cuMemcpyHtoD_v2(p, ctypes.c_void_p(arr.ctypes.data), len(raw_part))
                if r != 0:
                    raise RuntimeError("HtoD svd %d" % r)
                temps.append(p)
            if apply_s:
                uptr, sptr, vptr = temps
            else:
                uptr, vptr = temps
        else:
            assert tag == 'svd'
            uptr, _ut, _m, _r = self._raw[name + '.__U']
            if apply_s:
                sptr, _st, _1, _r2 = self._raw[name + '.__S']
            vptr, _vt, _r3, _n = self._raw[name + '.__V']
        try:
            # U' = dequant(U) [* S (columns)]
            self._ensure_scratch2(m * rank)
            grid = ((m * rank + 255) // 256, 1)
            self.cu.launch(self.f_q8_34, grid, (256, 1),
                           [ctypes.c_ulonglong(uptr.value),
                            ctypes.c_ulonglong(self._scratch2.value), m * rank])
            if apply_s:
                self.cu.launch(self.f_scols, grid, (256, 1),
                               [ctypes.c_ulonglong(self._scratch2.value),
                                ctypes.c_ulonglong(sptr.value), m * rank, rank])
            # V = dequant(Vt)
            self._ensure_scratch(rank * n)
            grid = ((rank * n + 255) // 256, 1)
            self.cu.launch(self.f_q8_34, grid, (256, 1),
                           [ctypes.c_ulonglong(vptr.value),
                            ctypes.c_ulonglong(self._scratch.value), rank * n])
            # N_c = U' @ V  (m, rank) @ (rank, n) -> (m, n)
            self._ensure_scratch3(m * n)
            self.cu.launch(self.cu.f_matmul,
                           ((n + 15) // 16, (m + 15) // 16), (16, 16),
                           [ctypes.c_ulonglong(self._scratch2.value),
                            ctypes.c_ulonglong(self._scratch.value),
                            ctypes.c_ulonglong(self._scratch3.value),
                            m, rank, n])
            # y = x @ N_c.T  (no upload - x_ptr is already on GPU)
            if out_ptr is None:
                out_ptr = self.cu.alloc(out_dim * 4)
            self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                           [ctypes.c_ulonglong(x_ptr.value),
                            ctypes.c_ulonglong(self._scratch3.value),
                            ctypes.c_ulonglong(out_ptr.value),
                            in_dim, out_dim])
            return out_ptr, out_dim
        finally:
            for p in temps:
                self.cu.free(p)

    def matvec_svd(self, x, name_a, name_b, s_name=None, raw_a=None, raw_b=None):
        self._bind()
        """.ntmre compression weight W = P @ Q without materializing W:
        h = x @ Q.T  (rank,), optional h *= S,  y = h @ P.T  (out,).
        name_a -> P (out, rank), name_b -> Q (rank, in)."""
        pb, gb, rank, in_dim, tb = self._weight_ptr(name_b, raw_b)
        pa, ga, out_dim, rank_a, ta = self._weight_ptr(name_a, raw_a)
        assert rank == rank_a
        try:
            x = np.ascontiguousarray(x, dtype=np.float32).ravel()
            self._ensure_x(max(in_dim, rank))
            r = self.cu.cu.cuMemcpyHtoD_v2(self._x, ctypes.c_void_p(x.ctypes.data), x.nbytes)
            if r != 0:
                raise RuntimeError("HtoD x %d" % r)
            # h = x @ Q.T
            wq = pb if gb == G_F32 else self._dequant(pb, gb, rank * in_dim)
            hp = self._matvec_dev(self._x, wq, in_dim, rank)
            # h *= S
            if s_name is not None:
                sp, _sg, _n, _m = self._raw[s_name]
                self.cu.launch(self.f_scale, ((rank + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(hp.value), ctypes.c_ulonglong(sp.value), rank])
            # y = h @ P.T
            wp = pa if ga == G_F32 else self._dequant(pa, ga, out_dim * rank)
            yp = self._matvec_dev(hp, wp, rank, out_dim)
            y = self.cu.from_gpu(yp, (out_dim,))
            self.cu.free(hp)
            self.cu.free(yp)
            return y
        finally:
            if ta:
                self.cu.free(pa)
            if tb:
                self.cu.free(pb)

    @staticmethod
    def _svd_blob_layout(blob, m, n):
        """Parse an NSVD .ntmre compression blob header: (rank, u_len, s_len, v_off,
        apply_s).  Handles both dialects:
          with-S:  rank + Q8(U) + f32 S + Q8(Vt)   (S applied unless pre-folded)
          no-S:    rank + Q8(A) + Q8(B)            (S already folded by writer)
        Pre-folded with-S blobs (U column norms ~ S) skip the S application."""
        import struct as _st
        from engine.bonsai_pc_app import _dequant_q8
        rank = _st.unpack_from('<I', blob, 0)[0]
        if rank == 0:
            raise ValueError('rank-0 .ntmre compression blob')
        u_len = 34 * ((m * rank + 31) // 32)
        v_len = 34 * ((rank * n + 31) // 32)
        if len(blob) >= 4 + u_len + 4 * rank + v_len:
            S = np.frombuffer(blob[4 + u_len:4 + u_len + 4 * rank], dtype=np.float32)
            if not np.all(np.isfinite(S)) or (S.size and float(np.median(np.abs(S))) > 1e4):
                raise ValueError(
                    '.ntmre compression tensor is corrupt (singular values non-finite or ~1e6) '
                    '- this file came from a broken conversion; re-convert from GGUF')
            U = _dequant_q8(blob[4:4 + u_len], m * rank).reshape(m, rank)
            cn_med = float(np.median(np.linalg.norm(U, axis=0)))
            s_med = float(np.median(np.abs(S))) if S.size else 1.0
            apply_s = not (cn_med > 3.0 and cn_med / max(s_med, 1e-30) < 3.0)
            return rank, u_len, 4 * rank, 4 + u_len + 4 * rank, apply_s
        if len(blob) >= 4 + u_len + v_len:
            return rank, u_len, 0, 4 + u_len, False
        raise ValueError('.ntmre compression blob too short for its shape')

    def upload_svd_nsvd(self, name, blob, m, n):
        self._bind()
        """NSVD .ntmre compression blob (either dialect).  Uploads the factor buffers
        (or marks the whole tensor for streaming when the resident budget
        is gone)."""
        if name in self._raw:
            return
        rank, u_len, s_len, v_off, apply_s = self._svd_blob_layout(blob, m, n)
        if len(blob) > self._budget:
            self._raw[name] = ('svd_stream', m, n, rank, apply_s)
            return
        self.upload_raw(name + '.__U', blob[4:4 + u_len], G_Q8_34, m, rank)
        if apply_s:
            self.upload_raw(name + '.__S', blob[4 + u_len:4 + u_len + s_len], G_F32, 1, rank)
        self.upload_raw(name + '.__V', blob[v_off:], G_Q8_34, rank, n)
        self._raw[name] = ('svd', m, n, rank, apply_s)

    def matvec_svd_reinterpret(self, x, name, out_dim, in_dim, blob=None):
        self._bind()
        """NSVD .ntmre compression: W_pt = (P@Q).ravel().reshape((out, in)) - materialize
        N_c = (U*S) @ Vt on GPU (or U @ Vt for pre-folded/no-S dialects),
        then matvec reading N_c as (out, in)."""
        tag, m, n, rank, apply_s = self._raw[name]
        temps = []
        sptr = None
        if tag == 'svd_stream':
            if blob is None:
                raise ValueError(f"{name}: streamed .ntmre compression needs the blob")
            _rank, u_len, s_len, v_off, apply_s = self._svd_blob_layout(blob, m, n)
            parts = [blob[4:4 + u_len]]
            if apply_s:
                parts.append(blob[4 + u_len:4 + u_len + s_len])
            parts.append(blob[v_off:])
            for raw_part in parts:
                arr = np.ascontiguousarray(np.frombuffer(raw_part, dtype=np.uint8))
                p = self.cu.alloc(len(raw_part))
                r = self.cu.cu.cuMemcpyHtoD_v2(p, ctypes.c_void_p(arr.ctypes.data), len(raw_part))
                if r != 0:
                    raise RuntimeError("HtoD svd %d" % r)
                temps.append(p)
            if apply_s:
                uptr, sptr, vptr = temps
            else:
                uptr, vptr = temps
        else:
            assert tag == 'svd'
            uptr, _ut, _m, _r = self._raw[name + '.__U']
            if apply_s:
                sptr, _st, _1, _r2 = self._raw[name + '.__S']
            vptr, _vt, _r3, _n = self._raw[name + '.__V']
        try:
            # U' = dequant(U) [* S (columns) for the raw-U dialect]
            self._ensure_scratch2(m * rank)
            grid = ((m * rank + 255) // 256, 1)
            self.cu.launch(self.f_q8_34, grid, (256, 1),
                           [ctypes.c_ulonglong(uptr.value), ctypes.c_ulonglong(self._scratch2.value), m * rank])
            if apply_s:
                self.cu.launch(self.f_scols, grid, (256, 1),
                               [ctypes.c_ulonglong(self._scratch2.value), ctypes.c_ulonglong(sptr.value), m * rank, rank])
            # V = dequant(Vt)
            self._ensure_scratch(rank * n)
            grid = ((rank * n + 255) // 256, 1)
            self.cu.launch(self.f_q8_34, grid, (256, 1),
                           [ctypes.c_ulonglong(vptr.value), ctypes.c_ulonglong(self._scratch.value), rank * n])
            # N_c = U' @ V   (m, rank) @ (rank, n) -> (m, n)
            self._ensure_scratch3(m * n)
            self.cu.launch(self.cu.f_matmul,
                           ((n + 15) // 16, (m + 15) // 16), (16, 16),
                           [ctypes.c_ulonglong(self._scratch2.value), ctypes.c_ulonglong(self._scratch.value),
                            ctypes.c_ulonglong(self._scratch3.value), m, rank, n])
            # y[o] = sum_i x[i] * N_c.ravel()[o*in + i]
            x = np.ascontiguousarray(x, dtype=np.float32).ravel()
            self._ensure_x(in_dim)
            r = self.cu.cu.cuMemcpyHtoD_v2(self._x, ctypes.c_void_p(x.ctypes.data), x.nbytes)
            if r != 0:
                raise RuntimeError("HtoD x %d" % r)
            yp = self._matvec_dev(self._x, self._scratch3, in_dim, out_dim)
            y = self.cu.from_gpu(yp, (out_dim,))
            self.cu.free(yp)
            return y
        finally:
            for p in temps:
                self.cu.free(p)

    def matvec_chunked(self, x, name, chunk_rows=4096, raw=None):
        self._bind()
        """Logits over a huge (out, in) weight: dequant row-chunks into a
        modest scratch, matvec each, assemble the full output vector.
        Streamed weights are uploaded chunk-by-chunk from the host bytes
        so the full tensor never needs a device allocation.

        DO NOT "FIX" THE OUT-OF-MEMORY FALLBACK IN HERE.
        ------------------------------------------------------------------
        This was changed to shrink the chunk and retry on OOM so the head
        would stay on the GPU. Measured, that made the model SLOWER:

            before (CPU fallback)      540 ms/token    1.85 tok/s
            after  (GPU, shrinking)   1126 ms/token    0.89 tok/s

    The reason is that with both cards full the head is not resident, it
    is STREAMED: every chunk copies its raw bytes host-to-device. For
    Gemma-4-26B's 262144x2816 head that is the whole weight crossing PCIe
    once per token.

        head in isolation, one call:
            CPU f32 dot              91.2 ms
            GPU chunk 4096          793.1 ms
            GPU chunk  512          520.1 ms
            GPU chunk  256          887.3 ms

    The GPU wins on compute. It loses badly on the transfer, and the
    transfer is the whole cost here. A 2.75 GB f32 matrix read once per
    token from system RAM at memory bandwidth beats pushing ~430 MB of
    quantised bytes across PCIe, even with a 3060 doing the arithmetic.

    So the OOM here is not a bug that needs smoothing over -- it is the
    engine correctly refusing a path that is 6x slower. The right fix is
    more VRAM, or a resident head, not a cleverer retry.
        """
        MIN_CHUNK = 64
        ptr, gtype, out_dim, in_dim = self._raw[name]
        streamed = (ptr == 'stream')
        if streamed and raw is None:
            raise ValueError(f"{name}: streamed weight needs raw bytes")
        # bytes per row in the raw stream (rows are block-aligned)
        if gtype == G_Q1_0:
            bpr = (in_dim // 128) * 18
        elif gtype == G_Q8_34:
            bpr = (in_dim // 32) * 34
        elif gtype == G_Q8_36:
            bpr = (in_dim // 32) * 36
        elif gtype == G_F32:
            bpr = in_dim * 4
        elif gtype == G_F16:
            bpr = in_dim * 2
        elif gtype in _GGML_BLOCKS:
            eb, bb = _GGML_BLOCKS[gtype]
            bpr = (in_dim // eb) * bb
        else:
            raise ValueError(gtype)
        x = np.ascontiguousarray(x, dtype=np.float32).ravel()
        self._ensure_x(in_dim)
        r = self.cu.cu.cuMemcpyHtoD_v2(self._x, ctypes.c_void_p(x.ctypes.data), x.nbytes)
        if r != 0:
            raise RuntimeError("HtoD x %d" % r)
        yp = self.cu.alloc(out_dim * 4)
        try:
            done = 0
            while done < out_dim:
                rows = min(chunk_rows, out_dim - done)
                n_elems = rows * in_dim
                stage = None
                if streamed:
                    chunk = raw[done * bpr:(done + rows) * bpr]
                    arr = np.ascontiguousarray(np.frombuffer(chunk, dtype=np.uint8))
                    stage = self.cu.alloc(len(chunk))
                    r = self.cu.cu.cuMemcpyHtoD_v2(stage, ctypes.c_void_p(arr.ctypes.data), len(chunk))
                    if r != 0:
                        self.cu.free(stage)
                        raise RuntimeError("HtoD chunk %d" % r)
                    base = stage.value
                else:
                    base = ptr.value + done * bpr
                try:
                    if gtype == G_F32:
                        wptr = ctypes.c_ulonglong(base)
                    else:
                        self._ensure_scratch(n_elems)
                        grid = ((n_elems + 255) // 256, 1)
                        fn = {G_Q1_0: self.f_q1_0, G_Q8_34: self.f_q8_34, G_Q8_36: self.f_q8_36,
                              G_F16: self.f_f16, G_Q4_0: self.f_q4_0, G_Q5_0: self.f_q5_0,
                              G_Q6_0: self.f_q6_0, G_Q4_K: self.f_q4_k, G_Q5_K: self.f_q5_k,
                              G_Q6_K: self.f_q6_k}[gtype]
                        self.cu.launch(fn, grid, (256, 1),
                                       [ctypes.c_ulonglong(base),
                                        ctypes.c_ulonglong(self._scratch.value), n_elems])
                        wptr = self._scratch
                    self._matvec_dev(self._x, wptr, in_dim, rows,
                                     out=ctypes.c_ulonglong(yp.value + done * 4))
                finally:
                    if stage is not None:
                        self.cu.free(stage)
                done += rows
            y = self.cu.from_gpu(yp, (out_dim,))
            self.cu.free(yp)
            return y
        except Exception:
            self.cu.free(yp)
            raise

    def close(self):
        """Free all GPU memory and destroy the CUDA context."""
        try:
            for ptr, *_ in self._raw.values():
                if hasattr(ptr, 'value'):
                    self.cu.free(ptr)
            for sp in ('_scratch', '_scratch2', '_scratch3', '_x'):
                p = getattr(self, sp, None)
                if p is not None:
                    self.cu.free(p)
            # Destroy the context itself so VRAM is fully released
            self.cu.close()
        except Exception:
            pass
        self._raw = {}
        if getattr(self, 'cu', None) is not None:
            try:
                self.cu.close()
            except Exception:
                pass
            self.cu = None


# Workaround: some sealed exe runtimes fail to keep ctypes in module globals.
# Re-import it at module load if missing.
if 'ctypes' not in globals():
    import ctypes
