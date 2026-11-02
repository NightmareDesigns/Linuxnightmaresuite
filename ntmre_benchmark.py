# -*- coding: utf-8 -*-
"""
ntmre_benchmark.py — Speed, latency AND CORRECTNESS benchmark
===========================================================================
Measures:
  - Tokens Per Second (tok/s)
  - Time To First Token (TTFT in ms)
  - Memory Bandwidth & GPU VRAM utilization
  - The actual generated TEXT
  - Whether that text is trustworthy at all

Why the last two matter
-----------------------
Speed alone cannot tell a working model from a broken one. This suite
reported a confident "13.12 tokens/sec" while `forward()` was returning a
constant all-zero logit vector, so every token was `argmax(zeros) == 0`.
Nothing printed the output, so nothing looked wrong.

Every scorecard here therefore carries the generated text and a verdict
from `ntmre_router.probe_logits`. A degenerate run prints
`!! INVALID RUN !!` and returns a non-zero exit code.

Usage:
  python ntmre_benchmark.py <model> [--tokens 100]
  python ntmre_benchmark.py <model> --strict          # fail on any doubt
  python ntmre_benchmark.py --ab <source.gguf> <converted.ntmre>
"""

import os
import sys
import time
import argparse
import ctypes
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

import ntmre_router
from tokenizer import Tokenizer


def get_gpu_info():
    """Query NVIDIA driver for active GPU names and total VRAM."""
    gpus = []
    try:
        cu = ctypes.CDLL("nvcuda.dll")
        if cu.cuInit(0) == 0:
            count = ctypes.c_int(0)
            cu.cuDeviceGetCount(ctypes.byref(count))
            for i in range(count.value):
                dev = ctypes.c_int(0)
                cu.cuDeviceGet(ctypes.byref(dev), i)
                name_buf = ctypes.create_string_buffer(128)
                cu.cuDeviceGetName(name_buf, 128, dev)
                total_mem = ctypes.c_size_t(0)
                cu.cuDeviceTotalMem_v2(ctypes.byref(total_mem), dev)
                gpus.append({
                    "id": i,
                    "name": name_buf.value.decode('utf-8', errors='ignore').strip(),
                    "vram_gb": total_mem.value / (1024 ** 3)
                })
    except Exception:
        pass
    return gpus


def resolve_tokenizer(engine, model_path):
    """Build a Tokenizer that actually has a vocabulary.

    `Tokenizer(engine.metadata, ...)` is not reliable on its own: the
    router does not always populate `metadata` with the GGUF tokenizer
    keys, and an empty dict yields 0 tokens, `model_type "llama"`, and an
    encoding of `[bos, 0, 0, 0, ...]`. Benchmarking that measures a model
    being fed nothing but token 0.

    Falls back to reading the vocab straight out of the GGUF.
    """
    meta = dict(getattr(engine, 'metadata', {}) or {})
    tok = Tokenizer(meta, model_path=model_path)
    if tok.tokens:
        return tok
    if str(model_path).lower().endswith('.gguf'):
        try:
            from nightmare_qwen35 import GGUFSource
            g = GGUFSource(model_path)
            tok = Tokenizer(dict(g._m.metadata or {}), model_path=model_path)
            g.close()
        except Exception:
            pass
    if not tok.tokens:
        raise RuntimeError(
            "could not load a vocabulary for this model — the benchmark "
            "would measure a model fed only token 0")
    return tok


def generate(engine, tokenizer, prompt, n_tokens, progress_cb=None):
    """Prefill + greedy-decode. Returns ids, text and first-token stats."""
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [getattr(tokenizer, 'bos_id', 1) or 1]
    n_prompt = len(ids)

    if hasattr(engine, 'reset'):
        engine.reset()
    if hasattr(engine, 'prefill_batch') and n_prompt > 1:
        engine.prefill_batch(ids[:-1], 0)
    else:
        for i, t in enumerate(ids[:-1]):
            engine.forward(t, i)

    t_start = time.perf_counter()
    first_logits = np.asarray(
        engine.forward(ids[-1], n_prompt - 1)).ravel()
    ttft_ms = (time.perf_counter() - t_start) * 1000.0

    out = [int(np.argmax(first_logits))]
    t_gen = time.perf_counter()
    for step in range(1, n_tokens):
        logits = np.asarray(
            engine.forward(out[-1], n_prompt - 1 + step)).ravel()
        out.append(int(np.argmax(logits)))
        if progress_cb:
            progress_cb(step, f"Generating: {step}/{n_tokens} tokens...")
    gen_time = time.perf_counter() - t_gen

    finite = np.isfinite(first_logits)
    stats = {}
    if finite.any():
        good = first_logits[finite]
        p = np.exp(good - good.max())
        p /= p.sum()
        stats = {
            "min": float(good.min()), "max": float(good.max()),
            "std": float(good.std()), "range": float(good.max() - good.min()),
            "top1_p": float(p.max()),
            "entropy": float(-(p * np.log(p + 1e-30)).sum()),
            "n_nan": int((~finite).sum()),
        }

    eos = getattr(tokenizer, 'eos_id', None)
    try:
        text = tokenizer.decode(out)
    except Exception:
        text = ''.join(getattr(tokenizer, 'decode_single', lambda t: '')(t)
                       for t in out)
    return {
        "prompt_ids": ids,
        "ids": out,
        "text": text,
        "ttft_ms": ttft_ms,
        "gen_time_s": gen_time,
        "tokens_per_sec": (len(out) - 1) / max(gen_time, 1e-9),
        "ms_per_token": (gen_time / max(len(out) - 1, 1)) * 1000.0,
        "stats": stats,
        "eos_id": eos,
    }


def judge(gen, strict=False):
    """Decide whether a generation is trustworthy.

    Returns (verdict, reason). verdict is 'OK', 'SUSPECT' or 'INVALID'.
    Anything other than OK must not be presented as a speed result.
    """
    st = gen.get("stats") or {}
    if not st:
        return "INVALID", "no usable logits were produced"

    if st.get("n_nan"):
        return "INVALID", "%d of the logits are NaN/inf" % st["n_nan"]
    if st["range"] == 0.0 and st["std"] == 0.0:
        return "INVALID", ("all logits are identical — the forward pass "
                           "produced no signal at all")
    # A healthy LM is peaked. Threshold is deliberately loose: small models
    # really can be flat on some prompts, and a false alarm here would make
    # people ignore the warning entirely.
    if st["entropy"] > 9.0 or st["top1_p"] < 1e-4:
        return "INVALID", ("logits are near-uniform (entropy %.2f, top1p "
                           "%.2e) — this is not a language-model "
                           "distribution" % (st["entropy"], st["top1_p"]))
    if st["entropy"] > 7.0:
        return "SUSPECT", ("low-confidence distribution (entropy %.2f, "
                           "top1p %.4f)" % (st["entropy"], st["top1_p"]))
    # Degenerate repetition is the signature of a broken KV cache or a
    # weight fed at the wrong stride, even when the logits look plausible.
    ids = gen.get("ids") or []
    if len(ids) >= 8:
        uniq = len(set(ids))
        if uniq <= 2:
            return "SUSPECT", ("output is only %d distinct token(s) across "
                               "%d — looks degenerate" % (uniq, len(ids)))
    if strict and not _looks_like_prose(gen.get("text", "")):
        return "SUSPECT", "generated text is not printable — check the tokenizer"
    return "OK", ""


def _looks_like_prose(text):
    """Very rough sanity check that the decoded text is real language."""
    if not text:
        return False
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    return printable / max(len(text), 1) > 0.9


def run_benchmark(model_path: str, n_tokens: int = 100,
                  prompt: str = "Explain quantum computing in simple terms:",
                  progress_cb=None, strict: bool = False,
                  status=lambda s: print(f"  {s}")):
    """Run speed + latency + correctness benchmark on model_path."""
    model_path = str(Path(model_path).resolve())
    fname = Path(model_path).name
    fsize_gb = os.path.getsize(model_path) / (1024 ** 3)

    if progress_cb:
        progress_cb(0, f"Loading model {fname} ({fsize_gb:.2f} GB)...")

    engine = ntmre_router.create_engine(model_path, force_cpu=False,
                                        status=status)
    tokenizer = resolve_tokenizer(engine, model_path)

    if progress_cb:
        progress_cb(10, "Generating...")

    gen = generate(engine, tokenizer, prompt, n_tokens, progress_cb)
    verdict, reason = judge(gen, strict=strict)

    # The router's own post-route probe is authoritative about hardware.
    if getattr(engine, '_nightmare_degenerate_output', False) and verdict == "OK":
        verdict, reason = "INVALID", "router's output probe rejected this engine"
    cpu_fallback = bool(getattr(engine, '_nightmare_cpu_fallback', False))

    engine_name = type(engine).__name__
    if hasattr(engine, 'close'):
        try:
            engine.close()
        except Exception:
            pass

    gpus = get_gpu_info()
    gpu_desc = "CPU Inference" if not gpus else " | ".join(
        f"GPU {g['id']}: {g['name']} ({g['vram_gb']:.1f} GB)" for g in gpus)

    st = gen["stats"]
    rule = "=" * 60
    text = gen["text"]
    lines = [
        rule,
        "  NIGHTMARE SUITE (NTMRE) - SPEED + CORRECTNESS SCORECARD",
        rule,
        f"Model:             {fname} ({fsize_gb:.2f} GB)",
        f"Engine:            {engine_name}",
        f"Hardware:          {gpu_desc}",
        "-" * 60,
        f"Generation Speed:  {gen['tokens_per_sec']:.2f} tokens/sec",
        f"Latency per Token: {gen['ms_per_token']:.2f} ms/token",
        f"Time to 1st Token: {gen['ttft_ms']:.1f} ms "
        f"(Prompt: {len(gen['prompt_ids'])} tokens)",
        f"Total Bench Time:  {gen['gen_time_s']:.2f} s ({n_tokens} tokens)",
        "-" * 60,
        "PROMPT:",
        "  " + prompt.replace("\n", "\n  "),
        "GENERATED:",
    ]
    if text:
        for i in range(0, len(text), 58):
            lines.append("  " + text[i:i + 58].replace("\n", "\n  "))
    else:
        lines.append("  <empty>")
    lines += [
        "-" * 60,
        f"logits: min={st.get('min', 0):.3f} max={st.get('max', 0):.3f} "
        f"std={st.get('std', 0):.4f} top1p={st.get('top1_p', 0):.4f} "
        f"entropy={st.get('entropy', 0):.3f}",
        f"VERDICT: {verdict}" + (f" - {reason}" if reason else ""),
    ]
    if cpu_fallback:
        lines.append("WARNING: no accelerator was engaged (CPU fallback)")
    if verdict != "OK":
        lines += [
            "",
            "!! INVALID RUN - the speed numbers above are NOT meaningful. !!",
            "!! The engine produced degenerate output. Do not share     !!",
            "!! this scorecard.                                            !!",
        ]
    lines.append(rule)

    return {
        "model_name": fname,
        "model_size_gb": fsize_gb,
        "engine": engine_name,
        "tokens_per_sec": gen['tokens_per_sec'],
        "ms_per_token": gen['ms_per_token'],
        "ttft_ms": gen['ttft_ms'],
        "total_time_s": gen['gen_time_s'],
        "tokens_count": n_tokens,
        "hardware": gpu_desc,
        "output_text": text,
        "verdict": verdict,
        "verdict_reason": reason,
        "stats": st,
        "valid": verdict == "OK",
        "scorecard": "\n".join(lines),
    }


def run_ab(source_path, converted_path, n_tokens=24, strict=False,
           status=lambda s: print(f"  {s}")):
    """A/B a source GGUF against its own NTMRE conversion.

    Greedy decoding is deterministic given identical logits, so any
    divergence is a container bug rather than sampling noise. This is the
    comparison the suite was built for and it had never been runnable.
    """
    prompts = [
        "The capital of France is",
        "Q: What is 2 + 2?\nA:",
        "Once upon a time there was a small robot who",
    ]
    results = {}
    for tag, path in (("GGUF", source_path), ("NTMRE", converted_path)):
        eng = ntmre_router.create_engine(path, force_cpu=False, status=status)
        tok = resolve_tokenizer(eng, path)
        runs = []
        for p in prompts:
            g = generate(eng, tok, p, n_tokens)
            v, why = judge(g, strict=strict)
            g["verdict"], g["verdict_reason"] = v, why
            runs.append(g)
        if hasattr(eng, 'close'):
            try:
                eng.close()
            except Exception:
                pass
        results[tag] = runs

    rule = "=" * 74
    lines = [rule,
             "  NTMRE CONTAINER A/B  (source GGUF vs its own conversion)",
             rule]
    identical = 0
    cosines = []
    for i, p in enumerate(prompts):
        a, b = results["GGUF"][i], results["NTMRE"][i]
        same = a["ids"] == b["ids"]
        identical += same
        lines += [
            "",
            f"PROMPT {p!r}",
            f"  GGUF  {a['tokens_per_sec']:5.1f} tok/s  "
            f"ttft {a['ttft_ms']:5.0f}ms  [{a['verdict']}]",
            f"        {a['text'][:60]!r}",
            f"  NTMRE {b['tokens_per_sec']:5.1f} tok/s  "
            f"ttft {b['ttft_ms']:5.0f}ms  [{b['verdict']}]",
            f"        {b['text'][:60]!r}",
            f"  identical greedy output: {same}",
        ]
        for tag in ("GGUF", "NTMRE"):
            st = results[tag][i]["stats"]
            if st:
                cosines.append(st["top1_p"])
    ga = sum(r["tokens_per_sec"] for r in results["GGUF"]) / len(prompts)
    na = sum(r["tokens_per_sec"] for r in results["NTMRE"]) / len(prompts)
    lines += [
        "",
        rule,
        f"identical greedy output: {identical}/{len(prompts)}",
        f"speed  GGUF  {ga:.2f} tok/s avg",
        f"speed  NTMRE {na:.2f} tok/s avg",
    ]
    if identical == len(prompts):
        lines.append("VERDICT: conversion is faithful - output matches the source")
    elif identical:
        lines.append("VERDICT: PARTIAL match - investigate before trusting output")
    else:
        lines.append("VERDICT: outputs DIVERGE - the conversion is lossy or wrong")
    lines.append(rule)
    return {
        "identical": identical,
        "total": len(prompts),
        "gguf_tps": ga,
        "ntmre_tps": na,
        "report": "\n".join(lines),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NTMRE speed + correctness benchmark")
    parser.add_argument("model", nargs="?", default=None,
                        help="Path to model file (.ntmre / .gguf)")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Number of tokens to generate (default: 100)")
    parser.add_argument("--prompt", type=str,
                        default="Explain quantum computing in simple terms:",
                        help="Benchmark prompt")
    parser.add_argument("--strict", action="store_true",
                        help="Treat SUSPECT results as failures (exit != 0)")
    parser.add_argument("--ab", nargs=2,
                        metavar=("SOURCE.GGUF", "CONVERTED.NTMRE"),
                        help="A/B a source GGUF against its conversion")
    parser.add_argument("--json", action="store_true",
                        help="Also emit the result as JSON")
    args = parser.parse_args()

    if args.ab:
        print()
        print("=" * 74)
        print("  NTMRE CONTAINER A/B")
        print(f"  source:    {args.ab[0]}")
        print(f"  converted: {args.ab[1]}")
        print("=" * 74)
        print()
        res = run_ab(args.ab[0], args.ab[1],
                     n_tokens=min(args.tokens, 24), strict=args.strict)
        print()
        print(res["report"])
        print()
        return 0 if res["identical"] == res["total"] else 1

    model_path = args.model
    if not model_path:
        models = ntmre_router.discover_models(Path("."))
        if not models:
            print("No models found to benchmark. Specify a model path:")
            print("   python ntmre_benchmark.py <path/to/model.ntmre>")
            return 1
        first_label, model_path = next(iter(models.items()))
        print(f"Auto-selected model: {first_label}")

    print()
    print("=" * 60)
    print("  STARTING NTMRE BENCHMARK (speed + correctness)")
    print(f"  Model:  {model_path}")
    print(f"  Tokens: {args.tokens}")
    print("=" * 60)
    print()

    res = run_benchmark(model_path, n_tokens=args.tokens, prompt=args.prompt,
                        strict=args.strict)
    print()
    print(res["scorecard"])
    print()

    if args.json:
        import json
        print(json.dumps(res, indent=2, default=str))
        print()

    if res["verdict"] == "INVALID":
        return 2
    if args.strict and res["verdict"] != "OK":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())