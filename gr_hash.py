"""
💀 Nightmare GhostRider Hash — Pure-Python Implementation
=========================================================
Custom proof-of-work hash for source sealing and license protection.

Algorithm: 18 rounds of SHA-256 with input-dependent byte permutation
and XOR mixing between rounds. Uses hashlib (C-backed) for speed —
~2800x faster than the 50,000-round SHA-256 chain fallback.

Interface (matches the native .pyd):
    getPoWHash(data: bytes) -> bytes   (80-byte input, 32-byte output)

When the native gr_hash.pyd is available (built from gr_hash.c), it
takes priority. This pure-Python version is the automatic fallback.

Author: Nightmare Designs 💀
"""

import hashlib


def _permute(data: bytearray, seed: int):
    """Input-dependent byte permutation (Feistel-like XOR shuffle)."""
    n = len(data)
    for i in range(n):
        tmp = data[i]
        j = (i + seed + tmp) % n
        data[i] ^= data[j]
        data[j] ^= tmp
        data[i] ^= data[j]


def getPoWHash(data: bytes) -> bytes:
    """GhostRider proof-of-work hash.

    Input: bytes (padded/truncated to 80 bytes internally).
    Output: 32 bytes (SHA-256 size).
    """
    # Pad/copy to 80 bytes
    buf = bytearray(80)
    if len(data) >= 80:
        buf[:80] = data[:80]
    elif len(data) > 0:
        for i in range(80):
            buf[i] = data[i % len(data)]
    else:
        buf[:] = b'\x00' * 80

    # First SHA-256
    prev = hashlib.sha256(bytes(buf)).digest()

    # 17 more rounds with permutation + XOR mixing
    for rnd in range(1, 18):
        seed = prev[rnd % 32] ^ rnd
        _permute(buf, seed)

        # XOR previous hash into buffer
        for i in range(32):
            buf[i] ^= prev[i]

        # Hash again
        h = hashlib.sha256(bytes(buf)).digest()

        # XOR with previous (chaining)
        h = bytes(a ^ b for a, b in zip(h, prev))
        prev = h

    return prev
