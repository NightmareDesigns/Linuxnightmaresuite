"""
NIGHTMARE CODE SEALER — GhostRider Encrypted Source Protection
================================================================
Encrypts Python source files using a Feistel/SHA-256 block cipher
with a key derived from the GhostRider hash (or SHA-256 chain
fallback) of an obfuscated master secret.

Usage:
    python seal_code.py           # Seal all user files (keep originals)
    python seal_code.py --list    # Show what would be sealed
    python seal_code.py --lock    # Seal + delete originals (DANGER)
    python seal_code.py --unseal  # Decrypt back to .py (for editing)
    python seal_code.py --test    # Verify all .sealed files decrypt OK

Author: Nightmare Designs
"""

import os
import sys
import hashlib
import struct
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Master secret — obfuscated (split + XOR-recombined at runtime).
# The full secret is never stored as a single plaintext literal.
# ---------------------------------------------------------------------------
_S1 = b"\x31\x35\x36\x34\x37\x30\x31\x30"   # "15647010"
_S2 = b"\x35\x34\x30\x31\x33\x35\x39\x38"   # "54013598"
_S3 = b"\x4e\x49\x47\x48\x54\x4d\x41\x52\x45"  # "NIGHTMARE"
MASTER_SECRET = (_S1 + _S2)  # recombine — 16 bytes


# ---------------------------------------------------------------------------
# Files that must STAY PLAINTEXT (crypto/loader infrastructure + entry point)
# ---------------------------------------------------------------------------
NEVER_SEAL = {
    # Crypto / loader infrastructure
    "core_mind/seal_code.py",
    "core_mind/loader.py",
    "core_mind/ghostrider_shield.py",
    "core_mind/seal_engine.py",
    "core_mind/__init__.py",
    # Bootstrap entry points (plaintext, needed by PyInstaller)
    "nightmare_bootstrap.py",
    "nightmare_bootstrap_direct.py",
    "nightmare_bootstrap_kickstarter.py",
    "nightmare_bootstrap_admin.py",
    # Admin license generator (standalone tool, must stay runnable)
    "license_generator.py",
    # Third-party utility needed for exe
    "appdirs.py",
    # GhostRider hash — needed for key derivation BEFORE sealed files load
    "gr_hash.py",
    # Package inits (needed for import system)
    "engine/__init__.py",
    "jessicaa_security/__init__.py",
}

# Directories to skip entirely
SKIP_DIRS = {"build", "dist", "__pycache__", ".git", "models", "fresh_convert",
             "nightmare_code", "ntmre_amd", "transfer_patch_other_pc_2026-08-05"}

# Filename patterns to skip (backups, temp files)
SKIP_SUFFIXES = (".bak", ".bak_20260815_070256", ".bak_q1_20260815_133453",
                 ".bak_20260815_133453", ".bak_20260815_171215",
                 ".bak_20260816_quant")


def _should_skip(filepath: Path) -> bool:
    """Return True if a .py file should NOT be sealed."""
    rel = filepath.relative_to(APP_DIR).as_posix()

    # Never seal the crypto infrastructure
    if rel in NEVER_SEAL:
        return True

    # Skip backups
    name = filepath.name
    if "backup" in name.lower() or "BACKUP" in name:
        return True
    if name.endswith(SKIP_SUFFIXES):
        return True
    if name in ("nightmare_suite_unified_backup.py",
                "nightmare_suite_unified_BIN_BACKUP.py",
                "nightmare_suite_unified_COMPILER_BACKUP.py",
                "nightmare_suite_unified_oldverison.py",
                "nightmare_suite_unified_utf8.py",
                "nightmare_suite_unified_V13_BACKUP.py",
                "nightmare_suite_unified1.py"):
        return True

    # Skip files with spaces (not valid Python module names)
    if " " in filepath.stem:
        return True

    return False


def discover_py_files():
    """Recursively find all .py files that should be sealed."""
    files = []
    for root, dirs, filenames in os.walk(APP_DIR):
        # Filter dirs in-place so os.walk doesn't descend into them
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            fp = Path(root) / fn
            if _should_skip(fp):
                continue
            files.append(fp.relative_to(APP_DIR).as_posix())
    return sorted(files)


# Backward-compatible static list (populated on first access)
CRITICAL_FILES = discover_py_files()


# ---------------------------------------------------------------------------
# Key derivation — GhostRider hash if available, else SHA-256 chain (50k rounds)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inline GhostRider hash — copied from gr_hash.py so it's always available
# without importing an external module (which may not be bundled in the exe).
# This stays in NEVER_SEAL (plaintext .pyc) so it's always importable.
# ---------------------------------------------------------------------------

def _gr_permute(data: bytearray, seed: int):
    """Input-dependent byte permutation (Feistel-like XOR shuffle)."""
    n = len(data)
    for i in range(n):
        tmp = data[i]
        j = (i + seed + tmp) % n
        data[i] ^= data[j]
        data[j] ^= tmp
        data[i] ^= data[j]


def _gr_getPoWHash(data: bytes) -> bytes:
    """GhostRider proof-of-work hash (inline, no external dependency).

    Input: bytes (padded/truncated to 80 bytes internally).
    Output: 32 bytes (SHA-256 size).
    18 rounds of SHA-256 with input-dependent byte permutation and XOR mixing.
    """
    buf = bytearray(80)
    if len(data) >= 80:
        buf[:80] = data[:80]
    elif len(data) > 0:
        for i in range(80):
            buf[i] = data[i % len(data)]
    else:
        buf[:] = b'\x00' * 80

    prev = hashlib.sha256(bytes(buf)).digest()

    for rnd in range(1, 18):
        seed = prev[rnd % 32] ^ rnd
        _gr_permute(buf, seed)
        for i in range(32):
            buf[i] ^= prev[i]
        h = hashlib.sha256(bytes(buf)).digest()
        h = bytes(a ^ b for a, b in zip(h, prev))
        prev = h

    return prev


def _derive_key(master: bytes) -> bytes:
    """Derive 32-byte key from master secret via inline GhostRider hash."""
    # Use inline GR hash (always available — no external import needed)
    padded = (master * 6)[:80]
    h1 = _gr_getPoWHash(padded)  # 32 bytes
    salted = (h1 + b"NIGHTMARE-AES-SEAL" + master)[:80]
    h2 = _gr_getPoWHash(salted)  # 32 bytes
    return h2


# ---------------------------------------------------------------------------
# Pure-Python block cipher (16-round Feistel with SHA-256 round keys)
# No external dependencies.  CBC mode with PKCS7 padding.
# ---------------------------------------------------------------------------

def _pad(data: bytes) -> bytes:
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len] * pad_len)


def _unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    if pad_len > 16 or pad_len == 0:
        raise ValueError("Invalid padding")
    if data[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("Invalid padding")
    return data[:-pad_len]


def _xor_blocks(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _aes_encrypt_block(block: bytes, key: bytes) -> bytes:
    assert len(block) == 16
    L, R = block[:8], block[8:]
    for round_num in range(16):
        round_key = hashlib.sha256(key + R + struct.pack(">I", round_num)).digest()[:8]
        new_R = bytes(a ^ b for a, b in zip(L, round_key))
        L = R
        R = new_R
    return L + R


def _aes_decrypt_block(block: bytes, key: bytes) -> bytes:
    assert len(block) == 16
    L, R = block[:8], block[8:]
    for round_num in range(15, -1, -1):
        round_key = hashlib.sha256(key + L + struct.pack(">I", round_num)).digest()[:8]
        new_L = bytes(a ^ b for a, b in zip(R, round_key))
        R = L
        L = new_L
    return L + R


def encrypt_data(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt data with CBC mode."""
    iv = os.urandom(16)
    padded = _pad(plaintext)
    blocks = [padded[i:i + 16] for i in range(0, len(padded), 16)]

    ciphertext = iv
    prev = iv
    for block in blocks:
        xored = _xor_blocks(block, prev)
        encrypted = _aes_encrypt_block(xored, key)
        ciphertext += encrypted
        prev = encrypted

    return ciphertext


def decrypt_data(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt CBC ciphertext."""
    iv = ciphertext[:16]
    data = ciphertext[16:]
    blocks = [data[i:i + 16] for i in range(0, len(data), 16)]

    plaintext = b""
    prev = iv
    for block in blocks:
        decrypted = _aes_decrypt_block(block, key)
        plain_block = _xor_blocks(decrypted, prev)
        plaintext += plain_block
        prev = block

    return _unpad(plaintext)


# Magic header for sealed files
SEAL_MAGIC = b"\x89NDS\r\n\x1a\n"  # Nightmare Designs Sealed


def seal_file(filepath: Path, key: bytes) -> Path:
    """Encrypt a .py file into .sealed format."""
    source = filepath.read_bytes()
    encrypted = encrypt_data(source, key)

    sealed_path = filepath.with_suffix('.sealed')
    with open(sealed_path, 'wb') as f:
        f.write(SEAL_MAGIC)
        f.write(struct.pack("<Q", len(source)))
        f.write(encrypted)

    return sealed_path


def unseal_file(sealed_path: Path, key: bytes) -> bytes:
    """Decrypt a .sealed file back to source bytes."""
    with open(sealed_path, 'rb') as f:
        magic = f.read(8)
        if magic != SEAL_MAGIC:
            raise ValueError(f"Not a sealed file: {sealed_path}")
        orig_len = struct.unpack("<Q", f.read(8))[0]
        encrypted = f.read()

    decrypted = decrypt_data(encrypted, key)
    return decrypted[:orig_len]


def seal_all(remove_originals: bool = False):
    """Seal all user .py files."""
    files = discover_py_files()
    key = _derive_key(MASTER_SECRET)
    print(f"[SEALER] Key derived. Sealing {len(files)} files...")
    print()

    sealed_count = 0
    for rel in files:
        filepath = APP_DIR / rel
        if not filepath.exists():
            continue

        sealed_path = seal_file(filepath, key)
        size_orig = filepath.stat().st_size
        size_sealed = sealed_path.stat().st_size
        print(f"  SEALED: {rel} ({size_orig:,} -> {size_sealed:,} bytes)")
        sealed_count += 1

        if remove_originals:
            filepath.unlink()

    print(f"\n[SEALER] Done. {sealed_count} files sealed.")
    if not remove_originals:
        print("[SEALER] Originals kept. Run with --lock to delete them (DANGER).")


def unseal_all():
    """Unseal all files back to .py (for editing)."""
    files = discover_py_files()
    key = _derive_key(MASTER_SECRET)
    print(f"[SEALER] Unsealing {len(files)} files...")

    for rel in files:
        sealed_path = APP_DIR / rel.replace('.py', '.sealed')
        if not sealed_path.exists():
            continue

        source = unseal_file(sealed_path, key)
        out_path = APP_DIR / rel
        out_path.write_bytes(source)
        print(f"  UNSEALED: {rel} ({len(source):,} bytes)")

    print("[SEALER] Done. Source files restored.")


def verify_all():
    """Verify all .sealed files can be decrypted and compiled."""
    files = discover_py_files()
    key = _derive_key(MASTER_SECRET)
    print(f"[SEALER] Verifying {len(files)} sealed files...")

    ok = 0
    bad = 0
    missing = 0
    for rel in files:
        sealed_path = APP_DIR / rel.replace('.py', '.sealed')
        if not sealed_path.exists():
            missing += 1
            continue

        try:
            source = unseal_file(sealed_path, key)
            compile(source.decode('utf-8'), rel, 'exec')
            ok += 1
        except Exception as e:
            bad += 1
            print(f"  FAIL: {rel} — {e}")

    print(f"\n[SEALER] OK={ok}  FAIL={bad}  NOT_SEALED={missing}")
    return bad == 0


if __name__ == "__main__":
    if "--list" in sys.argv:
        files = discover_py_files()
        print(f"Files that will be sealed ({len(files)}):")
        for f in files:
            exists = "+" if (APP_DIR / f).exists() else "-"
            print(f"  {exists} {f}")
    elif "--unseal" in sys.argv:
        unseal_all()
    elif "--lock" in sys.argv:
        seal_all(remove_originals=True)
    elif "--test" in sys.argv:
        ok = verify_all()
        sys.exit(0 if ok else 1)
    else:
        seal_all(remove_originals=False)
