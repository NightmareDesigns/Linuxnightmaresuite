"""
NIGHTMARE ENGINE SEALER — GhostRider Encrypted Source Protection
=================================================================
Extends the code sealer to handle .cu, .c, .h files in engine/.
Same crypto, same key, same .sealed format.

Usage:
    python seal_engine.py           # Seal all engine source files
    python seal_engine.py --list    # Show what would be sealed
    python seal_engine.py --unseal  # Decrypt back (for editing)
    python seal_engine.py --lock    # Seal + delete originals

Author: Nightmare Designs
"""

import os
import sys
from pathlib import Path

# Setup paths
APP_DIR = Path(__file__).parent.parent
ENGINE_DIR = APP_DIR.parent / "engine"
sys.path.insert(0, str(APP_DIR))

from core_mind.seal_code import (
    MASTER_SECRET, SEAL_MAGIC, _derive_key, encrypt_data, decrypt_data
)

# Engine source files to seal
ENGINE_FILES = [
    "crucible_v4.cu",
    "crucible_v4.c",
    "kernels_fused.cu",
    "collapse.cu",
    "nightmare_engine.cu",
    "lost_souls.c",
    "lost_souls.h",
    "ntmre_cpu.c",
    "ntmre_export.c",
    "ntmre_format.h",
    "gradient_sync.h",
    "gradient_sync.c",
    "hip_compat.h",
    # Python tools in engine/
    "nightmare_compress.py",
    "nightmare_convert.py",
    # Added v7
    "kernels.cu",
    "crucible_host.cu",
    "test_nvrtc.c",
]


import struct

def seal_engine_file(filepath: Path, key: bytes) -> Path:
    """
    Encrypt a source file into .sealed format.
    Output: filename.ext.sealed (preserves original extension info)
    """
    source = filepath.read_bytes()
    encrypted = encrypt_data(source, key)

    # Store original filename in header so we know what to restore
    orig_name = filepath.name.encode('utf-8')
    sealed_path = filepath.parent / (filepath.name + ".sealed")

    with open(sealed_path, 'wb') as f:
        f.write(SEAL_MAGIC)
        f.write(struct.pack("<H", len(orig_name)))  # filename length
        f.write(orig_name)                          # original filename
        f.write(struct.pack("<Q", len(source)))     # original size
        f.write(encrypted)                          # encrypted content

    return sealed_path


def unseal_engine_file(sealed_path: Path, key: bytes) -> tuple:
    """Decrypt a .sealed engine file. Returns (original_filename, source_bytes)."""
    with open(sealed_path, 'rb') as f:
        magic = f.read(8)
        if magic != SEAL_MAGIC:
            raise ValueError(f"Not a sealed file: {sealed_path}")
        name_len = struct.unpack("<H", f.read(2))[0]
        orig_name = f.read(name_len).decode('utf-8')
        orig_size = struct.unpack("<Q", f.read(8))[0]
        encrypted = f.read()

    decrypted = decrypt_data(encrypted, key)
    return orig_name, decrypted[:orig_size]


def seal_engine(remove_originals: bool = False):
    """Seal all engine source files."""
    if not ENGINE_DIR.exists():
        print(f"[ENGINE SEALER] ERROR: Engine directory not found: {ENGINE_DIR}")
        return

    key = _derive_key(MASTER_SECRET)
    print(f"[ENGINE SEALER] Key derived. Sealing {len(ENGINE_FILES)} engine files...")
    print(f"[ENGINE SEALER] Target: {ENGINE_DIR}")
    print()

    sealed_count = 0
    for filename in ENGINE_FILES:
        filepath = ENGINE_DIR / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} (not found)")
            continue

        sealed_path = seal_engine_file(filepath, key)
        size_orig = filepath.stat().st_size
        size_sealed = sealed_path.stat().st_size
        print(f"  SEALED: {filename} ({size_orig:,} -> {size_sealed:,} bytes)")
        sealed_count += 1

        if remove_originals:
            filepath.unlink()
            print(f"         (original deleted)")

    print(f"\n[ENGINE SEALER] Done. {sealed_count} engine files sealed.")
    if not remove_originals:
        print("[ENGINE SEALER] Originals kept. Run with --lock to delete them.")


def unseal_engine():
    """Unseal all engine files back to source (for editing)."""
    key = _derive_key(MASTER_SECRET)
    print(f"[ENGINE SEALER] Unsealing engine files...")

    for filename in ENGINE_FILES:
        sealed_path = ENGINE_DIR / (filename + ".sealed")
        if not sealed_path.exists():
            continue

        orig_name, source = unseal_engine_file(sealed_path, key)
        out_path = ENGINE_DIR / filename
        out_path.write_bytes(source)
        print(f"  UNSEALED: {filename} ({len(source):,} bytes)")

    print("[ENGINE SEALER] Done. Source files restored.")


def list_engine_files():
    """Show what would be sealed."""
    print("Engine files to seal:")
    for f in ENGINE_FILES:
        path = ENGINE_DIR / f
        exists = "✓" if path.exists() else "✗"
        size = f"({path.stat().st_size:,} bytes)" if path.exists() else ""
        print(f"  {exists} {f} {size}")


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_engine_files()
    elif "--unseal" in sys.argv:
        unseal_engine()
    elif "--lock" in sys.argv:
        seal_engine(remove_originals=True)
    else:
        seal_engine(remove_originals=False)
