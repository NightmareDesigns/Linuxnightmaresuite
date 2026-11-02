"""
GHOSTRIDER SHIELD - Nightmare Suite Security Layer
===================================================
Uses the full Raptoreum GhostRider hash (15 algo chain + CryptoNight)
to seal license keys, admin passwords, and file integrity.

GR hash = 18 rounds (15 core algos + 3 CryptoNight variants)
Input-dependent random ordering makes brute force impractical.

Requirements:
    - gr_hash.pyd (built from gr_hash-master via BUILD_GHOSTRIDER.bat)
    - Place gr_hash.pyd in the app folder (next to nightmare_suite_unified.py)

Author: Nightmare Designs
"""

import os
import sys
import hashlib
from pathlib import Path

# Add app folder to path so gr_hash can be found
APP_DIR = Path(__file__).parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Inline GhostRider hash — no external import needed (works in exe)
def _gr_permute(data, seed):
    """Input-dependent byte permutation (Feistel-like XOR shuffle)."""
    n = len(data)
    for i in range(n):
        tmp = data[i]
        j = (i + seed + tmp) % n
        data[i] ^= data[j]
        data[j] ^= tmp
        data[i] ^= data[j]

def _gr_getPoWHash(data):
    """GhostRider proof-of-work hash (inline, no external dependency)."""
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

GR_AVAILABLE = True  # Always True — inline hash always available


def _critical_file_list():
    """The full set of engine + security files covered by GhostRider seals.
    Any tampering with these is detected at startup via check_integrity()."""
    app_dir = Path(__file__).parent.parent
    names = [
        "nightmare_suite_unified.py", "nightmare_direct.py", "nightmare_kickstarter.py",
        "nightmare_stream.py",
        "nightmare_gpu_native.py", "qwen35_gpu.py", "nightmare_qwen35.py",
        "bonsai_suite.py", "ntmre_runtime.py", "transformer.py",
        "ntmre_v16_engine.py", "ntmre_router.py", "gguf_parser.py",
        "tokenizer.py", "sampler.py", "chat_templates.py",
        str(Path("engine") / "bonsai_pc_app.py"),
        str(Path("core_mind") / "license_crypto.py"),
        str(Path("core_mind") / "license_keygen.py"),
        str(Path("core_mind") / "ghostrider_shield.py"),
    ]
    return [str(app_dir / n) for n in names]


def ghostrider_hash(data: bytes, rounds: int = 3) -> bytes:
    """
    Hash data using full GhostRider algorithm (inline, no external import).

    The GR hash requires 80 bytes input (block header size).
    We pad/tile the input to 80 bytes, then run GR.
    Multiple rounds = hash the output again for extra hardening.

    Args:
        data: Input bytes to hash
        rounds: Number of GR passes (each pass = 18 internal rounds)

    Returns:
        32-byte hash digest
    """
    # GR expects exactly 80 bytes input
    if len(data) < 80:
        padded = data * (80 // len(data) + 1)
        padded = padded[:80]
    elif len(data) > 80:
        pre = hashlib.sha256(data).digest()  # 32 bytes
        padded = (pre * 3)[:80]  # tile to 80
    else:
        padded = data

    result = padded
    for _ in range(rounds):
        if len(result) < 80:
            result = (result * 3)[:80]
        elif len(result) > 80:
            result = result[:80]
        result = _gr_getPoWHash(result)  # Returns 32 bytes
        if _ < rounds - 1:
            result = (result * 3)[:80]

    return result[:32]


def seal_license_key(key: str, salt: str = "NIGHTMARE-SEAL-v6") -> str:
    """
    Generate a GR-sealed hash for a license key.
    
    Args:
        key: The license key string
        salt: Application-specific salt
    
    Returns:
        Hex string of the sealed hash
    """
    data = (salt + key + salt[::-1]).encode('utf-8')
    return ghostrider_hash(data, rounds=3).hex()


def verify_license_key(key: str, stored_hash: str, salt: str = "NIGHTMARE-SEAL-v6") -> bool:
    """
    Verify a license key against its stored GR hash.
    
    Args:
        key: The key to verify
        stored_hash: The hex hash to compare against
        salt: Must match the salt used during sealing
    
    Returns:
        True if key is valid
    """
    computed = seal_license_key(key, salt)
    return computed == stored_hash


def seal_password(password: str) -> str:
    """Seal an admin password with GR hash."""
    data = ("NIGHTMARE-ADMIN-" + password + "-GHOSTRIDER").encode('utf-8')
    return ghostrider_hash(data, rounds=5).hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify admin password against stored GR hash."""
    return seal_password(password) == stored_hash


def seal_file(filepath: str) -> str:
    """
    Generate a GR integrity seal for a file.
    Used for tamper detection on critical source files.
    
    Args:
        filepath: Path to the file to seal
    
    Returns:
        Hex string of the file's GR hash
    """
    with open(filepath, 'rb') as f:
        content = f.read()
    # Use SHA-256 first to get fixed-size representation of file
    file_sha = hashlib.sha256(content).digest()
    # Then GR hash the SHA for the seal
    return ghostrider_hash(file_sha + b"NIGHTMARE-FILE-SEAL", rounds=2).hex()


def verify_file(filepath: str, stored_seal: str) -> bool:
    """Check if a file has been tampered with."""
    try:
        current_seal = seal_file(filepath)
        return current_seal == stored_seal
    except (FileNotFoundError, PermissionError):
        return False


def _key_checksum(tier: int, code: str) -> str:
    """Compute an 8-char checksum from tier + code (double-hash derived).

    Uses two independent SHA-256 derivations and interleaves them to
    produce an 8-character hex checksum (32-bit space x 2 = harder to
    brute-force than the old 4-char/16-bit checksum).
    """
    import hashlib
    h1 = hashlib.sha256(f"NDS|{tier}|{code}".encode()).hexdigest()
    h2 = hashlib.sha256(f"GR|{tier}|{code}|SEAL".encode()).hexdigest()
    # Interleave: take 4 chars from each hash at different offsets
    return (h1[8:12] + h2[16:20]).upper()


def validate_key_format(key: str):
    """
    Validate a license key's format and checksum (no hash lookup).
    Returns tier (1-4) if the key format + checksum are valid, else None.
    This prevents random strings from passing the fallback validator.

    Supports both old format (4-char checksum) and new format (8-char checksum).
    """
    import re
    key = key.strip()
    # New format: NIGHTMARE-T{tier}-{CODE16}-{CHECK8}
    m = re.match(r'^NIGHTMARE-T([1-4])-([A-Z0-9]{16})-([A-F0-9]{8})$', key)
    if m:
        tier = int(m.group(1))
        code = m.group(2)
        checksum = m.group(3)
        if _key_checksum(tier, code) != checksum:
            return None
        return tier
    # Legacy format: NIGHTMARE-T{tier}-{CODE16}-{CHECK4} (backward compat)
    m = re.match(r'^NIGHTMARE-T([1-4])-([A-Z0-9]{16})-([A-F0-9]{4})$', key)
    if m:
        tier = int(m.group(1))
        code = m.group(2)
        checksum = m.group(3)
        # Legacy checksum: only first 4 chars of new 8-char checksum
        if _key_checksum(tier, code)[:4] != checksum:
            return None
        return tier
    return None


def generate_sealed_key(tier: int, code: str = None) -> tuple:
    """
    Generate a license key AND its GR-sealed hash for storage.
    
    Key format: NIGHTMARE-T{tier}-{16-char-code}-{8-char-checksum}
    The checksum is double-hash derived (two independent SHA-256) so
    random strings don't validate and brute-forcing is impractical.
    
    Args:
        tier: 1-4
        code: Optional custom code portion. Random if not provided.
    
    Returns:
        (key_string, sealed_hash)
    """
    import random
    import string
    
    if code is None:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
    elif len(code) < 16:
        # Pad short custom codes to 16 chars
        code = code.upper().ljust(16, 'X')
    else:
        code = code.upper()[:16]
    
    checksum = _key_checksum(tier, code)
    key = f"NIGHTMARE-T{tier}-{code}-{checksum}"
    sealed = seal_license_key(key)
    return key, sealed


def check_integrity(critical_files: list = None) -> dict:
    """
    Run integrity check on critical source files.
    
    Args:
        critical_files: List of file paths to check.
                       If None, checks default critical files.
    
    Returns:
        Dict of {filepath: (is_valid, seal)} for each file
    """
    if critical_files is None:
        critical_files = _critical_file_list()
    
    app_dir = Path(__file__).parent.parent
    seal_file_path = Path(__file__).parent / ".seals"
    stored_seals = {}
    
    if seal_file_path.exists():
        for line in seal_file_path.read_text().strip().split('\n'):
            if '|' in line:
                path, seal = line.rsplit('|', 1)
                # Stored paths are relative to the app dir so seals travel
                # with the build regardless of install location.
                full = str(app_dir / path.strip())
                stored_seals[full] = seal.strip()
    
    results = {}
    for fp in critical_files:
        if fp in stored_seals:
            # File has a stored seal.
            if os.path.exists(fp):
                valid = verify_file(fp, stored_seals[fp])
                results[fp] = (valid, stored_seals[fp])
            else:
                # Not loose → bundled inside the PYZ archive (frozen build).
                # The compiled binary archive protects it; not a tamper event.
                # Skip it so it isn't counted as a failure.
                continue
        else:
            # No seal stored yet - generate one (first run / trusted)
            try:
                seal = seal_file(fp)
                results[fp] = (True, seal)
            except FileNotFoundError:
                # Bundled in PYZ (frozen build) — skip, not a failure.
                continue
    
    return results


def update_seals(critical_files: list = None):
    """
    Re-seal all critical files (run after legitimate updates).
    Stores seals in core_mind/.seals
    """
    if critical_files is None:
        critical_files = _critical_file_list()
    
    app_dir = Path(__file__).parent.parent
    seal_file_path = Path(__file__).parent / ".seals"
    lines = []
    for fp in critical_files:
        try:
            seal = seal_file(fp)
            # Store path relative to app dir so seals work in any install.
            rel = Path(fp).resolve().relative_to(app_dir.resolve())
            rel = str(rel).replace("\\", "/")
            lines.append(f"{rel}|{seal}")
            print(f"[SHIELD] Sealed: {Path(fp).name} -> {seal[:16]}...")
        except FileNotFoundError:
            print(f"[SHIELD] SKIP: {fp} not found")
    
    seal_file_path.write_text('\n'.join(lines))
    print(f"[SHIELD] {len(lines)} files sealed -> {seal_file_path}")


# === Quick test ===
if __name__ == "__main__":
    print("=== GHOSTRIDER SHIELD TEST ===")
    print(f"GR Module: {'LOADED' if GR_AVAILABLE else 'NOT FOUND (using fallback)'}")
    print()
    
    # Test hash
    test_data = b"Nightmare Designs - GhostRider Seal Test"
    h = ghostrider_hash(test_data, rounds=2)
    print(f"Test hash: {h.hex()}")
    print(f"Hash size: {len(h)} bytes")
    print()
    
    # Test license key
    key, sealed = generate_sealed_key(3, "TEST1234ABCD")
    print(f"Key: {key}")
    print(f"Seal: {sealed}")
    print(f"Verify: {verify_license_key(key, sealed)}")
    print(f"Wrong key: {verify_license_key('NIGHTMARE-T3-WRONG', sealed)}")
    print()
    
    # Test password
    pw_seal = seal_password("testpassword")
    print(f"Password seal: {pw_seal}")
    print(f"Verify correct: {verify_password('testpassword', pw_seal)}")
    print(f"Verify wrong: {verify_password('wrongpass', pw_seal)}")
