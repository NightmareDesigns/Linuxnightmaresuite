"""
license_crypto.py — Nightmare Suite asymmetric license layer
=============================================================
Ed25519 signed licenses + hardware binding (node-locking).

This is the *verification* side only — it lives in the distributed app.
It contains the PUBLIC key (32 bytes).  The private key never ships; it
stays on the developer's offline machine and is used by license_keygen.py
to sign license payloads.

License key format
------------------
    ND-<base32 groups, 8 chars each, dash-separated>

The base32 decodes to an 82-byte blob:
    message (18 bytes):
        tier      : 1 byte   (1-4)
        flags     : 1 byte   (bit0 = HWID-bound, bit1 = expiry-bound)
        hwid      : 12 bytes (machine HWID, or zeros if floating)
        expiry    : 4 bytes  (uint32 BE: day-number since epoch, 0 = never)
    signature (64 bytes): Ed25519 over `message`

Verification
------------
    1. base32-decode the blob
    2. split message (18) / signature (64)
    3. Ed25519-verify signature against the embedded public key
    4. if flags bit0 (HWID-bound): current machine HWID must match
    5. if expiry != 0: must not be past expiry day

GhostRider remains in charge of .sealed module decryption and file
integrity; this module only handles license authenticity + node-locking.

Author: Nightmare Designs
"""

import base64
import hashlib
import struct
import time

# ── Embedded Ed25519 public key (raw 32 bytes) ──────────────────────
# Extracted from nightmare_public.pem.  The matching private key is held
# offline and is NOT present anywhere in the distributed software.
NIGHTMARE_PUBLIC_KEY_HEX = (
    "6bf915314677b12c0572533fb63cb000afc5753b7d40efd0d4dd6713f344d409"
)

# Payload layout: tier(1) flags(1) hwid(12) expiry(4) = 18 bytes
_MSG_FMT = ">BB12sI"
_MSG_SIZE = struct.calcsize(_MSG_FMT)        # 18
_SIG_SIZE = 64
_BLOB_SIZE = _MSG_SIZE + _SIG_SIZE           # 82

FLAG_HWID_BOUND = 0x01
FLAG_EXPIRY_BOUND = 0x02


# ── Ed25519 verification backend ────────────────────────────────────
_verify_backend = None   # 'cryptography' | 'pycryptodome' | 'none'
_verify_fn = None        # cached callable verify(message, sig) -> bool
_pub_obj = None


def _init_backend():
    """Lazily build the Ed25519 public-key verifier.

    Tries the `cryptography` package first (bundled in the PyInstaller
    build), then PyCryptodome.  Returns a callable verify(message, sig)
    -> bool, or None if no backend is available (caller falls back to
    the legacy GhostRider validator).
    """
    global _verify_backend, _pub_obj, _verify_fn
    if _verify_fn is not None:
        return _verify_fn
    if _verify_backend == "none":
        return None
    raw = bytes.fromhex(NIGHTMARE_PUBLIC_KEY_HEX)

    # 1) cryptography (preferred — already in the build)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        pub = Ed25519PublicKey.from_public_bytes(raw)

        def _verify_cryptography(message, signature):
            try:
                pub.verify(signature, message)
                return True
            except InvalidSignature:
                return False

        _pub_obj = pub
        _verify_backend = "cryptography"
        _verify_fn = _verify_cryptography
        return _verify_cryptography
    except Exception:
        pass

    # 2) PyCryptodome fallback
    try:
        from Crypto.Signature import eddsa
        # import_public_key accepts raw 32-byte Ed25519 public keys
        pub = eddsa.import_public_key(raw)
        _scheme = eddsa.new(pub, 'rfc8032')   # one reusable verify scheme

        def _verify_pycryptodome(message, signature):
            try:
                # NOTE: PyCryptodome's verify(msg, sig) takes the opposite
                # argument order from `cryptography`'s verify(sig, msg).
                _scheme.verify(message, signature)
                return True
            except (ValueError, TypeError):
                return False

        _pub_obj = pub
        _verify_backend = "pycryptodome"
        _verify_fn = _verify_pycryptodome
        return _verify_pycryptodome
    except Exception:
        pass

    _verify_backend = "none"
    return None


# ── License key parsing + verification ───────────────────────────────
def _b32_decode(text: str) -> bytes:
    """Decode the ND-<groups> key body into raw bytes."""
    body = text.strip().upper().replace(" ", "").replace("-", "").replace("\n", "")
    # strip an optional trailing checksum fragment after a final dash is
    # already removed; re-add RFC-4648 padding.
    pad = (-len(body)) % 8
    return base64.b32decode(body + ("=" * pad))


def parse_license_key(key: str):
    """Verify and parse an Ed25519-signed license key.

    Returns (tier, flags, hwid_bytes, expiry_day) on success, else None.
    Never raises — bad input simply returns None so the caller can fall
    through to the legacy validator.
    """
    if not key:
        return None
    k = key.strip()
    # Accept "ND-..." or a bare base32 blob
    if k.startswith("ND-"):
        k = k[3:]
    elif not k[:1].isalpha():
        return None

    verify = _init_backend()
    if verify is None:
        return None   # no Ed25519 backend — let legacy validator handle it

    try:
        blob = _b32_decode(k)
    except Exception:
        return None
    if len(blob) != _BLOB_SIZE:
        return None

    message = blob[:_MSG_SIZE]
    signature = blob[_MSG_SIZE:]
    if not verify(message, signature):
        return None

    try:
        tier, flags, hwid, expiry = struct.unpack(_MSG_FMT, message)
    except struct.error:
        return None
    if tier < 1 or tier > 4:
        return None
    return tier, flags, hwid, expiry


def is_expired(expiry_day: int, now_day: int = None) -> bool:
    """expiry_day is a day-number (days since 1970-01-01); 0 = never."""
    if not expiry_day:
        return False
    if now_day is None:
        now_day = int(time.time() // 86400)
    return now_day > expiry_day


# ── Hardware ID (node-locking) ──────────────────────────────────────
def _wmic_value(cmd: list) -> str:
    """Run a WMIC/PowerShell query and return its first non-header line."""
    import subprocess
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                      timeout=6).decode("utf-8", "ignore")
    except Exception:
        return ""
    for line in out.splitlines():
        line = line.strip()
        if line and line.lower() not in ("uuid", "processorid", "serialnumber",
                                        "value", ""):
            return line
    return ""


def _ps_value(script: str) -> str:
    """Run a PowerShell one-liner and return its trimmed output."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            stderr=subprocess.DEVNULL, timeout=8).decode("utf-8", "ignore")
        return out.strip()
    except Exception:
        return ""


def _mac_address() -> str:
    """First non-zero MAC as a hex string (uuid.getnode fallback)."""
    try:
        import uuid
        node = uuid.getnode()
        if node:
            return f"{node:012x}"
    except Exception:
        pass
    return ""


def get_hwid() -> bytes:
    """Return a stable 12-byte hardware fingerprint for this machine.

    Combines motherboard UUID, CPU processor id, and the primary MAC.
    Each source is best-effort and falls back to "" — the HWID is still
    stable as long as the available sources don't change.
    """
    parts = []

    # Motherboard / product UUID — try WMIC then PowerShell
    mb = _wmic_value(["wmic", "csproduct", "get", "UUID"]) or\
        _ps_value("(Get-CimInstance Win32_ComputerSystemProduct).UUID")
    parts.append(mb)

    # CPU processor id
    cpu = _wmic_value(["wmic", "cpu", "get", "ProcessorId"]) or\
        _ps_value("(Get-CimInstance Win32_Processor).ProcessorId")
    parts.append(cpu)

    # MAC address
    parts.append(_mac_address())

    # Fallback: hostname + any MAC if the above all failed
    if not any(parts):
        try:
            import socket
            parts.append(socket.gethostname())
        except Exception:
            parts.append("unknown")

    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.digest()[:12]


def get_hwid_hex() -> str:
    """Hex HWID for display (so enterprise users can email it)."""
    return get_hwid().hex()


def check_node_lock(license_hwid: bytes) -> bool:
    """True if this machine's HWID matches the (HWID-bound) license."""
    if not license_hwid or license_hwid == b"\x00" * 12:
        return True   # not bound — floating license
    return get_hwid()[:len(license_hwid)] == license_hwid


def validate_license(key: str):
    """Full validation: signature + expiry + node-lock.

    Returns tier (1-4) on success, else None.  This is the single entry
    point the app should call for the new asymmetric keys.
    """
    parsed = parse_license_key(key)
    if parsed is None:
        return None
    tier, flags, hwid, expiry = parsed
    if flags & FLAG_EXPIRY_BOUND and is_expired(expiry):
        return None
    if flags & FLAG_HWID_BOUND and not check_node_lock(hwid):
        return None
    return tier


if __name__ == "__main__":
    print("=== Nightmare License Crypto (verify side) ===")
    print("Public key :", NIGHTMARE_PUBLIC_KEY_HEX)
    print("Backend    :", _init_backend() is not None)
    print("HWID       :", get_hwid_hex())
    # Smoke-test: a clearly invalid key must return None, not crash.
    print("bad key    :", validate_license("ND-AAAAAAAAAAAAAAAA"))
