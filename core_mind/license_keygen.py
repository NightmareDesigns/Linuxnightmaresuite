"""
license_keygen.py — OFFLINE developer tool to sign Nightmare licenses
====================================================================
NOT DISTRIBUTED.  Run this on your offline machine with the private key.
The shipped app contains only the public key (in license_crypto.py) and
cannot generate keys.

Key format produced:
    ND-<base32 groups>

    payload (18 bytes): tier(1) flags(1) hwid(12) expiry-day(4)
    signature (64 bytes): Ed25519 over the payload

Tier policy (matches the Kickstarter / pricing spec):
    T1 Explorer   ($199)      — floating (no HWID), bulk-distributable
    T2 Builder    ($599)      — floating (no HWID), bulk-distributable
    T3 Creator    ($5,999/yr)  — HWID-bound (node-locked)
    T4 Enterprise ($50,000+/yr)— HWID-bound (node-locked)

Requirements:
    pip install cryptography
    Keep nightmare_private.pem OFF the distributed build.

Usage:
    python license_keygen.py --tier 1
    python license_keygen.py --tier 2 --count 1000 > t2_keys.txt
    python license_keygen.py --tier 3 --hwid cf180aa1e089c891dd2cf495
    python license_keygen.py --tier 4 --hwid cf180aa1e089c891dd2cf495 --days 365
    python license_keygen.py --show-hwid        # print this machine's HWID
"""

import argparse
import base64
import struct
import sys
import time

MSG_FMT = ">BB12sI"
MSG_SIZE = struct.calcsize(MSG_FMT)        # 18
FLAG_HWID_BOUND = 0x01
FLAG_EXPIRY_BOUND = 0x02


def _load_private_key(path: str):
    """Load an Ed25519 private key from a PEM file.

    Accepts a standard PKCS#8 PEM (from `cryptography`) OR a non-standard
    PEM whose base64 body is the raw 32-byte Ed25519 seed.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        load_pem_private_key, Encoding, PrivateFormat, NoEncryption)

    raw_pem = open(path, "rb").read()
    # 1) standard PKCS#8 PEM
    try:
        key = load_pem_private_key(raw_pem, password=None)
        if hasattr(key, "sign"):
            return key
    except Exception:
        pass
    # 2) raw 32-byte seed wrapped in a PEM body
    try:
        body = b""
        for line in raw_pem.splitlines():
            line = line.strip()
            if line.startswith(b"-----") or not line:
                continue
            body += line
        seed = base64.b64decode(body)
        if len(seed) == 32:
            return Ed25519PrivateKey.from_private_bytes(seed)
    except Exception:
        pass
    raise ValueError(
        f"Could not load an Ed25519 private key from {path}. "
        "Provide a PKCS#8 PEM or a PEM whose body is the raw 32-byte seed.")


def _b32_grouped(blob: bytes) -> str:
    b32 = base64.b32encode(blob).decode().rstrip("=")
    return "-".join(b32[i:i + 8] for i in range(0, len(b32), 8))


def generate_key(priv_key, tier: int, hwid_hex: str = None,
                 days: int = None) -> str:
    """Sign and return one ND-... license key string."""
    tier = int(tier)
    if tier < 1 or tier > 4:
        raise ValueError("tier must be 1-4")

    # HWID binding policy: T3/T4 are node-locked by default.
    hwid = b"\x00" * 12
    flags = 0
    if hwid_hex:
        hwid = bytes.fromhex(hwid_hex)
        if len(hwid) != 12:
            raise ValueError("hwid must be 12 bytes (24 hex chars)")
        flags |= FLAG_HWID_BOUND
    elif tier >= 3:
        raise ValueError(
            f"T{tier} licenses must be HWID-bound. Pass --hwid (the "
            "customer's HWID, shown by the app's license dialog).")

    expiry_day = 0
    if days:
        expiry_day = int(time.time() // 86400) + int(days)
        flags |= FLAG_EXPIRY_BOUND

    message = struct.pack(MSG_FMT, tier, flags, hwid, expiry_day)
    signature = priv_key.sign(message)
    return "ND-" + _b32_grouped(message + signature)


def main():
    ap = argparse.ArgumentParser(description="Nightmare offline license keygen")
    ap.add_argument("--keyfile", default="nightmare_private.pem",
                    help="Path to the Ed25519 private key PEM")
    ap.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                    help="License tier (1-4)")
    ap.add_argument("--hwid", help="12-byte hex HWID to node-lock (T3/T4)")
    ap.add_argument("--days", type=int, help="License validity in days (expiry)")
    ap.add_argument("--count", type=int, default=1,
                    help="Number of keys to generate (for T1/T2 bulk runs)")
    ap.add_argument("--show-hwid", action="store_true",
                    help="Print this machine's HWID and exit")
    args = ap.parse_args()

    if args.show_hwid:
        from license_crypto import get_hwid_hex
        print(get_hwid_hex())
        return

    if args.tier is None:
        ap.error("--tier is required (unless --show-hwid)")

    priv = _load_private_key(args.keyfile)
    for _ in range(max(1, args.count)):
        print(generate_key(priv, args.tier, args.hwid, args.days))


if __name__ == "__main__":
    main()
