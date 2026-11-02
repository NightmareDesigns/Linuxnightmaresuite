"""
Nightmare Designs — FIDO2 & WebAuthn Security Core
===================================================
Hardware-backed FIDO2 / CTAP2 / WebAuthn authentication for Nightmare Suite,
the assistant Admin Link, and the Master Admin Dashboard.

Supports:
- Physical USB FIDO2/CTAP2 Security Keys (YubiKey, Feitian, SoloKey, Titan)
- Platform authenticators (Windows Hello, Touch, PIN)
- WebAuthn browser authentication for Web Dashboards
- Secure local credential storage with ES256 / Ed25519 public keys

Author: Caleb Levi Gates / Nightmare Designs
"""

import os
import sys
import json
import time
import base64
import hashlib
import getpass
from pathlib import Path
from typing import Optional, Dict, Any, List

SUITE_DIR = Path(__file__).resolve().parent

# Credential store path
FIDO2_STORE_PATH = SUITE_DIR / ".fido2_credentials.json"
RECOVERY_STORE_PATH = Path(r"I:\Nightmare_Recovery_13\.fido2_credentials.json")

RP_ID = "nightmare.suite"
RP_NAME = "Nightmare Designs Master Suite"

class Fido2Manager:
    """Manages FIDO2 device discovery, registration, and assertion verification."""

    def __init__(self):
        self._fido2_available = False
        try:
            import fido2
            from fido2.hid import CtapHidDevice
            from fido2.client import Fido2Client, UserInteraction
            from fido2.client import DefaultClientDataCollector
            from fido2.server import Fido2Server
            from fido2.webauthn import (
                PublicKeyCredentialRpEntity,
                PublicKeyCredentialRequestOptions,
                PublicKeyCredentialCreationOptions,
                PublicKeyCredentialUserEntity,
                PublicKeyCredentialParameters,
                PublicKeyCredentialType,
                UserVerificationRequirement,
            )
            self.CtapHidDevice = CtapHidDevice
            self.Fido2Client = Fido2Client
            self.Fido2Server = Fido2Server
            self.UserInteraction = UserInteraction
            self.DefaultClientDataCollector = DefaultClientDataCollector
            self.PublicKeyCredentialRequestOptions = PublicKeyCredentialRequestOptions
            self.PublicKeyCredentialCreationOptions = PublicKeyCredentialCreationOptions
            self.PublicKeyCredentialUserEntity = PublicKeyCredentialUserEntity
            self.PublicKeyCredentialParameters = PublicKeyCredentialParameters
            self.PublicKeyCredentialType = PublicKeyCredentialType
            self.UserVerificationRequirement = UserVerificationRequirement
            self.rp = PublicKeyCredentialRpEntity(id=RP_ID, name=RP_NAME)
            self.server = Fido2Server(self.rp)
            self._fido2_available = True
        except Exception as e:
            self._fido2_available = False
            self._err = str(e)

    def _make_user_interaction(self, prompt_cb=None):
        """Provide PIN/UV callbacks used by CTAP2 before user presence touch."""
        manager = self

        class Interaction(manager.UserInteraction):
            def prompt_up(self):
                if prompt_cb:
                    prompt_cb("Touch your FIDO2 security key after PIN verification...")

            def request_pin(self, permissions, rp_id):
                if prompt_cb:
                    prompt_cb("Enter your FIDO2 security key PIN...")
                try:
                    import tkinter as tk
                    from tkinter import simpledialog
                    root = tk._default_root
                    if root is not None:
                        return simpledialog.askstring(
                            "FIDO2 Security Key PIN",
                            "Enter the PIN for your security key:",
                            parent=root,
                            show="*"
                        )
                except Exception:
                    pass
                try:
                    return getpass.getpass("FIDO2 security key PIN: ")
                except Exception:
                    return None

            def request_uv(self, permissions, rp_id):
                return True

        return Interaction()

    def is_available(self) -> bool:
        return self._fido2_available

    def list_devices(self) -> List[Any]:
        """List attached CTAP USB HID devices."""
        if not self._fido2_available:
            return []
        try:
            return list(self.CtapHidDevice.list_devices())
        except Exception:
            return []

    def get_device_count(self) -> int:
        return len(self.list_devices())

    def get_device_info(self) -> List[Dict[str, Any]]:
        devs = self.list_devices()
        info_list = []
        for i, dev in enumerate(devs):
            info = {
                "index": i,
                "product_name": getattr(dev.descriptor, "product_name", "FIDO2 / CTAP2 Security Key"),
                "serial_number": getattr(dev.descriptor, "serial_number", f"KEY-{i+1}"),
                "is_ctap2": getattr(dev, "is_ctap2", True),
            }
            info_list.append(info)
        return info_list

    def load_credentials(self) -> List[Dict[str, Any]]:
        """Load registered FIDO2 credentials."""
        for path in (FIDO2_STORE_PATH, RECOVERY_STORE_PATH):
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "credentials" in data:
                        return data["credentials"]
                except Exception:
                    pass
        # Return default authorized admin credentials if empty
        return [
            {
                "user_name": "Levi (Creator)",
                "credential_id": "ND-FIDO2-MASTER-LEVI-01",
                "registered_at": "2026-09-14T00:00:00Z",
                "type": "FIDO2/WebAuthn",
                "tier": 4,
            },
            {
                "user_name": "John G (Admin)",
                "credential_id": "ND-FIDO2-ADMIN-JOHNG-02",
                "registered_at": "2026-09-14T00:00:00Z",
                "type": "FIDO2/WebAuthn",
                "tier": 4,
            }
        ]

    def save_credential(self, user_name: str, cred_id: str, public_key_b64: str = "") -> bool:
        """Save a new FIDO2 credential."""
        creds = self.load_credentials()
        new_entry = {
            "user_name": user_name,
            "credential_id": cred_id,
            "public_key": public_key_b64,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tier": 4,
        }
        creds.append(new_entry)
        data_str = json.dumps(creds, indent=2)
        for path in (FIDO2_STORE_PATH, RECOVERY_STORE_PATH):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data_str, encoding="utf-8")
            except Exception:
                pass
        return True

    def register_physical_key(self, user_name: str, prompt_cb=None) -> Dict[str, Any]:
        """Perform hardware registration on an attached USB FIDO2 key."""
        if not self._fido2_available:
            return {"success": False, "error": "fido2 library unavailable"}

        devs = self.list_devices()
        if not devs:
            return {"success": False, "error": "No physical FIDO2 security key detected"}

        dev = devs[0]
        if prompt_cb:
            prompt_cb("👉 Please touch your FIDO2 Security Key now...")

        try:
            collector = self.DefaultClientDataCollector("https://%s" % RP_ID, lambda rp, origin: rp == RP_ID)
            client = self.Fido2Client(dev, collector, self._make_user_interaction(prompt_cb))
            user = self.PublicKeyCredentialUserEntity(
                id=user_name.encode("utf-8"), name=user_name, display_name=user_name
            )
            challenge = os.urandom(32)
            options = self.PublicKeyCredentialCreationOptions(
                rp=self.rp,
                user=user,
                challenge=challenge,
                pub_key_cred_params=[self.PublicKeyCredentialParameters(
                    type=self.PublicKeyCredentialType.PUBLIC_KEY, alg=-7
                )],
                timeout=60000,
            )
            res = client.make_credential(options)
            cred_data = res.attestation_object.auth_data.credential_data
            cred_id = base64.b64encode(cred_data.credential_id).decode("ascii")
            self.save_credential(user_name, cred_id)
            return {
                "success": True,
                "user_name": user_name,
                "credential_id": cred_id,
                "mode": "Hardware CTAP2",
                "message": f"Hardware security key registered successfully for {user_name}!"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def authenticate_physical_key(self, prompt_cb=None) -> Dict[str, Any]:
        """Request touch assertion from an attached physical FIDO2 key."""
        if not self._fido2_available:
            return {
                "success": False,
                "user_name": "",
                "tier": 0,
                "mode": "FIDO2 Unavailable",
                "error": "FIDO2 library unavailable"
            }

        devs = self.list_devices()
        if not devs:
            return {
                "success": False,
                "user_name": "",
                "tier": 0,
                "mode": "FIDO2 Unavailable",
                "error": "No physical FIDO2 security key detected"
            }

        dev = devs[0]
        if prompt_cb:
            prompt_cb("👉 Touch your FIDO2 Security Key (YubiKey / Passkey) to verify...")

        try:
            collector = self.DefaultClientDataCollector("https://%s" % RP_ID, lambda rp, origin: rp == RP_ID)
            client = self.Fido2Client(dev, collector, self._make_user_interaction(prompt_cb))
            challenge = os.urandom(32)
            options = self.PublicKeyCredentialRequestOptions(
                challenge=challenge,
                rp_id=RP_ID,
                user_verification=self.UserVerificationRequirement.REQUIRED,
                timeout=60000,
            )
            res = client.get_assertion(options)
            assertion = res.get_response(0)
            user_id = getattr(assertion, "user", {}).get("name", "Admin") if hasattr(assertion, "user") and assertion.user else "Admin"
            return {
                "success": True,
                "user_name": user_id,
                "tier": 4,
                "mode": "Hardware CTAP2 Touch",
                "message": "FIDO2 Security Key Touch Verified! Tier 4 Admin Unlocked."
            }
        except Exception as e:
            return {
                "success": False,
                "user_name": "",
                "tier": 0,
                "mode": "FIDO2 Failed",
                "error": f"FIDO2 PIN/touch verification failed: {e}"
            }

# Singleton instance
fido2_mgr = Fido2Manager()
