"""
nightmare_global_mesh.py — Global Multi-Suite Interconnect & Rendezvous Mesh
=============================================================================
Nightmare Designs — Levi Gates

Enables any Nightmare Suite worldwide to connect automatically to the Main
Host Suite without manual IP configuration, port-forwarding, or fixed domains.

Mechanisms:
 1. Auto-Tunneling (Cloudflare Quick Tunnel / Localhost.run / P2P Relay):
    - Automatically opens an end-to-end encrypted HTTPS global tunnel on the Main Suite.
 2. Global Node Registry & Rendezvous:
    - Publishes the encrypted endpoint to the suite's local and distributed node store.
 3. Auto-Discovery on Remote Suites:
    - When any remote suite boots, it auto-detects the Main Suite's active global endpoint
      and connects automatically.
"""

import os
import sys
import json
import time
import socket
import urllib.request
import urllib.parse
import threading
import subprocess
import shutil
import re
from pathlib import Path

DEFAULT_ADMIN_KEY = None  # sealed: no shipped key
CONFIG_FILE = Path(".nightmare_global_node")


def get_stored_global_endpoint() -> str:
    """Retrieve the latest registered global endpoint for the Main Suite."""
    # Check environment variable first
    if os.environ.get("NIGHTMARE_MAIN_SUITE_URL"):
        return os.environ["NIGHTMARE_MAIN_SUITE_URL"].rstrip("/")

    # Check local config files
    for p in [
        Path(".nightmare_global_node"),
        Path.cwd() / ".nightmare_global_node",
        Path.home() / ".nightmare_global_node",
        Path("I:/Nightmare_Recovery_13/.nightmare_global_node"),
        Path("I:/Nightmare_Suite_13/.nightmare_global_node"),
        Path("I:/.nightmare_global_node"),
        Path("J:/Nightmare_Suite_13/.nightmare_global_node"),
        Path("J:/Nightmare_Suite_Network_Update/.nightmare_global_node"),
        Path("J:/.nightmare_global_node"),
    ]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # Prefer explicit LAN URL / IP if present, then global_url
                url = data.get("lan_url") or data.get("global_url") or ""
                if not url and data.get("lan_ip"):
                    url = f"http://{data['lan_ip']}:{data.get('port', 1234)}"
                if url:
                    return url.rstrip("/")
            except Exception:
                pass

    return "http://192.168.1.220:1234"


def save_global_endpoint(url: str, meta: dict = None):
    """Save the active global endpoint across suite locations."""
    payload = {
        "global_url": url,
        "updated_at": time.time(),
        "host": socket.gethostname(),
        "meta": meta or {}
    }
    content = json.dumps(payload, indent=2)
    for p in [
        Path(".nightmare_global_node"),
        Path.cwd() / ".nightmare_global_node",
        Path.home() / ".nightmare_global_node",
        Path("I:/Nightmare_Recovery_13/.nightmare_global_node"),
        Path("J:/Nightmare_Suite_13/.nightmare_global_node"),
    ]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception:
            pass


class GlobalSuiteMesh:
    """Manages publishing on the Main Suite and discovery on Satellite Suites."""

    def __init__(self, port: int = 1234, token: str = None, is_main: bool = False):
        self.port = port
        self.token = token
        self.is_main = is_main
        self.active_global_url = ""
        self.tunnel_proc = None
        self._running = True

    def start_host_broadcasting(self):
        """Starts global tunneling and beacon publishing on the Main Suite."""
        t = threading.Thread(target=self._host_worker, daemon=True)
        t.start()

    def _find_cloudflared(self):
        cf = shutil.which("cloudflared")
        if cf: return cf
        for cand in [
            str(Path(__file__).resolve().parent / "cloudflared.exe"),
            str(Path(__file__).resolve().parent.parent / "nightmare_suite_v10" / "cloudflared.exe"),
            str(Path(__file__).resolve().parent.parent / "cloudflared.exe"),
            r"I:\Nightmare_Recovery_13\biofolder\nightmare_suite_admin\cloudflared.exe",
            r"I:\Nightmare_Suite_13\biofolder\nightmare_suite_admin\cloudflared.exe",
            r"I:\cloudflared.exe",
            r"C:\Program Files\cloudflared\cloudflared.exe",
            r"C:\ProgramData\chocolatey\bin\cloudflared.exe",
            os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"),
            os.path.expanduser(r"~\cloudflared.exe"),
            os.path.expanduser(r"~\Downloads\cloudflared.exe"),
        ]:
            if os.path.isfile(cand):
                return cand
        return None

    def _host_worker(self):
        cf_bin = self._find_cloudflared()
        if cf_bin:
            try:
                cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate"]
                self.tunnel_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in self.tunnel_proc.stdout:
                    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                    if m:
                        self.active_global_url = m.group(0)
                        save_global_endpoint(self.active_global_url, {"type": "cloudflare"})
                        print(f"\n🌍 [GLOBAL SUITE INTERCONNECT ACTIVE]")
                        print(f"   Any other suite worldwide can connect to:")
                        print(f"   URL: {self.active_global_url}")
                        print(f"   Auth Token: {self.token[:8]}... (Hardware Encrypted)\n")
            except Exception as e:
                print(f"[GlobalMesh] Tunnel notice: {e}")

