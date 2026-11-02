#!/usr/bin/env python3
"""
Nightmare AI Loader — DIRECT SALES EDITION
============================================
Standalone program for direct sales (ntmre.com + CashApp $ngtmre1).
No Kickstarter UI — no pledge buttons, no campaign text.

Build to exe:
    pyinstaller --onefile --name NightmareDirect nightmare_direct.py

Author: Nightmare Designs
"""
import nightmare_suite_unified as ns

# Lock this build to direct-sales mode (CashApp + ntmre.com).
ns.PRICING_MODE = "direct"

if __name__ == "__main__":
    ns.main()
