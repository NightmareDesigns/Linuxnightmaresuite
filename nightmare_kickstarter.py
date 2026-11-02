#!/usr/bin/env python3
"""
Nightmare AI Loader — KICKSTARTER EDITION
==========================================
Standalone program for the Kickstarter campaign.
Kickstarter pledges ONLY — no CashApp, no ntmre.com direct purchase, no
direct payment buttons of any kind. All payments go through Kickstarter's
own secure system while the campaign is live.

Build to exe:
    pyinstaller --onefile --name NightmareKickstarter nightmare_kickstarter.py

Author: Nightmare Designs
"""
import nightmare_suite_unified as ns

# Lock this build to Kickstarter-pledge mode (no direct payments).
ns.PRICING_MODE = "kickstarter"

if __name__ == "__main__":
    ns.main()
