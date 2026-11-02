#!/usr/bin/env python3
"""
Model Scanner Tool
==================
Quickly scan filesystem for GGUF models and their metadata.

Usage:
    python model_scanner.py [directory]
    python model_scanner.py --recursive --min-size 26000
    python model_scanner.py --json output.json
"""

import os
from pathlib import Path
import argparse
import json

# Try to import GGUFModelLoader for metadata discovery
try:
    from gguf_loader import GGUFModelLoader
    HAS_GGUF_LOADER = True
except ImportError:
    HAS_GGUF_LOADER = False


def find_gguf_files(directory, recursive=True):
    """Find all .gguf files in the given directory."""
    gguf_files = []
    search_path = Path(directory)
    
    if not search_path.exists():
        print(f"❌ Directory not found: {directory}")
        return gguf_files
    
    if recursive:
        for gguf_file in Path(directory).rglob("*.gguf"):
            gguf_files.append(gguf_file)
    else:
        for gguf_file in Path(directory).glob("*.gguf"):
            gguf_files.append(gguf_file)
    
    return sorted(gguf_files)


def get_model_info(filepath, use_loader=False):
    """Get metadata about a GGUF model."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    
    # Try to get metadata from GGUFModelLoader if available and requested
    if use_loader and HAS_GGUF_LOADER:
        try:
            loader = GGUFModelLoader()
            success = loader.load_model(filepath)
            if success:
                print(f"  ✅ {filepath.name} loaded via GGUFModelLoader")
                return {"path": str(filepath), "size_mb": size_mb, "loaded_via_loader": True}
        except Exception as e:
            pass
    
    # Fallback to direct file size check
    if size_mb >= 26000:
        label = "[LARGE] (>= 26B)"
    else:
        label = "[SMALL]"
    
    return {"path": str(filepath), "size_mb": round(size_mb, 1), "label": label}


def main():
    parser = argparse.ArgumentParser(description="Scan for GGUF models and their metadata")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to scan (default: current directory)")
    parser.add_argument("--recursive", action="store_true", default=True, 
                        help="Recursively search subdirectories (default: True)")
    parser.add_argument("--min-size", type=float, default=None,
                        help="Minimum model size in MB. Only show models >= this size.")
    parser.add_argument("--no-loader", action="store_true",
                        help="Skip GGUFModelLoader metadata discovery for speed")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output results as JSON to stdout")
    
    args = parser.parse_args()
    
    # Find GGUF files
    gguf_files = find_gguf_files(args.directory, recursive=args.recursive)
    
    if not gguf_files:
        print(f"No .gguf files found in {args.directory}")
        return
    
    print(f"Found {len(gguf_files)} GGUF model(s) in {args.directory}\n")
    
    results = []
    for filepath in gguf_files:
        info = get_model_info(filepath, use_loader=not args.no_loader)
        if info and (args.min_size is None or info["size_mb"] >= args.min_size):
            print(f"  {info['label']} {filepath.name} ({info['size_mb']:.1f} MB)")
            results.append(info)
    
    # Output JSON if requested
    if args.json:
        print("\n=== JSON Output ===")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
