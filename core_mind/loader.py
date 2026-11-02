"""
NIGHTMARE SEALED MODULE LOADER — GhostRider Hardened
====================================================
Imports .sealed files at runtime by decrypting them in memory.
Source code never touches disk as plaintext.

Handles top-level modules, subpackages (engine.bonsai_pc_app),
and packages (core_mind) with dotted names. Works both in dev
mode (.py present → normal import) and frozen exe mode
(.py absent → load from .sealed).

Usage:
    from core_mind.loader import install_seal_hook
    install_seal_hook()
    # Now 'import transformer' auto-loads from .sealed if .py is missing

Author: Nightmare Designs
"""

import sys
import os
import types
import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path

# Resolve the application directory — works in dev and frozen (exe) mode
if getattr(sys, 'frozen', False):
    # PyInstaller: sys._MEIPASS points to the data root in both onedir and onefile
    if hasattr(sys, '_MEIPASS'):
        APP_DIR = Path(sys._MEIPASS)
    else:
        APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).resolve().parent.parent

# Import the sealer's crypto functions (must stay plaintext)
from core_mind.seal_code import (
    MASTER_SECRET, _derive_key, unseal_file
)

# Cache the derived key (computed once on first import — expensive)
_KEY_CACHE = None


def _get_key():
    """Return the AES-equivalent key, deriving it once via GhostRider/SHA chain."""
    global _KEY_CACHE
    if _KEY_CACHE is None:
        _KEY_CACHE = _derive_key(MASTER_SECRET)
    return _KEY_CACHE


def _sealed_to_py_path(sealed_path: str) -> str:
    """Convert a .sealed path to its .py equivalent for __file__ compatibility."""
    if sealed_path.endswith('.sealed'):
        return sealed_path[:-7] + '.py'  # replace ".sealed" with ".py"
    return sealed_path


class SealedLoader(importlib.abc.Loader):
    """Custom loader that decrypts .sealed files in memory and execs them."""

    def create_module(self, spec):
        return None  # Use default module creation

    def exec_module(self, module):
        spec = module.__spec__
        sealed_path = Path(spec.origin)

        key = _get_key()
        try:
            source_bytes = unseal_file(sealed_path, key)
            source_code = source_bytes.decode('utf-8')
        except Exception as e:
            raise ImportError(
                f"[SHIELD] Failed to unseal {module.__name__}: {e}\n"
                f"Wrong key or corrupted .sealed file."
            )

        # Set __file__ to the .py equivalent so code that does
        # Path(__file__).parent or __file__.replace('.py', ...) still works
        py_path = _sealed_to_py_path(str(sealed_path))
        module.__file__ = py_path
        module.__loader__ = self
        module.__sealed__ = True

        code = compile(source_code, py_path, 'exec')
        exec(code, module.__dict__)


class SealedFinder(importlib.abc.MetaPathFinder):
    """
    Import hook that finds .sealed files when .py is missing.
    Install with: install_seal_hook()

    Handles:
      - top-level modules:  transformer → transformer.sealed
      - subpackages:        engine.bonsai_pc_app → engine/bonsai_pc_app.sealed
      - packages:          core_mind → core_mind/__init__.sealed
    """

    def find_spec(self, fullname, path, target=None):
        parts = fullname.split('.')
        leaf = parts[-1]

        # Search dirs: parent package __path__ first, then APP_DIR
        search_dirs = []
        if path:
            search_dirs.extend(str(p) for p in path)
        search_dirs.append(str(APP_DIR))

        for sd in search_dirs:
            base = Path(sd)

            # Try as a module: {leaf}.sealed
            sealed_path = base / (leaf + '.sealed')
            py_path = base / (leaf + '.py')
            if sealed_path.exists() and not py_path.exists():
                return importlib.machinery.ModuleSpec(
                    fullname, SealedLoader(), origin=str(sealed_path)
                )

            # Try as a package: {leaf}/__init__.sealed
            pkg_sealed = base / leaf / '__init__.sealed'
            pkg_py = base / leaf / '__init__.py'
            pkg_dir = base / leaf
            if pkg_sealed.exists() and not pkg_py.exists():
                spec = importlib.machinery.ModuleSpec(
                    fullname, SealedLoader(),
                    origin=str(pkg_sealed), is_package=True
                )
                # Set search path as attribute (compatible with all Python/ModuleSpec variants)
                spec.submodule_search_locations = [str(pkg_dir)]
                return spec

        return None


def install_seal_hook():
    """
    Install the sealed import hook at the front of sys.meta_path.
    After this, any 'import module_name' will auto-load from .sealed
    if the .py file is missing (sealed/locked mode).
    """
    for finder in sys.meta_path:
        if isinstance(finder, SealedFinder):
            return  # Already installed

    sys.meta_path.insert(0, SealedFinder())
    print("[SHIELD] GhostRider sealed import hook active.")


def sealed_import(module_name: str, package_dir: Path = None):
    """
    Explicitly import a sealed module by name (alternative to the hook).
    Looks for {module_name}.sealed, decrypts in memory, returns the module.
    Falls back to normal import if .sealed doesn't exist (dev mode).
    """
    if package_dir is None:
        package_dir = APP_DIR

    sealed_path = package_dir / f"{module_name}.sealed"

    if not sealed_path.exists():
        return importlib.import_module(module_name)

    if module_name in sys.modules:
        return sys.modules[module_name]

    key = _get_key()
    try:
        source_bytes = unseal_file(sealed_path, key)
        source_code = source_bytes.decode('utf-8')
    except Exception as e:
        raise ImportError(f"[SHIELD] Failed to unseal {module_name}: {e}")

    module = types.ModuleType(module_name)
    py_path = str(package_dir / f"{module_name}.py")
    module.__file__ = py_path
    module.__loader__ = SealedLoader()
    module.__sealed__ = True
    module.__dict__['__name__'] = module_name
    module.__dict__['__builtins__'] = __builtins__

    sys.modules[module_name] = module
    try:
        exec(compile(source_code, py_path, 'exec'), module.__dict__)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"[SHIELD] Error executing sealed {module_name}: {e}")

    return module


def verify_seal_integrity():
    """
    Check that all .sealed files can be decrypted and compiled successfully.
    Returns list of (filename, ok, error) tuples.
    """
    from core_mind.seal_code import CRITICAL_FILES

    key = _get_key()
    results = []

    for filename in CRITICAL_FILES:
        sealed_path = APP_DIR / filename.replace('.py', '.sealed')
        if not sealed_path.exists():
            results.append((filename, None, "not sealed"))
            continue

        try:
            source = unseal_file(sealed_path, key)
            compile(source.decode('utf-8'), filename, 'exec')
            results.append((filename, True, "OK"))
        except Exception as e:
            results.append((filename, False, str(e)))

    return results
