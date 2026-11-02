"""Build script for the native gr_hash GhostRider hash extension.

Usage:
    python setup_gr_hash.py build_ext --inplace

Produces gr_hash*.pyd (Windows) or gr_hash*.so (Linux) in the current directory.
"""

from setuptools import setup, Extension

module = Extension(
    'gr_hash',
    sources=['gr_hash.c'],
    extra_compile_args=['/O2'] if __import__('platform').system() == 'Windows' else ['-O2'],
)

setup(
    name='gr_hash',
    version='1.0',
    description='Nightmare GhostRider Hash — native proof-of-work hash',
    ext_modules=[module],
    zip_safe=False,
)
