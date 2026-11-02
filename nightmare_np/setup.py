"""nightmare_np — setup for optional pip install (not required).

You do NOT need pip at all. `nightmare_np` is pure stdlib + a hand-written C
DLL. To use it, either keep this folder next to your code and `import
nightmare_np`, or run install.bat. This setup.py exists only so that `pip
install -e .` also works if you ever want it — it never touches numpy.
"""
from setuptools import setup, find_packages

setup(
    name="nightmare_np",
    version="1.0.0",
    description="Hand-written numpy replacement for the Nightmare Suite (no numpy, stdlib only)",
    packages=find_packages(),
    include_package_data=True,
    package_data={"nightmare_np": ["nightmare_np_c.dll"]},
    python_requires=">=3.8",
)
