"""
    Use the setup.py legacy version for package dependencies management.

    The problem is related to tensorflow package:
    when tensorflow-cpu or tensorflow-gpu are already installed, `pip` does not
    recognize them and forces a re-install of tensorflow, which might not be the
    best optimized package version for the current platform architecture

    Solution:
    all package metadata are handled statically by setup.cfg in PEP 517 style
    package dependencies are handled dinamically by this module.

"""
from setuptools import setup

requirements = [
    "awkward",
    "numpy",
    "jupyterlab",
    "matplotlib",
    "scipy",
    "sklearn",
    "pyyaml",
    "uproot",
]

try:
    import tensorflow
except ImportError:
    requirements.append("tensorflow")    


if __name__ == "__main__":
    setup(
        install_requires=requirements
    )