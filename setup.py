from __future__ import print_function
import sys
from setuptools import setup, find_packages

requirements = [
    "numpy >= 1.18",
    "scipy >= 1.4",
    "matplotlib >= 3.2",
    "tensorflow >= 2.5.0",
    "pyyaml >= 6.0",
]

if sys.version_info < (3, 6):
    print("quake requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)

with open("README.md") as f:
    long_desc = f.read()

setup(
    name="quake",
    version="1.0.0",
    description="Quantum kernel classifier for neutrino physics applications",
    author=None,
    author_email=None,
    url="https://github.com/qismib/QuaKe.git",
    long_description=long_desc,
    entry_points={"console_scripts": ["quake = quake.scripts.quake:main"]},
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
# TODO: change package author name and email