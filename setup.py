"""Setup the package."""
from setuptools import setup, find_packages

# Version number

import re
VERSIONFILE = "flexconv/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")


with open("README.md", "r") as fh:
    long_description = fh.read()

# Alle requirements
requirements = [
    "fft-conv-pytorch"
]

setup(
    name="flexconv",
    version=verstr,
    author="J. Gerbscheid",
    author_email="jonathan.gerbscheid@student.uva.nl",
    description="Pytorch FlexConv implementation + modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonathan-gerb/flexconv",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: All rights reserved",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha'
    ],
    python_requires=">=3.8",
    # zip_safe=False
)
