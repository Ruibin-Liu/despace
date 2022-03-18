__package_name__ = "despace"
__version__ = "0.1.1"
__author__ = "Ruibin Liu"

# imports
# -------

import os

# config
# ------

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = 'Despace - A spatial awareness'

# requirements
# ------------

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().strip().split('\n')

setup(
    name=__package_name__,
    version=__version__,
    author=__author__,
    author_email="ruibinliuphd@gmail.com",
    description="A spatial decomposition tool for sorting or indexing N-D data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ruibin-Liu/despace",
    project_urls={
        "Bug Tracker": "https://github.com/Ruibin-Liu/despace/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.6",
)
