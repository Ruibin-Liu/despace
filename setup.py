__package_name__ = "despace"
__version__ = "0.1.0"
__author__ = "Ruibin Liu"

import setuptools


def get_long_description() -> str:
    """
    Get the long description as written in the project README.md file.
    """
    with open("README.md", "r", encoding="utf-8") as rf:
        return rf.read()


requires = [
    'numpy',
    'matplotlib',
]


setuptools.setup(
    name=__package_name__,
    version=__version__,
    author=__author__,
    author_email="ruibinliuphd@gmail.com",
    description="A spatial decomposition tool for sorting or indexing N-D data.",
    long_description=get_long_description(),
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
    package_dir={"": "despace"},
    packages=setuptools.find_packages(where="despace"),
    install_requires=requires,
    python_requires=">=3.6",
)
