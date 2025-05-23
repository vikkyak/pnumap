import platform
from setuptools import setup


def readme():
    try:
        with open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()


configuration = {
    "name": "pnumap",
    "version": "0.5.8",
    "description": "Uniform Manifold Approximation and Projection",
    "long_description": readme(),
    "long_description_content_type": "text/x-rst",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    "keywords": "dimension reduction t-sne manifold",
    "url": "http://github.com/lmcinnes/umap",
    "maintainer": "Leland McInnes",
    "maintainer_email": "leland.mcinnes@gmail.com",
    "license": "BSD",
    "packages": ["pnumap"],
    "install_requires": [
        "numpy >= 1.23",
        "scipy >= 1.3.1",
        #"scikit-learn >= 1.6",
        "scikit-learn >= 1.1, <1.6",
        "numba >= 0.51.2",
        "pynndescent >= 0.5",
        "tqdm",
    ],
    "extras_require": {
        "plot": [
            "pandas",
            "matplotlib",
            "datashader",
            "bokeh",
            "holoviews",
            "colorcet",
            "seaborn",
            "scikit-image",
            "dask",
        ],
        "parametric_umap": ["tensorflow >= 2.1"],
        "tbb": ["tbb >= 2019.0"],
    },
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
