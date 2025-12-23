from setuptools import setup, find_packages

setup(
    name="mpear",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "astropy",
        "spectres",
    ],
    extras_require={
        "dev": [
            "matplotlib",
        ]
    },
    author="Eleonora Alei et al.",
    author_email="eleonora.alei@nasa.gov",
    description="Multi-bandpass Photometry for Exoplanet Atmosphere Reconnaissance (MPEAR) with the Habitable Worlds Observatory (HWO)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eleonoraalei/mpear",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.6",
)
