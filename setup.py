from setuptools import setup, find_packages

setup(
    name="svc_spectral_analysis",        # Renamed to reflect the broader scope
    version="1.0.0",
    description="Tools for reading and analysing SVC HR-1024 .sig files",
    author="",
    packages=find_packages(),            # Auto-discovers svc_reader, svc_analysis, etc.
    python_requires=">=3.8",
    install_requires=[
        # Shared dependencies
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        # New dependencies for svc_analysis
        "scikit-learn>=1.0",             # For PCA and other ML/stats analysis
        "matplotlib>=3.4",               # For plotting results
    ],
    entry_points={
        "console_scripts": [
            "svc-reader=svc_reader.__main__:main",
            "svc-analysis=svc_analysis.__main__:main", 
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)