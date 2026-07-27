from setuptools import setup, find_packages
setup(
    name="svc_spectral_analysis",
    version="1.0.0",
    description="Tools for reading and analysing SVC HR-1024 .sig files",
    author="Nicola Falco",
    author_email= "nicolafalco@lbl.gov",
    packages=find_packages(),
    python_requires=">=3.8",
    package_data={'svc_analysis': ['config.yaml']}, 
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "seaborn>=0.12", 
        "pyyaml>=6.0",
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