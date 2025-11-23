"""
SchwabGym Package Configuration
================================

High-fidelity reinforcement learning environment for algorithmic trading.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import os

from setuptools import find_packages, setup

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="schwabgym",
    version="1.0.0",
    author="Bryant Clark",
    author_email="bryant.clark@example.com",
    description="High-fidelity RL environment for training algorithmic trading agents on the Charles Schwab platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bryantclark/SchwabGym",
    project_urls={
        "Bug Tracker": "https://github.com/bryantclark/SchwabGym/issues",
        "Documentation": "https://schwabgym.readthedocs.io",
        "Source Code": "https://github.com/bryantclark/SchwabGym",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "rl": [
            "stable-baselines3>=2.0.0",
            "torch>=2.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "schwabgym-demo=schwabgym.cli:demo",
            "schwabgym-validate=schwabgym.cli:validate_data",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "algorithmic trading",
        "reinforcement learning",
        "schwab api",
        "trading simulator",
        "market microstructure",
        "gymnasium",
        "deep learning",
        "quantitative finance",
    ],
)
