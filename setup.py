#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI Setup Script

This script is used to install the HarborAI package.
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Extract version from harborai/__init__.py
version = {}
with open(os.path.join(this_directory, "harborai", "__init__.py"), encoding="utf-8") as f:
    exec(f.read(), version)

setup(
    name="harborai",
    version=version["__version__"],
    author="HarborAI Team",
    author_email="team@harborai.dev",
    description="A unified LLM client with OpenAI-compatible interface and advanced features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harborai/harborai",
    project_urls={
        "Bug Reports": "https://github.com/harborai/harborai/issues",
        "Source": "https://github.com/harborai/harborai",
        "Documentation": "https://docs.harborai.dev",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "database": [
            "asyncpg>=0.28.0",
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "asyncpg>=0.28.0",
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "harborai=harborai.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "harborai": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "llm",
        "openai",
        "ai",
        "machine-learning",
        "artificial-intelligence",
        "chatgpt",
        "gpt",
        "api",
        "client",
        "sdk",
        "deepseek",
        "doubao",
        "wenxin",
        "reasoning",
        "structured-output",
    ],
)