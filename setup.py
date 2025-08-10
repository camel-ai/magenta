"""
Setup script for Math Reasoning Library
"""

from setuptools import setup, find_packages

with open("math_reasoning_lib/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="math_reasoning_lib",
    version="0.1.0",
    author="Math Reasoning Team",
    author_email="team@mathlib.com",
    description="统一的数学推理库，支持不同benchmark的端到端处理",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "dataclasses;python_version<'3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-reasoning=math_reasoning_lib.examples.basic_usage:main",
        ],
    },
) 