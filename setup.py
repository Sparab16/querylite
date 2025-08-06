"""
Setup script for the querylite package.
"""
from setuptools import setup, find_packages

setup(
    name="querylite",
    version="1.0.0",
    description="A lightweight columnar storage engine with SQL query support",
    author="Shreyas Parab",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "lark-parser"
    ],
    entry_points={
        "console_scripts": [
            "querylite=querylite.cli:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
