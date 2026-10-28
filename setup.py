from setuptools import setup, find_packages

setup(
    name="microquant",
    version="1.0.0",
    description="Aetheris-MicroQuant: Quantitative Order Book Microstructure & High-Frequency Alpha Engine",
    author="Daniel Richard Xavier",
    author_email="drx2023l@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "microquant=microquant.cli:main",
        ],
    },
)
