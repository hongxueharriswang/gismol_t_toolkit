from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gismol",
    version="1.0.0",
    author="Harris Wang",
    author_email="harrisw@athabascau.ca",
    description="General Intelligent System Modelling Language – Python implementation of Constrained Object Hierarchies (COH)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gismol",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "neural": [
            "torch>=1.9.0",
        ],
        "nlp": [
            "sentence-transformers>=2.0",
        ],
        "all": [
            "torch>=1.9.0",
            "sentence-transformers>=2.0",
        ],
    },
)