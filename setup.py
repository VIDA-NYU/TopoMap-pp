from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="topomap",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.8.2",
        "mlpack>=4.3.0.post1",
        "networkx>=3.2.1",
        "numpy>=1.25.0",
        "pandas>=2.2.2",
        "plotly>=5.18.0",
        "scikit_learn>=1.4.1.post1",
        "scipy>=1.13.1",
        "diskannpy==0.7.0"
    ],
    author="Vitoria Guardieiro, Felipe Inagaki de Oliveira, Harish Doraiswamy, Luis Gustavo Nonato, Claudio Silva",
    author_email="vitoriaguardieiro@gmail.com",
    description="TopoMap++: A faster and more space efficient technique to compute projections with topological guarantees",
    long_description=long_description,
    keywords=[
        "Topological data analysis", 
        "Computational topology", 
        "High-dimensional data", 
        "Projection"
    ],
    url="https://github.com/viguardieiro/TopoMap",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)