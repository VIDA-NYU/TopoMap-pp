from setuptools import setup, find_packages
from pathlib import Path

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
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords=[
        "Topological data analysis", 
        "Computational topology", 
        "High-dimensional data", 
        "Projection"
    ],
    url="https://github.com/viguardieiro/TopoMap",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
