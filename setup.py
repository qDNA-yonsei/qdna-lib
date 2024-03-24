"""
pypi setup
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qdna-lib",
    version="0.0.3",
    author="qdna team",
    author_email="ifa@yonsei.ac.kr",
    description="A quantum computing library for data science and artificial intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qDNA-yonsei/qdna-lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'qiskit>=1.0.0',
        'qiskit-machine-learning>=0.7.2',
        'qiskit-aer>=0.13.3',
        'qiskit-algorithms>=0.3.0',
        'qclib>=0.1.12',
        'dwave-ocean-sdk>=6.9.0',
        'torch',
        'networkx'
    ]
)
