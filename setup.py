"""
Setup script for Quantum Hamiltonian Simulation Framework.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    requirements = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name='quantum-hamiltonian-simulation',
    version='1.0.0',
    author='Quantum Computing Research',
    author_email='research@quantum.example.com',
    description='Comprehensive framework for quantum Hamiltonian simulation algorithms',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/heysayan/QtmHamiltonianSimulation',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    python_requires='>=3.8',

    install_requires=read_requirements('requirements.txt'),

    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    keywords='quantum computing hamiltonian simulation qiskit trotterization qsvt',

    project_urls={
        'Bug Reports': 'https://github.com/heysayan/QtmHamiltonianSimulation/issues',
        'Source': 'https://github.com/heysayan/QtmHamiltonianSimulation',
    },
)
