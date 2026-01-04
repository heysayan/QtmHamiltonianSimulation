"""
Pytest configuration and fixtures for test suite.
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@pytest.fixture
def small_hamiltonian():
    """Create a small 2-qubit Heisenberg Hamiltonian for testing."""
    from utils.hamiltonian_utils import create_test_hamiltonian
    return create_test_hamiltonian(2, "heisenberg")


@pytest.fixture
def medium_hamiltonian():
    """Create a medium 3-qubit Heisenberg Hamiltonian for testing."""
    from utils.hamiltonian_utils import create_test_hamiltonian
    return create_test_hamiltonian(3, "heisenberg")


@pytest.fixture
def ising_hamiltonian():
    """Create a 2-qubit transverse Ising Hamiltonian for testing."""
    from utils.hamiltonian_utils import create_test_hamiltonian
    return create_test_hamiltonian(2, "transverse_ising")


@pytest.fixture
def short_time():
    """Standard short evolution time for testing."""
    return 0.3


@pytest.fixture
def medium_time():
    """Standard medium evolution time for testing."""
    return 0.5


@pytest.fixture
def long_time():
    """Standard long evolution time for testing."""
    return 1.0


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "numerical: marks numerical validation tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark numerical validation tests
        if "numerical_validation" in str(item.fspath):
            item.add_marker(pytest.mark.numerical)

        # Mark slow tests
        if "comprehensive" in item.name or "scaling" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark unit tests
        if "test_algorithms.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
