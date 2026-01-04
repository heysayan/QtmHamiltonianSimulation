"""
Basic tests for Hamiltonian simulation algorithms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from utils.hamiltonian_utils import create_test_hamiltonian
from algorithms.trotterization import TrotterizationSimulator
from algorithms.taylor_lcu import TaylorLCUSimulator
from algorithms.qsp import QSPSimulator, QubitizationSimulator
from algorithms.qsvt import QSVTSimulator


def test_create_test_hamiltonians():
    """Test that we can create various test Hamiltonians."""
    n_qubits = 2

    # Test different Hamiltonian types
    h_heisenberg = create_test_hamiltonian(n_qubits, "heisenberg")
    assert h_heisenberg.num_qubits == n_qubits
    assert len(h_heisenberg.paulis) > 0

    h_ising = create_test_hamiltonian(n_qubits, "transverse_ising")
    assert h_ising.num_qubits == n_qubits
    assert len(h_ising.paulis) > 0


def test_trotterization():
    """Test Trotterization algorithm."""
    n_qubits = 2
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    # First-order Trotter
    simulator = TrotterizationSimulator(hamiltonian, time, order=1)
    circuit = simulator.build_circuit(n_trotter_steps=5)

    assert circuit is not None
    assert circuit.num_qubits == n_qubits

    # Check error estimation
    error = simulator.estimate_error(5)
    assert error >= 0


def test_taylor_lcu():
    """Test Taylor-LCU algorithm."""
    n_qubits = 2
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    simulator = TaylorLCUSimulator(hamiltonian, time)
    circuit = simulator.build_circuit(truncation_order=5)

    assert circuit is not None
    assert circuit.num_qubits >= n_qubits  # May have ancillas

    # Check error estimation
    error = simulator.estimate_error(5)
    assert error >= 0


def test_qsp():
    """Test QSP algorithm."""
    n_qubits = 2
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    simulator = QSPSimulator(hamiltonian, time)
    circuit = simulator.build_circuit(degree=5)

    assert circuit is not None
    assert circuit.num_qubits >= n_qubits

    # Check error estimation
    error = simulator.estimate_error(5)
    assert error >= 0


def test_qubitization():
    """Test Qubitization algorithm."""
    n_qubits = 2
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    simulator = QubitizationSimulator(hamiltonian, time)
    circuit = simulator.build_circuit()

    assert circuit is not None
    assert circuit.num_qubits >= n_qubits


def test_qsvt():
    """Test QSVT algorithm."""
    n_qubits = 2
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    simulator = QSVTSimulator(hamiltonian, time)
    circuit = simulator.build_circuit(degree=5)

    assert circuit is not None
    assert circuit.num_qubits >= n_qubits

    # Check error estimation
    error = simulator.estimate_error(5)
    assert error >= 0

    # Check query complexity
    query_complexity = simulator.get_query_complexity(5)
    assert query_complexity > 0


def test_grover_standard():
    """Test standard Grover's algorithm."""
    from grover.standard_grover import StandardGrover

    n_qubits = 2
    marked_states = [3]  # Mark state |11⟩

    grover = StandardGrover(n_qubits, marked_states)
    circuit = grover.build_circuit()

    assert circuit is not None
    assert circuit.num_qubits == n_qubits

    # Check success probability
    prob = grover.success_probability(grover._optimal_iterations())
    assert 0 <= prob <= 1


def test_grover_via_taylor():
    """Test Grover via Taylor-LCU."""
    from grover.hamiltonian_grover import GroverViaTaylor

    n_qubits = 2
    marked_states = [3]

    grover = GroverViaTaylor(n_qubits, marked_states)
    circuit = grover.build_circuit(num_iterations=1, truncation_order=5)

    assert circuit is not None
    assert circuit.num_qubits >= n_qubits


def test_grover_via_qsvt():
    """Test Grover via QSVT."""
    from grover.hamiltonian_grover import GroverViaQSVT

    n_qubits = 2
    marked_states = [3]

    grover = GroverViaQSVT(n_qubits, marked_states)
    circuit = grover.build_circuit(num_iterations=1, polynomial_degree=5)

    assert circuit is not None
    assert circuit.num_qubits >= n_qubits


def test_linear_solver_qsvt():
    """Test QSVT linear system solver."""
    from algorithms.linear_solver_qsvt import QSVTLinearSolver
    import numpy as np

    # Simple 2x2 system
    A = np.array([[2, 1], [1, 2]], dtype=complex)
    b = np.array([1, 1], dtype=complex)

    solver = QSVTLinearSolver(A, b)
    circuit = solver.build_circuit(polynomial_degree=5)

    assert circuit is not None
    assert circuit.num_qubits >= solver.n_qubits

    # Check that classical solution works
    x = solver.solve()
    assert x is not None
    assert len(x) == len(b)

    # Verify Ax = b
    result = A @ x
    error = np.linalg.norm(result - b)
    assert error < 1e-6


if __name__ == "__main__":
    print("Running tests...")
    pytest.main([__file__, "-v"])
