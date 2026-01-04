"""
Test that example scripts run without errors.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.utils.hamiltonian_utils import create_test_hamiltonian
from src.algorithms.trotterization import first_order_trotter, second_order_trotter
from src.algorithms.taylor_lcu import taylor_series_simulation
from src.algorithms.qsp import qsp_simulation, qubitization_simulation
from src.algorithms.qsvt import qsvt_simulation, qsvt_optimal_simulation
from src.grover.standard_grover import create_grover_circuit
from src.grover.hamiltonian_grover import (
    grover_taylor_simulation,
    grover_qsvt_simulation
)


def test_quick_example():
    """Test basic usage examples from README."""
    print("Testing quick example from README...")

    # Create Hamiltonian
    hamiltonian = create_test_hamiltonian(n_qubits=3, hamiltonian_type="heisenberg")
    time = 1.0

    # Trotterization
    trotter_circuit = first_order_trotter(hamiltonian, time, n_trotter_steps=10)
    assert trotter_circuit is not None
    assert trotter_circuit.num_qubits == 3

    # Second-order
    trotter2_circuit = second_order_trotter(hamiltonian, time, n_trotter_steps=10)
    assert trotter2_circuit is not None

    # Taylor
    taylor_circuit = taylor_series_simulation(hamiltonian, time, truncation_order=10)
    assert taylor_circuit is not None

    # QSP
    qsp_circuit = qsp_simulation(hamiltonian, time, degree=10)
    assert qsp_circuit is not None

    # Qubitization
    qubit_circuit = qubitization_simulation(hamiltonian, time)
    assert qubit_circuit is not None

    # QSVT
    qsvt_circuit = qsvt_simulation(hamiltonian, time, degree=10)
    assert qsvt_circuit is not None

    print("✓ All basic examples work correctly")


def test_grover_examples():
    """Test Grover algorithm examples."""
    print("\nTesting Grover examples...")

    n_qubits = 3
    marked_states = [7]

    # Standard Grover
    grover_circuit = create_grover_circuit(n_qubits, marked_states)
    assert grover_circuit is not None
    assert grover_circuit.num_qubits == n_qubits

    # Taylor method
    taylor_grover = grover_taylor_simulation(n_qubits, marked_states,
                                              num_iterations=2,
                                              truncation_order=8)
    assert taylor_grover is not None

    # QSVT method
    qsvt_grover = grover_qsvt_simulation(n_qubits, marked_states,
                                          num_iterations=2,
                                          polynomial_degree=8)
    assert qsvt_grover is not None

    print("✓ All Grover examples work correctly")


def test_advanced_usage():
    """Test advanced usage patterns."""
    print("\nTesting advanced usage...")

    hamiltonian = create_test_hamiltonian(n_qubits=2, hamiltonian_type="transverse_ising")
    time = 0.5

    # QSVT with automatic parameters
    circuit, metadata = qsvt_optimal_simulation(hamiltonian, time, target_error=1e-3)

    assert circuit is not None
    assert 'degree' in metadata
    assert 'query_complexity' in metadata
    assert 'estimated_error' in metadata
    assert metadata['estimated_error'] <= 1e-3 * 1.1  # Allow small margin

    print(f"✓ Automatic QSVT: degree={metadata['degree']}, "
          f"queries={metadata['query_complexity']}, "
          f"error={metadata['estimated_error']:.2e}")


def test_different_hamiltonians():
    """Test with different types of Hamiltonians."""
    print("\nTesting different Hamiltonians...")

    n_qubits = 3
    time = 0.5

    for h_type in ["heisenberg", "transverse_ising"]:
        hamiltonian = create_test_hamiltonian(n_qubits, h_type)

        # Test each algorithm
        circuits = {
            'Trotter': first_order_trotter(hamiltonian, time, 5),
            'Taylor': taylor_series_simulation(hamiltonian, time, 8),
            'QSP': qsp_simulation(hamiltonian, time, 8),
            'QSVT': qsvt_simulation(hamiltonian, time, 8)
        }

        for name, circuit in circuits.items():
            assert circuit is not None, f"{name} failed for {h_type}"

        print(f"✓ All algorithms work with {h_type} Hamiltonian")


def test_scaling():
    """Test that algorithms scale to different system sizes."""
    print("\nTesting scaling with system size...")

    time = 0.5

    for n_qubits in [2, 3, 4]:
        hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")

        # Test representative algorithms
        trotter_circuit = first_order_trotter(hamiltonian, time, 5)
        qsvt_circuit = qsvt_simulation(hamiltonian, time, 5)

        assert trotter_circuit.num_qubits == n_qubits
        # QSVT may have ancilla qubits
        assert qsvt_circuit.num_qubits >= n_qubits

        print(f"✓ Algorithms scale correctly to {n_qubits} qubits")


def main():
    """Run all example tests."""
    print("="*80)
    print("TESTING EXAMPLE USAGE")
    print("="*80)

    tests = [
        ("Quick examples", test_quick_example),
        ("Grover examples", test_grover_examples),
        ("Advanced usage", test_advanced_usage),
        ("Different Hamiltonians", test_different_hamiltonians),
        ("Scaling", test_scaling)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"EXAMPLE TESTS: {passed} passed, {failed} failed")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
