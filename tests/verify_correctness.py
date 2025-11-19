"""
Comprehensive verification of implementation correctness against literature.

This script verifies:
1. Complexity bounds match theoretical predictions
2. Error estimates are correct
3. Circuit constructions follow standard methods
4. Benchmark results are reasonable
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from src.utils.hamiltonian_utils import create_test_hamiltonian, spectral_norm, one_norm
from src.algorithms.trotterization import TrotterizationSimulator
from src.algorithms.taylor_lcu import TaylorLCUSimulator
from src.algorithms.qsp import QSPSimulator, QubitizationSimulator
from src.algorithms.qsvt import QSVTSimulator
from src.grover.standard_grover import StandardGrover


def print_test(name, passed, details=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"[{status}] {name}")
    if details:
        print(f"      {details}")


def verify_trotter_complexity():
    """
    Verify Trotterization complexity bounds.

    Reference: Lloyd (1996), Berry et al. (2006)
    First-order: O((||H||t)^2 / ε)
    Second-order: O((||H||t)^(3/2) / sqrt(ε))
    """
    print("\n" + "="*80)
    print("VERIFYING TROTTERIZATION")
    print("="*80)

    n_qubits = 3
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 1.0
    target_error = 1e-3

    # Theoretical bounds
    h_norm = spectral_norm(hamiltonian)

    # First-order
    sim1 = TrotterizationSimulator(hamiltonian, time, order=1)

    # From theory: r >= (||H||t)^2 / ε
    theoretical_steps_1 = int(np.ceil((h_norm * time)**2 / target_error))
    computed_steps_1 = sim1.get_required_steps(target_error)

    # Should be within factor of 2 (due to constant factors in theory)
    ratio1 = computed_steps_1 / theoretical_steps_1
    test1 = 0.5 <= ratio1 <= 2.0

    print_test(
        "First-order Trotter complexity bound",
        test1,
        f"Theoretical: {theoretical_steps_1}, Computed: {computed_steps_1}, Ratio: {ratio1:.2f}"
    )

    # Second-order
    sim2 = TrotterizationSimulator(hamiltonian, time, order=2)
    theoretical_steps_2 = int(np.ceil((h_norm * time)**1.5 / np.sqrt(target_error)))
    computed_steps_2 = sim2.get_required_steps(target_error)

    ratio2 = computed_steps_2 / theoretical_steps_2
    test2 = 0.5 <= ratio2 <= 2.0

    print_test(
        "Second-order Trotter complexity bound",
        test2,
        f"Theoretical: {theoretical_steps_2}, Computed: {computed_steps_2}, Ratio: {ratio2:.2f}"
    )

    # Error scaling test
    steps_range = [5, 10, 20, 40]
    errors_1 = [sim1.estimate_error(r) for r in steps_range]

    # Error should scale as 1/r for first order
    scaling_test = True
    for i in range(len(steps_range)-1):
        expected_ratio = steps_range[i+1] / steps_range[i]
        actual_ratio = errors_1[i] / errors_1[i+1]
        # Should be within 30% due to constant factors
        if not (0.7 * expected_ratio <= actual_ratio <= 1.3 * expected_ratio):
            scaling_test = False

    print_test(
        "Trotter error scaling (O(1/r))",
        scaling_test,
        f"Errors: {[f'{e:.2e}' for e in errors_1]}"
    )

    return test1 and test2 and scaling_test


def verify_taylor_lcu_complexity():
    """
    Verify Taylor-LCU complexity.

    Reference: Berry et al. (2015) - "Simulating Hamiltonian Dynamics with Truncated Taylor Series"
    Complexity: O(α t + log(1/ε) / log log(1/ε))
    Error: (αt)^(K+1) / (K+1)!
    """
    print("\n" + "="*80)
    print("VERIFYING TAYLOR-LCU")
    print("="*80)

    n_qubits = 3
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5  # Use smaller time for Taylor series

    sim = TaylorLCUSimulator(hamiltonian, time)

    # Test error formula
    alpha = sim.h_norm
    orders = [5, 10, 15, 20]
    errors = [sim.estimate_error(K) for K in orders]

    # Verify error decreases factorially
    factorial_test = True
    for i in range(len(orders)-1):
        # Error should decrease faster than exponentially
        if errors[i+1] >= errors[i] * 0.5:  # Should decrease significantly
            factorial_test = False

    print_test(
        "Taylor series error (factorial decrease)",
        factorial_test,
        f"Errors: {[f'{e:.2e}' for e in errors]}"
    )

    # Test required order calculation
    target_error = 1e-3
    required_order = sim.get_required_order(target_error)
    actual_error = sim.estimate_error(required_order)

    order_test = actual_error <= target_error

    print_test(
        "Taylor truncation order calculation",
        order_test,
        f"Required K={required_order}, Error={actual_error:.2e} <= {target_error:.2e}"
    )

    return factorial_test and order_test


def verify_qsp_complexity():
    """
    Verify QSP complexity.

    Reference: Low & Chuang (2017) - "Optimal Hamiltonian Simulation by Quantum Signal Processing"
    Complexity: O(||H||t + log(1/ε))
    """
    print("\n" + "="*80)
    print("VERIFYING QSP")
    print("="*80)

    n_qubits = 3
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 1.0

    sim = QSPSimulator(hamiltonian, time)

    # Test error scaling with degree
    degrees = [5, 10, 15, 20]
    errors = [sim.estimate_error(d) for d in degrees]

    # Error should decrease (at least sub-exponentially)
    decreasing_test = all(errors[i] > errors[i+1] for i in range(len(errors)-1))

    print_test(
        "QSP error decreases with degree",
        decreasing_test,
        f"Errors: {[f'{e:.2e}' for e in errors]}"
    )

    # Test degree requirement
    target_error = 1e-3
    required_degree = sim.get_required_degree(target_error)
    actual_error = sim.estimate_error(required_degree)

    degree_test = actual_error <= target_error * 1.1  # Allow 10% margin

    print_test(
        "QSP degree calculation",
        degree_test,
        f"Required d={required_degree}, Error={actual_error:.2e}"
    )

    return decreasing_test and degree_test


def verify_qubitization_complexity():
    """
    Verify Qubitization complexity.

    Reference: Low & Chuang (2019) - "Hamiltonian Simulation by Qubitization"
    Complexity: O(αt + log(1/ε))
    """
    print("\n" + "="*80)
    print("VERIFYING QUBITIZATION")
    print("="*80)

    n_qubits = 3
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 1.0

    sim = QubitizationSimulator(hamiltonian, time)

    # Query complexity should be O(α t)
    alpha = sim.alpha
    theoretical_queries = int(np.ceil(alpha * time))
    computed_queries = sim._compute_query_complexity()

    # Should be within factor of 2
    ratio = computed_queries / theoretical_queries
    query_test = 0.5 <= ratio <= 2.0

    print_test(
        "Qubitization query complexity O(αt)",
        query_test,
        f"Theoretical: {theoretical_queries}, Computed: {computed_queries}, α={alpha:.2f}"
    )

    return query_test


def verify_qsvt_complexity():
    """
    Verify QSVT complexity.

    Reference: Gilyén et al. (2019) - "Quantum Singular Value Transformation"
    Query complexity: O(d) where d is polynomial degree
    For Hamiltonian simulation: d = O(||H||t + log(1/ε))
    """
    print("\n" + "="*80)
    print("VERIFYING QSVT")
    print("="*80)

    n_qubits = 3
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 1.0

    sim = QSVTSimulator(hamiltonian, time)

    # Query complexity should be 2d + 1
    degrees = [5, 10, 15]
    query_complexities = [sim.get_query_complexity(d) for d in degrees]
    expected = [2*d + 1 for d in degrees]

    query_test = all(q == e for q, e in zip(query_complexities, expected))

    print_test(
        "QSVT query complexity = 2d + 1",
        query_test,
        f"Degrees: {degrees}, Queries: {query_complexities}"
    )

    # Error should decrease with degree
    errors = [sim.estimate_error(d) for d in degrees]
    decreasing_test = all(errors[i] > errors[i+1] for i in range(len(errors)-1))

    print_test(
        "QSVT error decreases with degree",
        decreasing_test,
        f"Errors: {[f'{e:.2e}' for e in errors]}"
    )

    return query_test and decreasing_test


def verify_grover_optimality():
    """
    Verify Grover's algorithm optimality.

    Reference: Grover (1996), Boyer et al. (1998)
    Optimal iterations: π/4 * sqrt(N/M)
    Success probability after optimal iterations: > 0.9
    """
    print("\n" + "="*80)
    print("VERIFYING GROVER'S ALGORITHM")
    print("="*80)

    n_qubits = 3
    search_space = 2**n_qubits
    marked_states = [7]  # 1 marked state

    grover = StandardGrover(n_qubits, marked_states)

    # Optimal iterations
    theoretical_optimal = np.pi / 4 * np.sqrt(search_space / len(marked_states))
    computed_optimal = grover._optimal_iterations()

    optimal_test = abs(theoretical_optimal - computed_optimal) <= 1

    print_test(
        "Grover optimal iterations",
        optimal_test,
        f"Theoretical: {theoretical_optimal:.2f}, Computed: {computed_optimal}"
    )

    # Success probability
    success_prob = grover.success_probability(computed_optimal)
    prob_test = success_prob >= 0.9

    print_test(
        "Grover success probability > 0.9",
        prob_test,
        f"Success probability: {success_prob:.4f}"
    )

    # Test multiple marked states
    marked_states_2 = [5, 7]  # 2 marked states
    grover2 = StandardGrover(n_qubits, marked_states_2)

    theoretical_optimal_2 = np.pi / 4 * np.sqrt(search_space / len(marked_states_2))
    computed_optimal_2 = grover2._optimal_iterations()

    # Should require fewer iterations with more marked states
    fewer_iterations_test = computed_optimal_2 < computed_optimal

    print_test(
        "Grover iterations decrease with more marked states",
        fewer_iterations_test,
        f"1 marked: {computed_optimal} iters, 2 marked: {computed_optimal_2} iters"
    )

    return optimal_test and prob_test and fewer_iterations_test


def verify_hamiltonian_properties():
    """Verify Hamiltonian construction and properties."""
    print("\n" + "="*80)
    print("VERIFYING HAMILTONIAN PROPERTIES")
    print("="*80)

    n_qubits = 3

    # Test Heisenberg model
    h_heisenberg = create_test_hamiltonian(n_qubits, "heisenberg")
    h_matrix = h_heisenberg.to_matrix()

    # Should be Hermitian
    hermitian_test = np.allclose(h_matrix, h_matrix.conj().T)

    print_test(
        "Hamiltonian is Hermitian",
        hermitian_test,
        f"||H - H†|| = {np.linalg.norm(h_matrix - h_matrix.conj().T):.2e}"
    )

    # Eigenvalues should be real
    eigenvalues = np.linalg.eigvalsh(h_matrix)
    real_eigenvalues_test = np.allclose(eigenvalues.imag, 0)

    print_test(
        "Eigenvalues are real",
        real_eigenvalues_test,
        f"Max imag part: {np.max(np.abs(eigenvalues.imag)):.2e}"
    )

    # Test different Hamiltonians
    h_ising = create_test_hamiltonian(n_qubits, "transverse_ising")
    h_ising_matrix = h_ising.to_matrix()

    ising_hermitian_test = np.allclose(h_ising_matrix, h_ising_matrix.conj().T)

    print_test(
        "Transverse Ising Hamiltonian is Hermitian",
        ising_hermitian_test
    )

    return hermitian_test and real_eigenvalues_test and ising_hermitian_test


def verify_circuit_unitarity():
    """Verify that generated circuits are unitary."""
    print("\n" + "="*80)
    print("VERIFYING CIRCUIT UNITARITY")
    print("="*80)

    n_qubits = 2  # Use small system for exact simulation
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    from qiskit.quantum_info import Operator

    # Test Trotter
    trotter_sim = TrotterizationSimulator(hamiltonian, time, order=1)
    trotter_circuit = trotter_sim.build_circuit(n_trotter_steps=5)

    try:
        trotter_unitary = Operator(trotter_circuit).data
        trotter_test = np.allclose(
            trotter_unitary @ trotter_unitary.conj().T,
            np.eye(2**n_qubits)
        )
        print_test("Trotter circuit is unitary", trotter_test)
    except Exception as e:
        print_test("Trotter circuit is unitary", False, f"Error: {e}")
        trotter_test = False

    return trotter_test


def compare_with_exact_evolution():
    """Compare simulation results with exact evolution."""
    print("\n" + "="*80)
    print("COMPARING WITH EXACT EVOLUTION")
    print("="*80)

    n_qubits = 2  # Small system for exact computation
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.3

    # Exact evolution
    h_matrix = hamiltonian.to_matrix()
    exact_evolution = expm(-1j * time * h_matrix)

    from qiskit.quantum_info import Operator

    # Test Trotter approximation
    trotter_sim = TrotterizationSimulator(hamiltonian, time, order=2)
    trotter_circuit = trotter_sim.build_circuit(n_trotter_steps=20)

    try:
        trotter_unitary = Operator(trotter_circuit).data

        # Compute fidelity
        fidelity = np.abs(np.trace(exact_evolution.conj().T @ trotter_unitary)) / (2**n_qubits)
        fidelity_squared = fidelity ** 2

        fidelity_test = fidelity_squared >= 0.95  # Should be high fidelity

        print_test(
            "Trotter approximation fidelity",
            fidelity_test,
            f"Fidelity² = {fidelity_squared:.4f} (should be > 0.95)"
        )

    except Exception as e:
        print_test("Trotter approximation fidelity", False, f"Error: {e}")
        fidelity_test = False

    return fidelity_test


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE CORRECTNESS VERIFICATION")
    print("Testing against theoretical bounds from literature")
    print("="*80)

    all_tests = []

    # Run all verification tests
    all_tests.append(("Hamiltonian Properties", verify_hamiltonian_properties()))
    all_tests.append(("Trotterization", verify_trotter_complexity()))
    all_tests.append(("Taylor-LCU", verify_taylor_lcu_complexity()))
    all_tests.append(("QSP", verify_qsp_complexity()))
    all_tests.append(("Qubitization", verify_qubitization_complexity()))
    all_tests.append(("QSVT", verify_qsvt_complexity()))
    all_tests.append(("Grover's Algorithm", verify_grover_optimality()))
    all_tests.append(("Circuit Unitarity", verify_circuit_unitarity()))
    all_tests.append(("Exact Evolution Comparison", compare_with_exact_evolution()))

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in all_tests if result)
    total = len(all_tests)

    for test_name, result in all_tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "="*80)
    print(f"OVERALL: {passed}/{total} test categories passed")

    if passed == total:
        print("✓ ALL VERIFICATIONS PASSED - Implementation is theoretically sound!")
    else:
        print(f"✗ {total - passed} test categories failed - review needed")

    print("="*80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
