"""
Interactive demonstration script for Hamiltonian simulation algorithms.

This script provides an interactive walkthrough of all implemented algorithms
with visualizations and comparisons.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.utils.hamiltonian_utils import create_test_hamiltonian
from src.algorithms.trotterization import TrotterizationSimulator
from src.algorithms.taylor_lcu import TaylorLCUSimulator
from src.algorithms.qsp import QSPSimulator, QubitizationSimulator
from src.algorithms.qsvt import QSVTSimulator
from src.utils.circuit_metrics import print_circuit_summary, compare_circuits
from src.benchmarks.hamiltonian_benchmark import HamiltonianBenchmark
from src.grover.hamiltonian_grover import GroverComparison


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_hamiltonian_creation():
    """Demonstrate Hamiltonian creation."""
    print_section("PART 1: Creating Test Hamiltonians")

    # Create different Hamiltonians
    n_qubits = 3

    print("Creating a 3-qubit Heisenberg Hamiltonian...")
    h_heisenberg = create_test_hamiltonian(n_qubits, "heisenberg")
    print(f"✓ Created Heisenberg model with {len(h_heisenberg.paulis)} Pauli terms")
    print(f"  Terms: {[str(p) for p in list(h_heisenberg.paulis)[:3]]}...")

    print("\nCreating a Transverse Ising Hamiltonian...")
    h_ising = create_test_hamiltonian(n_qubits, "transverse_ising")
    print(f"✓ Created Transverse Ising with {len(h_ising.paulis)} Pauli terms")

    return h_heisenberg


def demo_trotterization(hamiltonian):
    """Demonstrate Trotterization algorithm."""
    print_section("PART 2: Trotterization (Product Formulas)")

    time = 1.0
    n_steps = 10

    print("First-Order Trotterization:")
    print(f"  Evolution time: {time}")
    print(f"  Trotter steps: {n_steps}")

    simulator = TrotterizationSimulator(hamiltonian, time, order=1)
    circuit = simulator.build_circuit(n_steps)

    print_circuit_summary(circuit, "First-Order Trotter Circuit")

    # Error analysis
    error = simulator.estimate_error(n_steps)
    print(f"Estimated error: {error:.2e}")

    # Required steps for target error
    target_error = 1e-3
    required_steps = simulator.get_required_steps(target_error)
    print(f"Steps needed for ε={target_error}: {required_steps}")

    print("\n" + "-"*80)
    print("Second-Order Trotterization:")

    simulator2 = TrotterizationSimulator(hamiltonian, time, order=2)
    circuit2 = simulator2.build_circuit(n_steps)

    print_circuit_summary(circuit2, "Second-Order Trotter Circuit")

    error2 = simulator2.estimate_error(n_steps)
    print(f"Estimated error: {error2:.2e}")
    print(f"Improvement over 1st order: {error/error2:.1f}x")


def demo_taylor_lcu(hamiltonian):
    """Demonstrate Taylor-LCU algorithm."""
    print_section("PART 3: Truncated Taylor Series (LCU)")

    time = 1.0
    truncation_order = 10

    print("Taylor Series Approximation:")
    print(f"  Evolution time: {time}")
    print(f"  Truncation order: {truncation_order}")

    simulator = TaylorLCUSimulator(hamiltonian, time)
    circuit = simulator.build_circuit(truncation_order)

    print_circuit_summary(circuit, "Taylor-LCU Circuit")

    # Error analysis
    error = simulator.estimate_error(truncation_order)
    print(f"Estimated error: {error:.2e}")

    # Required order
    target_error = 1e-3
    required_order = simulator.get_required_order(target_error)
    print(f"Order needed for ε={target_error}: {required_order}")

    print(f"\nNormalization factor: {simulator.h_norm:.3f}")


def demo_qsp(hamiltonian):
    """Demonstrate QSP algorithm."""
    print_section("PART 4: Quantum Signal Processing (QSP)")

    time = 1.0
    degree = 10

    print("QSP with polynomial approximation:")
    print(f"  Evolution time: {time}")
    print(f"  Polynomial degree: {degree}")

    simulator = QSPSimulator(hamiltonian, time)
    circuit = simulator.build_circuit(degree)

    print_circuit_summary(circuit, "QSP Circuit")

    # Error analysis
    error = simulator.estimate_error(degree)
    print(f"Estimated error: {error:.2e}")

    # Required degree
    target_error = 1e-3
    required_degree = simulator.get_required_degree(target_error)
    print(f"Degree needed for ε={target_error}: {required_degree}")


def demo_qubitization(hamiltonian):
    """Demonstrate Qubitization algorithm."""
    print_section("PART 5: Qubitization (Optimal Hamiltonian Simulation)")

    time = 1.0

    print("Qubitization with automatic query complexity:")
    print(f"  Evolution time: {time}")

    simulator = QubitizationSimulator(hamiltonian, time)
    query_complexity = simulator._compute_query_complexity()

    print(f"  Computed query complexity: {query_complexity}")
    print(f"  Alpha (1-norm): {simulator.alpha:.3f}")

    circuit = simulator.build_circuit(query_complexity)

    print_circuit_summary(circuit, "Qubitization Circuit")

    # Error analysis
    error = simulator.estimate_error(query_complexity)
    print(f"Estimated error: {error:.2e}")


def demo_qsvt(hamiltonian):
    """Demonstrate QSVT algorithm."""
    print_section("PART 6: Quantum Singular Value Transform (QSVT)")

    time = 1.0
    degree = 10

    print("QSVT - The Grand Unification:")
    print(f"  Evolution time: {time}")
    print(f"  Polynomial degree: {degree}")

    simulator = QSVTSimulator(hamiltonian, time)
    circuit = simulator.build_circuit(degree)

    print_circuit_summary(circuit, "QSVT Circuit")

    # Error analysis
    error = simulator.estimate_error(degree)
    print(f"Estimated error: {error:.2e}")

    # Query complexity
    query_complexity = simulator.get_query_complexity(degree)
    print(f"Query complexity: {query_complexity}")

    print(f"\nSpectral norm: {simulator.spectral_norm:.3f}")
    print(f"Alpha (1-norm): {simulator.alpha:.3f}")


def demo_comprehensive_benchmark(hamiltonian):
    """Run comprehensive benchmark of all algorithms."""
    print_section("PART 7: Comprehensive Benchmark Comparison")

    time = 1.0

    print("Running all algorithms with comparable parameters...")
    print("This may take a moment...\n")

    benchmark = HamiltonianBenchmark(hamiltonian, time)

    # Run benchmarks
    benchmark.benchmark_trotter(order=1, n_steps=10)
    benchmark.benchmark_trotter(order=2, n_steps=10)
    benchmark.benchmark_taylor_lcu(truncation_order=10)
    benchmark.benchmark_qsp(degree=10)
    benchmark.benchmark_qubitization()
    benchmark.benchmark_qsvt(degree=10)

    # Print summary
    benchmark.print_summary_table()

    # Create and save plot
    print("Generating comparison plot...")
    benchmark.plot_comparison(save_path='benchmark_comparison.png')
    print("✓ Plot saved as 'benchmark_comparison.png'")


def demo_grover_comparison():
    """Demonstrate Grover's algorithm implementations."""
    print_section("PART 8: Grover's Search via Hamiltonian Simulation")

    n_qubits = 3
    marked_state = [7]  # |111⟩

    print(f"Grover's search problem:")
    print(f"  Search space: {2**n_qubits} items")
    print(f"  Marked state: {marked_state} (|111⟩)")

    comparison = GroverComparison(n_qubits, marked_state)
    print(f"  Optimal iterations: {comparison.optimal_iterations}\n")

    print("Comparing three implementations:")
    print("  1. Standard Grover (oracle + diffusion)")
    print("  2. Hamiltonian simulation via Taylor-LCU")
    print("  3. Hamiltonian simulation via QSVT\n")

    results = comparison.compare_all(
        num_iterations=comparison.optimal_iterations,
        taylor_order=8,
        qsvt_degree=8
    )

    df = comparison.print_comparison_table(results)

    print("\nKey Observations:")
    print("  • Standard Grover is most efficient (as expected)")
    print("  • Hamiltonian methods demonstrate generality")
    print("  • Same framework solves different problems")


def plot_error_vs_resources():
    """Plot error vs. computational resources for different algorithms."""
    print_section("PART 9: Error vs. Resource Trade-offs")

    n_qubits = 3
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 1.0

    steps_range = range(5, 51, 5)
    trotter_errors = []
    trotter_depths = []

    taylor_orders = range(5, 21, 2)
    taylor_errors = []
    taylor_depths = []

    print("Computing error vs. resource trade-offs...")

    # Trotter
    for steps in steps_range:
        sim = TrotterizationSimulator(hamiltonian, time, order=1)
        error = sim.estimate_error(steps)
        circuit = sim.build_circuit(steps)
        trotter_errors.append(error)
        trotter_depths.append(circuit.depth())

    # Taylor
    for order in taylor_orders:
        sim = TaylorLCUSimulator(hamiltonian, time)
        error = sim.estimate_error(order)
        circuit = sim.build_circuit(order)
        taylor_errors.append(error)
        taylor_depths.append(circuit.depth())

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Error vs. parameter
    ax1.semilogy(steps_range, trotter_errors, 'o-', label='Trotter', linewidth=2)
    ax1.semilogy(taylor_orders, taylor_errors, 's-', label='Taylor-LCU', linewidth=2)
    ax1.set_xlabel('Parameter (Steps / Order)', fontsize=12)
    ax1.set_ylabel('Estimated Error', fontsize=12)
    ax1.set_title('Error vs. Algorithm Parameter', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error vs. depth
    ax2.loglog(trotter_depths, trotter_errors, 'o-', label='Trotter', linewidth=2)
    ax2.loglog(taylor_depths, taylor_errors, 's-', label='Taylor-LCU', linewidth=2)
    ax2.set_xlabel('Circuit Depth', fontsize=12)
    ax2.set_ylabel('Estimated Error', fontsize=12)
    ax2.set_title('Error vs. Circuit Depth', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('error_vs_resources.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'error_vs_resources.png'\n")


def main():
    """Run the complete interactive demonstration."""
    print("\n" + "="*80)
    print("  QUANTUM HAMILTONIAN SIMULATION: INTERACTIVE DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates all implemented algorithms with")
    print("detailed explanations and visualizations.\n")

    input("Press Enter to begin...")

    # Part 1: Hamiltonian Creation
    hamiltonian = demo_hamiltonian_creation()
    input("\nPress Enter to continue...")

    # Part 2: Trotterization
    demo_trotterization(hamiltonian)
    input("\nPress Enter to continue...")

    # Part 3: Taylor-LCU
    demo_taylor_lcu(hamiltonian)
    input("\nPress Enter to continue...")

    # Part 4: QSP
    demo_qsp(hamiltonian)
    input("\nPress Enter to continue...")

    # Part 5: Qubitization
    demo_qubitization(hamiltonian)
    input("\nPress Enter to continue...")

    # Part 6: QSVT
    demo_qsvt(hamiltonian)
    input("\nPress Enter to continue...")

    # Part 7: Comprehensive Benchmark
    demo_comprehensive_benchmark(hamiltonian)
    input("\nPress Enter to continue...")

    # Part 8: Grover Comparison
    demo_grover_comparison()
    input("\nPress Enter to continue...")

    # Part 9: Error vs. Resources
    plot_error_vs_resources()

    print_section("DEMONSTRATION COMPLETE")
    print("All algorithms have been demonstrated!")
    print("\nGenerated files:")
    print("  • benchmark_comparison.png")
    print("  • error_vs_resources.png")
    print("\nThank you for exploring Hamiltonian simulation algorithms!")


if __name__ == "__main__":
    main()
