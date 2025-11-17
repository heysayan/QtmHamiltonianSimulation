"""
Example: Comprehensive comparison of Hamiltonian simulation algorithms.

This script demonstrates all implemented Hamiltonian simulation algorithms
and compares their performance on different test Hamiltonians.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils.hamiltonian_utils import create_test_hamiltonian
from benchmarks.hamiltonian_benchmark import run_comprehensive_benchmark


def main():
    """Run comprehensive Hamiltonian simulation benchmark."""

    print("\n" + "="*80)
    print("HAMILTONIAN SIMULATION ALGORITHM COMPARISON")
    print("="*80 + "\n")

    # Test parameters
    n_qubits = 3
    evolution_time = 1.0

    # Test different types of Hamiltonians
    hamiltonian_types = ["heisenberg", "transverse_ising"]

    for h_type in hamiltonian_types:
        print(f"\n{'='*80}")
        print(f"Testing with {h_type.upper()} Hamiltonian")
        print(f"{'='*80}\n")

        # Create test Hamiltonian
        hamiltonian = create_test_hamiltonian(n_qubits, h_type)

        print(f"Hamiltonian type: {h_type}")
        print(f"Number of qubits: {n_qubits}")
        print(f"Number of terms: {len(hamiltonian.paulis)}")
        print(f"Evolution time: {evolution_time}")
        print()

        # Run benchmark
        plot_path = f"hamiltonian_comparison_{h_type}.png"
        df = run_comprehensive_benchmark(
            hamiltonian,
            evolution_time,
            plot_path=plot_path
        )

        # Save results to CSV
        csv_path = f"results_{h_type}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}\n")


def demo_individual_algorithms():
    """Demonstrate individual algorithm usage."""

    from algorithms.trotterization import TrotterizationSimulator
    from algorithms.taylor_lcu import TaylorLCUSimulator
    from algorithms.qsp import QSPSimulator
    from algorithms.qsvt import QSVTSimulator
    from utils.circuit_metrics import print_circuit_summary

    print("\n" + "="*80)
    print("INDIVIDUAL ALGORITHM DEMONSTRATIONS")
    print("="*80 + "\n")

    # Create a simple test Hamiltonian
    n_qubits = 2
    hamiltonian = create_test_hamiltonian(n_qubits, "heisenberg")
    time = 0.5

    print(f"Test Hamiltonian: {n_qubits} qubits, {len(hamiltonian.paulis)} terms")
    print(f"Evolution time: {time}\n")

    # 1. Trotterization
    print("1. FIRST-ORDER TROTTERIZATION")
    print("-" * 40)
    trotter_sim = TrotterizationSimulator(hamiltonian, time, order=1)
    trotter_circuit = trotter_sim.build_circuit(n_trotter_steps=5)
    print_circuit_summary(trotter_circuit, "Trotter Circuit")

    # 2. Taylor-LCU
    print("\n2. TAYLOR SERIES (LCU)")
    print("-" * 40)
    taylor_sim = TaylorLCUSimulator(hamiltonian, time)
    taylor_circuit = taylor_sim.build_circuit(truncation_order=5)
    print_circuit_summary(taylor_circuit, "Taylor-LCU Circuit")

    # 3. QSP
    print("\n3. QUANTUM SIGNAL PROCESSING")
    print("-" * 40)
    qsp_sim = QSPSimulator(hamiltonian, time)
    qsp_circuit = qsp_sim.build_circuit(degree=5)
    print_circuit_summary(qsp_circuit, "QSP Circuit")

    # 4. QSVT
    print("\n4. QUANTUM SINGULAR VALUE TRANSFORM")
    print("-" * 40)
    qsvt_sim = QSVTSimulator(hamiltonian, time)
    qsvt_circuit = qsvt_sim.build_circuit(degree=5)
    print_circuit_summary(qsvt_circuit, "QSVT Circuit")


def complexity_analysis():
    """Analyze scaling of algorithms with system size."""

    from algorithms.trotterization import TrotterizationSimulator
    from algorithms.qsvt import QSVTSimulator
    import matplotlib.pyplot as plt

    print("\n" + "="*80)
    print("COMPLEXITY SCALING ANALYSIS")
    print("="*80 + "\n")

    qubit_range = range(2, 6)
    time = 1.0

    trotter_depths = []
    qsvt_depths = []
    trotter_gates = []
    qsvt_gates = []

    for n in qubit_range:
        print(f"Analyzing {n} qubits...")

        hamiltonian = create_test_hamiltonian(n, "heisenberg")

        # Trotter
        trotter_sim = TrotterizationSimulator(hamiltonian, time, order=1)
        trotter_circ = trotter_sim.build_circuit(n_trotter_steps=10)
        trotter_depths.append(trotter_circ.depth())
        trotter_gates.append(trotter_circ.size())

        # QSVT
        qsvt_sim = QSVTSimulator(hamiltonian, time)
        qsvt_circ = qsvt_sim.build_circuit(degree=10)
        qsvt_depths.append(qsvt_circ.depth())
        qsvt_gates.append(qsvt_circ.size())

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(list(qubit_range), trotter_depths, 'o-', label='Trotter', linewidth=2)
    ax1.plot(list(qubit_range), qsvt_depths, 's-', label='QSVT', linewidth=2)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Circuit Depth')
    ax1.set_title('Circuit Depth vs System Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(qubit_range), trotter_gates, 'o-', label='Trotter', linewidth=2)
    ax2.plot(list(qubit_range), qsvt_gates, 's-', label='QSVT', linewidth=2)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Gate Count')
    ax2.set_title('Gate Count vs System Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('complexity_scaling.png', dpi=300, bbox_inches='tight')
    print("\nScaling plot saved to: complexity_scaling.png")


if __name__ == "__main__":
    # Run main benchmark
    main()

    # Demonstrate individual algorithms
    demo_individual_algorithms()

    # Complexity analysis
    complexity_analysis()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80 + "\n")
