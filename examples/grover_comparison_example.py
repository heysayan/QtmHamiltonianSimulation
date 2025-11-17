"""
Example: Comparison of Grover's search implementations.

This script compares:
1. Standard Grover's algorithm
2. Grover via Hamiltonian simulation (Taylor-LCU)
3. Grover via Hamiltonian simulation (QSVT)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from grover.standard_grover import StandardGrover
from grover.hamiltonian_grover import GroverViaTaylor, GroverViaQSVT, GroverComparison
from utils.circuit_metrics import analyze_circuit


def compare_grover_implementations():
    """Compare different Grover implementations."""

    print("\n" + "="*80)
    print("GROVER'S ALGORITHM COMPARISON")
    print("="*80 + "\n")

    # Test parameters
    n_qubits = 3
    marked_state = [7]  # Search for state |111⟩

    print(f"Search space: {2**n_qubits} items")
    print(f"Marked state: {marked_state}")
    print()

    # Create comparison object
    comparison = GroverComparison(n_qubits, marked_state)

    print(f"Optimal number of iterations: {comparison.optimal_iterations}\n")

    # Compare all methods
    results = comparison.compare_all(
        num_iterations=comparison.optimal_iterations,
        taylor_order=10,
        qsvt_degree=10
    )

    # Print comparison table
    df = comparison.print_comparison_table(results)

    # Save results
    df.to_csv('grover_comparison_results.csv', index=False)
    print("Results saved to: grover_comparison_results.csv\n")

    # Create visualization
    plot_grover_comparison(results)


def plot_grover_comparison(results: dict):
    """
    Create comparison plots for Grover implementations.

    Args:
        results: Results dictionary from GroverComparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Grover's Algorithm: Standard vs Hamiltonian Simulation",
                 fontsize=16, fontweight='bold')

    methods = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Extract metrics
    depths = [results[m]['metrics']['depth'] for m in methods]
    gates = [results[m]['metrics']['total_gates'] for m in methods]
    cnots = [results[m]['metrics']['cnot_count'] for m in methods]
    qubits = [results[m]['metrics']['num_qubits'] for m in methods]

    # Circuit Depth
    axes[0, 0].bar(methods, depths, color=colors)
    axes[0, 0].set_title('Circuit Depth', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Depth')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Total Gates
    axes[0, 1].bar(methods, gates, color=colors)
    axes[0, 1].set_title('Total Gate Count', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Gates')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # CNOT Count
    axes[1, 0].bar(methods, cnots, color=colors)
    axes[1, 0].set_title('CNOT Count', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('CNOTs')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # Qubit Count
    axes[1, 1].bar(methods, qubits, color=colors)
    axes[1, 1].set_title('Qubit Count', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Qubits')
    axes[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig('grover_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to: grover_comparison.png\n")


def scaling_analysis():
    """Analyze how methods scale with problem size."""

    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80 + "\n")

    qubit_range = range(2, 5)  # 2 to 4 qubits
    taylor_order = 8
    qsvt_degree = 8

    results_data = {
        'n_qubits': [],
        'standard_depth': [],
        'standard_gates': [],
        'taylor_depth': [],
        'taylor_gates': [],
        'qsvt_depth': [],
        'qsvt_gates': [],
    }

    for n in qubit_range:
        print(f"Analyzing {n} qubits...")

        marked_state = [2**n - 1]  # All-ones state

        # Optimal iterations
        search_space = 2 ** n
        optimal_iter = int(np.round((np.pi / 4) * np.sqrt(search_space)))

        # Standard Grover
        standard = StandardGrover(n, marked_state)
        standard_circ = standard.build_circuit(optimal_iter)

        # Taylor Grover
        taylor = GroverViaTaylor(n, marked_state)
        taylor_circ = taylor.build_circuit(optimal_iter, taylor_order)

        # QSVT Grover
        qsvt = GroverViaQSVT(n, marked_state)
        qsvt_circ = qsvt.build_circuit(optimal_iter, qsvt_degree)

        # Record metrics
        results_data['n_qubits'].append(n)
        results_data['standard_depth'].append(standard_circ.depth())
        results_data['standard_gates'].append(standard_circ.size())
        results_data['taylor_depth'].append(taylor_circ.depth())
        results_data['taylor_gates'].append(taylor_circ.size())
        results_data['qsvt_depth'].append(qsvt_circ.depth())
        results_data['qsvt_gates'].append(qsvt_circ.size())

    # Plot scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    n_vals = results_data['n_qubits']

    # Depth scaling
    ax1.plot(n_vals, results_data['standard_depth'], 'o-', label='Standard', linewidth=2)
    ax1.plot(n_vals, results_data['taylor_depth'], 's-', label='Taylor-LCU', linewidth=2)
    ax1.plot(n_vals, results_data['qsvt_depth'], '^-', label='QSVT', linewidth=2)
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Circuit Depth', fontsize=12)
    ax1.set_title('Circuit Depth vs Problem Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gate scaling
    ax2.plot(n_vals, results_data['standard_gates'], 'o-', label='Standard', linewidth=2)
    ax2.plot(n_vals, results_data['taylor_gates'], 's-', label='Taylor-LCU', linewidth=2)
    ax2.plot(n_vals, results_data['qsvt_gates'], '^-', label='QSVT', linewidth=2)
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('Gate Count', fontsize=12)
    ax2.set_title('Gate Count vs Problem Size', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grover_scaling.png', dpi=300, bbox_inches='tight')
    print("\nScaling plot saved to: grover_scaling.png")

    # Save data
    df = pd.DataFrame(results_data)
    df.to_csv('grover_scaling_data.csv', index=False)
    print("Scaling data saved to: grover_scaling_data.csv\n")


def demo_individual_methods():
    """Demonstrate each Grover method individually."""

    from utils.circuit_metrics import print_circuit_summary

    print("\n" + "="*80)
    print("INDIVIDUAL METHOD DEMONSTRATIONS")
    print("="*80 + "\n")

    n_qubits = 3
    marked_state = [5]  # Search for state |101⟩
    num_iterations = 2

    print(f"Problem: Search {2**n_qubits} items for state {marked_state}")
    print(f"Using {num_iterations} Grover iterations\n")

    # 1. Standard Grover
    print("1. STANDARD GROVER'S ALGORITHM")
    print("-" * 40)
    standard = StandardGrover(n_qubits, marked_state)
    standard_circ = standard.build_circuit(num_iterations)
    print_circuit_summary(standard_circ, "Standard Grover")
    print(f"Success probability: {standard.success_probability(num_iterations):.4f}")

    # 2. Grover via Taylor
    print("\n2. GROVER VIA TAYLOR-LCU")
    print("-" * 40)
    taylor = GroverViaTaylor(n_qubits, marked_state)
    taylor_circ = taylor.build_circuit(num_iterations, truncation_order=8)
    print_circuit_summary(taylor_circ, "Taylor-LCU Grover")
    print(f"Estimated error: {taylor.estimate_error(8, num_iterations):.2e}")

    # 3. Grover via QSVT
    print("\n3. GROVER VIA QSVT")
    print("-" * 40)
    qsvt = GroverViaQSVT(n_qubits, marked_state)
    qsvt_circ = qsvt.build_circuit(num_iterations, polynomial_degree=8)
    print_circuit_summary(qsvt_circ, "QSVT Grover")
    print(f"Estimated error: {qsvt.estimate_error(8, num_iterations):.2e}")


if __name__ == "__main__":
    # Main comparison
    compare_grover_implementations()

    # Scaling analysis
    scaling_analysis()

    # Individual demonstrations
    demo_individual_methods()

    print("\n" + "="*80)
    print("ALL GROVER EXAMPLES COMPLETED!")
    print("="*80 + "\n")
