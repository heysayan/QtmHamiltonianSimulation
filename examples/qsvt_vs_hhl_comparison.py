"""
Comparison between QSVT and HHL Linear System Solvers.

This script compares the QSVT-based and HHL algorithms for solving
linear systems Ax=b using the same metrics used throughout the repository.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from algorithms.linear_solver_qsvt import QSVTLinearSolver
from algorithms.linear_solver_hhl import HHLSolver
from utils.circuit_metrics import analyze_circuit


def compare_algorithms(A: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """
    Compare QSVT and HHL algorithms on the same linear system.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("QSVT vs HHL: Linear System Solver Comparison")
    print("="*80)
    print(f"\nMatrix dimensions: {A.shape[0]}×{A.shape[1]}")
    print(f"Vector b: {b}")
    
    results = []
    
    # QSVT Solver
    print("\n--- QSVT Solver ---")
    qsvt_solver = QSVTLinearSolver(A, b)
    print(f"Condition number: {qsvt_solver.condition_number:.4f}")
    
    # Build circuit with degree 10
    degree = 10
    start_time = time.time()
    qsvt_circuit = qsvt_solver.build_circuit(polynomial_degree=degree)
    qsvt_time = time.time() - start_time
    
    qsvt_metrics = analyze_circuit(qsvt_circuit)
    qsvt_error = qsvt_solver.estimate_error(degree)
    qsvt_queries = qsvt_solver.get_query_complexity(degree)
    
    results.append({
        'Algorithm': 'QSVT',
        'Total Qubits': qsvt_metrics['num_qubits'],
        'System Qubits': qsvt_solver.n_qubits,
        'Ancilla Qubits': qsvt_metrics['num_qubits'] - qsvt_solver.n_qubits,
        'Circuit Depth': qsvt_metrics['depth'],
        'Total Gates': qsvt_metrics['total_gates'],
        'Single-Qubit Gates': qsvt_metrics['single_qubit_gates'],
        'Two-Qubit Gates': qsvt_metrics['two_qubit_gates'],
        'CNOT Count': qsvt_metrics['cnot_count'],
        'T Count': qsvt_metrics['t_count'],
        'Query Complexity': qsvt_queries,
        'Estimated Error': qsvt_error,
        'Construction Time (s)': qsvt_time,
        'Complexity': f"O(κ log(1/ε))",
    })
    
    print(f"Qubits: {qsvt_metrics['num_qubits']}")
    print(f"Depth: {qsvt_metrics['depth']}")
    print(f"Gates: {qsvt_metrics['total_gates']}")
    print(f"Query Complexity: {qsvt_queries}")
    print(f"Error: {qsvt_error:.2e}")
    
    # HHL Solver
    print("\n--- HHL Solver ---")
    hhl_solver = HHLSolver(A, b)
    print(f"Condition number: {hhl_solver.condition_number:.4f}")
    
    start_time = time.time()
    hhl_circuit = hhl_solver.build_circuit(use_qpe=True)
    hhl_time = time.time() - start_time
    
    hhl_metrics = analyze_circuit(hhl_circuit)
    hhl_error = hhl_solver.estimate_error()
    hhl_queries = hhl_solver.get_query_complexity()
    
    results.append({
        'Algorithm': 'HHL',
        'Total Qubits': hhl_metrics['num_qubits'],
        'System Qubits': hhl_solver.n_qubits,
        'Ancilla Qubits': hhl_metrics['num_qubits'] - hhl_solver.n_qubits,
        'Circuit Depth': hhl_metrics['depth'],
        'Total Gates': hhl_metrics['total_gates'],
        'Single-Qubit Gates': hhl_metrics['single_qubit_gates'],
        'Two-Qubit Gates': hhl_metrics['two_qubit_gates'],
        'CNOT Count': hhl_metrics['cnot_count'],
        'T Count': hhl_metrics['t_count'],
        'Query Complexity': hhl_queries,
        'Estimated Error': hhl_error,
        'Construction Time (s)': hhl_time,
        'Complexity': f"O(log(N)κ²/ε)",
    })
    
    print(f"Qubits: {hhl_metrics['num_qubits']}")
    print(f"Depth: {hhl_metrics['depth']}")
    print(f"Gates: {hhl_metrics['total_gates']}")
    print(f"Query Complexity: {hhl_queries}")
    print(f"Error: {hhl_error:.2e}")
    
    # Classical solution for verification
    print("\n--- Classical Solution (for verification) ---")
    x_classical = qsvt_solver.solve()
    residual = np.linalg.norm(A @ x_classical - b)
    print(f"Solution: {x_classical}")
    print(f"Residual: {residual:.2e}")
    
    return pd.DataFrame(results)


def run_multiple_comparisons():
    """Run comparisons for different system sizes and conditions."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE QSVT vs HHL COMPARISON")
    print("="*80)
    
    # Test Case 1: Simple 2×2 well-conditioned
    print("\n\n" + "="*80)
    print("TEST CASE 1: 2×2 Well-Conditioned System")
    print("="*80)
    A1 = np.array([[2, 1], [1, 2]], dtype=complex)
    b1 = np.array([3, 3], dtype=complex)
    df1 = compare_algorithms(A1, b1)
    
    # Test Case 2: 2×2 ill-conditioned
    print("\n\n" + "="*80)
    print("TEST CASE 2: 2×2 Ill-Conditioned System")
    print("="*80)
    A2 = np.array([[1, 0.99], [0.99, 0.98]], dtype=complex)
    b2 = np.array([1, 1], dtype=complex)
    df2 = compare_algorithms(A2, b2)
    
    # Test Case 3: 4×4 diagonal
    print("\n\n" + "="*80)
    print("TEST CASE 3: 4×4 Diagonal System")
    print("="*80)
    A3 = np.diag([4, 3, 2, 1]).astype(complex)
    b3 = np.array([4, 6, 4, 2], dtype=complex)
    df3 = compare_algorithms(A3, b3)
    
    # Combine results
    df1['Test Case'] = '2×2 Well-Conditioned'
    df2['Test Case'] = '2×2 Ill-Conditioned'
    df3['Test Case'] = '4×4 Diagonal'
    
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df_combined.to_string(index=False))
    
    # Save to CSV
    df_combined.to_csv('qsvt_vs_hhl_comparison.csv', index=False)
    print("\nResults saved to: qsvt_vs_hhl_comparison.csv")
    
    return df_combined


def create_comparison_plots(df: pd.DataFrame):
    """
    Create visualization plots comparing QSVT and HHL.
    
    Args:
        df: DataFrame with comparison results
    """
    # Filter to just one test case for clarity
    df_plot = df[df['Test Case'] == '2×2 Well-Conditioned'].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QSVT vs HHL: Linear System Solver Comparison', 
                 fontsize=16, fontweight='bold')
    
    algorithms = df_plot['Algorithm'].values
    x_pos = np.arange(len(algorithms))
    
    # Total Qubits
    axes[0, 0].bar(x_pos, df_plot['Total Qubits'], color=['steelblue', 'orange'])
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(algorithms)
    axes[0, 0].set_ylabel('Qubits')
    axes[0, 0].set_title('Total Qubit Count')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Circuit Depth
    axes[0, 1].bar(x_pos, df_plot['Circuit Depth'], color=['steelblue', 'orange'])
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(algorithms)
    axes[0, 1].set_ylabel('Depth')
    axes[0, 1].set_title('Circuit Depth')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Total Gates
    axes[0, 2].bar(x_pos, df_plot['Total Gates'], color=['steelblue', 'orange'])
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(algorithms)
    axes[0, 2].set_ylabel('Gates')
    axes[0, 2].set_title('Total Gate Count')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # CNOT Count
    axes[1, 0].bar(x_pos, df_plot['CNOT Count'], color=['steelblue', 'orange'])
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(algorithms)
    axes[1, 0].set_ylabel('CNOTs')
    axes[1, 0].set_title('CNOT Gate Count')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Query Complexity
    axes[1, 1].bar(x_pos, df_plot['Query Complexity'], color=['steelblue', 'orange'])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(algorithms)
    axes[1, 1].set_ylabel('Queries')
    axes[1, 1].set_title('Query Complexity')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Estimated Error (log scale)
    axes[1, 2].bar(x_pos, df_plot['Estimated Error'], color=['steelblue', 'orange'])
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(algorithms)
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].set_title('Estimated Error (log scale)')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qsvt_vs_hhl_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: qsvt_vs_hhl_comparison.png")


def print_analysis():
    """Print detailed analysis of QSVT vs HHL."""
    
    print("\n\n" + "="*80)
    print("DETAILED ANALYSIS: QSVT vs HHL")
    print("="*80)
    
    print("""
**Theoretical Comparison**

| Aspect | QSVT | HHL |
|--------|------|-----|
| **Query Complexity** | O(κ log(1/ε)) | O(log(N) s² κ² / ε) |
| **Year Introduced** | 2019 | 2009 |
| **Key Innovation** | Polynomial SVT | Phase Estimation + Rotation |
| **Scaling with κ** | Linear | Quadratic |
| **Scaling with ε** | Logarithmic | Linear (in 1/ε) |
| **System Size (N)** | Implicit in encoding | Logarithmic advantage |

**Key Observations:**

1. **Query Complexity**: QSVT has better scaling with both κ and ε
   - QSVT: O(κ log(1/ε)) - linear in κ, logarithmic in 1/ε
   - HHL: O(log(N) κ²/ε) - quadratic in κ, linear in 1/ε
   - For large κ or high precision, QSVT is more efficient

2. **Circuit Structure**:
   - QSVT: Uses block encoding + signal processing rotations
   - HHL: Uses Quantum Phase Estimation (QPE) + controlled rotations
   - HHL requires more complex QPE infrastructure

3. **Qubit Overhead**:
   - QSVT: O(log N) system + O(log m) ancillas for m-term decomposition
   - HHL: O(log N) system + O(log(κ/ε)) for QPE + 1 ancilla
   - Similar scaling, but HHL's QPE adds overhead

4. **Practical Implementation**:
   - QSVT: Simpler phase computation, direct polynomial approximation
   - HHL: Requires precise QPE, more complex gate decomposition
   - QSVT is generally easier to implement with current tools

5. **Historical Context**:
   - HHL (2009): Pioneering work, first quantum linear solver
   - QSVT (2019): Unified framework, improved complexity
   - QSVT represents algorithmic advancement over HHL

**When to Use Each:**

- **QSVT**: 
  ✓ High-precision requirements (small ε)
  ✓ Ill-conditioned matrices (large κ)
  ✓ Modern quantum hardware with good gate fidelity
  ✓ When simplicity of implementation matters

- **HHL**:
  ✓ Historical/educational purposes
  ✓ When QPE infrastructure already available
  ✓ Specific applications leveraging QPE structure
  ✓ Benchmarking against classical approaches

**Conclusion:**

QSVT provides asymptotic improvements over HHL and is the preferred
modern approach for quantum linear system solving. However, HHL remains
historically important as the first quantum linear solver and demonstrates
fundamental techniques like QPE that have broad applications.
""")


if __name__ == "__main__":
    # Run comprehensive comparison
    df = run_multiple_comparisons()
    
    # Create visualization
    create_comparison_plots(df)
    
    # Print analysis
    print_analysis()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80 + "\n")
