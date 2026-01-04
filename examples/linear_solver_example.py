"""
Example: QSVT-based Quantum Linear System Solver

This script demonstrates the QSVT linear solver implementation,
including circuit construction, verification, and performance evaluation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.linear_solver_qsvt import QSVTLinearSolver, solve_linear_system_qsvt
from utils.circuit_metrics import print_circuit_summary, analyze_circuit


def example_simple_2x2():
    """Solve a simple 2x2 linear system."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple 2x2 System")
    print("="*80)
    
    # Define system: [[2, 1], [1, 2]] * x = [3, 3]
    # Solution should be x = [1, 1]
    A = np.array([[2, 1], [1, 2]], dtype=complex)
    b = np.array([3, 3], dtype=complex)
    
    print("\nLinear System: Ax = b")
    print(f"Matrix A:\n{A}")
    print(f"Vector b: {b}")
    
    # Create solver
    solver = QSVTLinearSolver(A, b)
    
    print(f"\nMatrix Properties:")
    print(f"  Dimension: {solver.n}x{solver.n}")
    print(f"  Condition number: {solver.condition_number:.4f}")
    print(f"  Spectral norm: {solver.spectral_norm:.4f}")
    
    # Build quantum circuit
    polynomial_degree = 10
    circuit = solver.build_circuit(polynomial_degree)
    
    print(f"\nQuantum Circuit:")
    print(f"  Polynomial degree: {polynomial_degree}")
    print(f"  System qubits: {solver.n_qubits}")
    print(f"  Ancilla qubits: {solver.n_ancilla}")
    print(f"  Total qubits: {circuit.num_qubits}")
    print(f"  Query complexity: {solver.get_query_complexity(polynomial_degree)}")
    
    # Get classical solution for comparison
    x_classical = solver.solve()
    print(f"\nClassical Solution: {x_classical}")
    
    # Verify solution
    residual = A @ x_classical - b
    print(f"Residual norm: {np.linalg.norm(residual):.2e}")
    
    # Print detailed circuit metrics
    print_circuit_summary(circuit, "QSVT Linear Solver")
    
    return circuit, solver


def example_diagonal_matrix():
    """Solve a diagonal system."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Diagonal Matrix System")
    print("="*80)
    
    # Diagonal system: diag([4, 3, 2, 1]) * x = [4, 6, 4, 2]
    # Solution: x = [1, 2, 2, 2]
    A = np.diag([4, 3, 2, 1])
    b = np.array([4, 6, 4, 2])
    
    print(f"\nDiagonal Matrix A:\n{A}")
    print(f"Vector b: {b}")
    
    solver = QSVTLinearSolver(A, b)
    
    print(f"\nMatrix Properties:")
    print(f"  Dimension: {solver.n}x{solver.n}")
    print(f"  Condition number: {solver.condition_number:.4f}")
    print(f"  Singular values: {solver.singular_values}")
    
    # Build circuit
    degree = 15
    circuit = solver.build_circuit(degree)
    
    print(f"\nCircuit with polynomial degree {degree}:")
    print(f"  Total qubits: {circuit.num_qubits}")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Gate count: {circuit.size()}")
    
    # Solve and verify
    x_solution = solver.solve()
    print(f"\nExpected solution: [1, 2, 2, 2]")
    print(f"Computed solution: {x_solution}")
    
    verification = A @ x_solution
    print(f"Verification (Ax): {verification}")
    print(f"Error: {np.linalg.norm(verification - b):.2e}")
    
    return circuit, solver


def example_ill_conditioned():
    """Demonstrate behavior with ill-conditioned matrix."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Ill-Conditioned System")
    print("="*80)
    
    # Ill-conditioned system
    A = np.array([[1, 0.99], [0.99, 0.98]])
    b = np.array([1, 1])
    
    print(f"\nMatrix A:\n{A}")
    print(f"Vector b: {b}")
    
    solver = QSVTLinearSolver(A, b)
    
    print(f"\nMatrix Properties:")
    print(f"  Condition number: {solver.condition_number:.2f}")
    print(f"  Spectral norm: {solver.spectral_norm:.4f}")
    
    # Required degree depends on condition number
    required_degree = solver._estimate_required_degree(target_error=1e-3)
    print(f"\nRequired polynomial degree for ε=1e-3: {required_degree}")
    
    # Build circuit with estimated degree
    circuit = solver.build_circuit(required_degree, cutoff=0.05)
    
    metrics = analyze_circuit(circuit)
    print(f"\nCircuit Metrics:")
    print(f"  Depth: {metrics['depth']}")
    print(f"  Total gates: {metrics['total_gates']}")
    print(f"  CNOT count: {metrics['cnot_count']}")
    
    # Classical solution
    x_solution = solver.solve()
    print(f"\nSolution: {x_solution}")
    print(f"Residual: {np.linalg.norm(A @ x_solution - b):.2e}")
    
    return circuit, solver


def compare_polynomial_degrees():
    """Compare circuit complexity for different polynomial degrees."""
    print("\n" + "="*80)
    print("COMPARISON: Polynomial Degree Scaling")
    print("="*80)
    
    A = np.array([[3, 1], [1, 2]], dtype=complex)
    b = np.array([1, 1], dtype=complex)
    
    solver = QSVTLinearSolver(A, b)
    
    degrees = [5, 10, 15, 20, 25]
    depths = []
    gate_counts = []
    cnot_counts = []
    errors = []
    
    print(f"\nSystem: 2x2, condition number = {solver.condition_number:.2f}")
    print("\nDegree | Depth | Gates | CNOTs | Est. Error")
    print("-" * 50)
    
    for degree in degrees:
        circuit = solver.build_circuit(degree)
        metrics = analyze_circuit(circuit)
        error = solver.estimate_error(degree)
        
        depths.append(metrics['depth'])
        gate_counts.append(metrics['total_gates'])
        cnot_counts.append(metrics['cnot_count'])
        errors.append(error)
        
        print(f"{degree:6d} | {metrics['depth']:5d} | {metrics['total_gates']:5d} | "
              f"{metrics['cnot_count']:5d} | {error:.2e}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(degrees, depths, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Polynomial Degree')
    axes[0, 0].set_ylabel('Circuit Depth')
    axes[0, 0].set_title('Circuit Depth vs Polynomial Degree')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(degrees, gate_counts, 's-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Polynomial Degree')
    axes[0, 1].set_ylabel('Total Gate Count')
    axes[0, 1].set_title('Gate Count vs Polynomial Degree')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(degrees, cnot_counts, '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Polynomial Degree')
    axes[1, 0].set_ylabel('CNOT Count')
    axes[1, 0].set_title('CNOT Count vs Polynomial Degree')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(degrees, errors, 'd-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Polynomial Degree')
    axes[1, 1].set_ylabel('Estimated Error')
    axes[1, 1].set_title('Approximation Error vs Polynomial Degree')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_solver_degree_scaling.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: linear_solver_degree_scaling.png")


def compare_system_sizes():
    """Compare how circuit scales with system size."""
    print("\n" + "="*80)
    print("COMPARISON: System Size Scaling")
    print("="*80)
    
    sizes = [2, 4, 8]
    polynomial_degree = 10
    
    depths = []
    gate_counts = []
    qubit_counts = []
    
    print(f"\nPolynomial degree: {polynomial_degree}")
    print("\nSize | Qubits | Depth | Gates | Condition #")
    print("-" * 55)
    
    for n in sizes:
        # Create diagonal matrix with decreasing values
        A = np.diag(np.linspace(n, 1, n))
        b = np.ones(n)
        
        solver = QSVTLinearSolver(A, b)
        circuit = solver.build_circuit(polynomial_degree)
        metrics = analyze_circuit(circuit)
        
        depths.append(metrics['depth'])
        gate_counts.append(metrics['total_gates'])
        qubit_counts.append(metrics['num_qubits'])
        
        print(f"{n:4d} | {metrics['num_qubits']:6d} | {metrics['depth']:5d} | "
              f"{metrics['total_gates']:5d} | {solver.condition_number:11.2f}")
    
    # Plot scaling
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(sizes, qubit_counts, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('System Size (n)')
    axes[0].set_ylabel('Total Qubits')
    axes[0].set_title('Qubit Requirements vs System Size')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sizes, depths, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('System Size (n)')
    axes[1].set_ylabel('Circuit Depth')
    axes[1].set_title('Circuit Depth vs System Size')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(sizes, gate_counts, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('System Size (n)')
    axes[2].set_ylabel('Gate Count')
    axes[2].set_title('Gate Count vs System Size')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_solver_size_scaling.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: linear_solver_size_scaling.png")


def demo_convenience_function():
    """Demonstrate the convenience function."""
    print("\n" + "="*80)
    print("CONVENIENCE FUNCTION DEMO")
    print("="*80)
    
    A = np.array([[5, 2], [2, 3]], dtype=complex)
    b = np.array([1, 2], dtype=complex)
    
    print(f"\nUsing solve_linear_system_qsvt() function:")
    print(f"Matrix A:\n{A}")
    print(f"Vector b: {b}")
    
    # Use convenience function
    circuit, metadata = solve_linear_system_qsvt(A, b, polynomial_degree=15)
    
    print(f"\nMetadata returned:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nCircuit stats:")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Gates: {circuit.size()}")


def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("QSVT QUANTUM LINEAR SYSTEM SOLVER - EXAMPLES")
    print("="*80)
    
    # Example 1: Simple 2x2 system
    example_simple_2x2()
    
    # Example 2: Diagonal matrix
    example_diagonal_matrix()
    
    # Example 3: Ill-conditioned matrix
    example_ill_conditioned()
    
    # Comparison: Polynomial degree scaling
    compare_polynomial_degrees()
    
    # Comparison: System size scaling
    compare_system_sizes()
    
    # Convenience function demo
    demo_convenience_function()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
