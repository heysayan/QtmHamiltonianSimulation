"""
Benchmark suite for QSVT Linear System Solver.

Integrates the linear solver with the existing benchmark framework,
evaluating it using the same metrics as Hamiltonian simulation algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import time
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.linear_solver_qsvt import QSVTLinearSolver
from utils.circuit_metrics import analyze_circuit


@dataclass
class LinearSolverBenchmarkResult:
    """Container for linear solver benchmark results."""
    system_size: int
    condition_number: float
    polynomial_degree: int
    num_qubits: int
    circuit_depth: int
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    cnot_count: int
    t_count: int
    construction_time: float
    estimated_error: float
    query_complexity: int
    ancilla_qubits: int


class LinearSolverBenchmark:
    """
    Benchmark suite for QSVT linear system solver.
    
    Evaluates circuit complexity, scalability, and performance metrics.
    """
    
    def __init__(self, matrix_A: np.ndarray, vector_b: np.ndarray):
        """
        Initialize benchmark suite.
        
        Args:
            matrix_A: Coefficient matrix
            vector_b: Right-hand side vector
        """
        self.A = matrix_A
        self.b = vector_b
        self.solver = QSVTLinearSolver(matrix_A, vector_b)
        self.results = []
    
    def benchmark_with_degree(self, polynomial_degree: int, cutoff: float = 0.1):
        """
        Benchmark solver with specific polynomial degree.
        
        Args:
            polynomial_degree: Degree of polynomial approximation
            cutoff: Regularization cutoff
        """
        print(f"\nBenchmarking with polynomial degree {polynomial_degree}...")
        
        start_time = time.time()
        circuit = self.solver.build_circuit(polynomial_degree, cutoff)
        construction_time = time.time() - start_time
        
        metrics = analyze_circuit(circuit)
        estimated_error = self.solver.estimate_error(polynomial_degree, cutoff)
        query_complexity = self.solver.get_query_complexity(polynomial_degree)
        
        n_ancilla = metrics['num_qubits'] - self.solver.n_qubits
        
        result = LinearSolverBenchmarkResult(
            system_size=self.solver.n,
            condition_number=self.solver.condition_number,
            polynomial_degree=polynomial_degree,
            num_qubits=metrics['num_qubits'],
            circuit_depth=metrics['depth'],
            total_gates=metrics['total_gates'],
            single_qubit_gates=metrics['single_qubit_gates'],
            two_qubit_gates=metrics['two_qubit_gates'],
            cnot_count=metrics['cnot_count'],
            t_count=metrics['t_count'],
            construction_time=construction_time,
            estimated_error=estimated_error,
            query_complexity=query_complexity,
            ancilla_qubits=n_ancilla
        )
        
        self.results.append(result)
        self._print_result(result)
        
        return result
    
    def run_degree_sweep(self, degrees: List[int]):
        """
        Run benchmark across multiple polynomial degrees.
        
        Args:
            degrees: List of polynomial degrees to test
        """
        print("="*80)
        print("POLYNOMIAL DEGREE SWEEP")
        print("="*80)
        print(f"System size: {self.solver.n}x{self.solver.n}")
        print(f"Condition number: {self.solver.condition_number:.2f}")
        print("="*80)
        
        for degree in degrees:
            self.benchmark_with_degree(degree)
        
        return self._create_comparison_dataframe()
    
    def _print_result(self, result: LinearSolverBenchmarkResult):
        """Print individual benchmark result."""
        print(f"  Qubits: {result.num_qubits} "
              f"(system: {result.num_qubits - result.ancilla_qubits}, "
              f"ancilla: {result.ancilla_qubits})")
        print(f"  Depth: {result.circuit_depth}")
        print(f"  Gates: {result.total_gates} ({result.cnot_count} CNOTs)")
        print(f"  Query Complexity: {result.query_complexity}")
        print(f"  Estimated Error: {result.estimated_error:.2e}")
        print(f"  Construction time: {result.construction_time:.4f}s")
    
    def _create_comparison_dataframe(self) -> pd.DataFrame:
        """Create comparison DataFrame from results."""
        data = []
        for result in self.results:
            data.append({
                'Polynomial Degree': result.polynomial_degree,
                'Total Qubits': result.num_qubits,
                'System Qubits': result.num_qubits - result.ancilla_qubits,
                'Ancilla Qubits': result.ancilla_qubits,
                'Circuit Depth': result.circuit_depth,
                'Total Gates': result.total_gates,
                '1-Qubit Gates': result.single_qubit_gates,
                '2-Qubit Gates': result.two_qubit_gates,
                'CNOT Count': result.cnot_count,
                'T Count': result.t_count,
                'Query Complexity': result.query_complexity,
                'Estimated Error': result.estimated_error,
                'Construction Time (s)': result.construction_time,
            })
        
        return pd.DataFrame(data)
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Create comparison plots.
        
        Args:
            save_path: Path to save plot (if None, display only)
        """
        if not self.results:
            print("No results to plot. Run benchmarks first.")
            return
        
        df = self._create_comparison_dataframe()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QSVT Linear Solver Performance vs Polynomial Degree',
                     fontsize=16, fontweight='bold')
        
        # Circuit Depth
        axes[0, 0].plot(df['Polynomial Degree'], df['Circuit Depth'],
                       'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Polynomial Degree')
        axes[0, 0].set_ylabel('Circuit Depth')
        axes[0, 0].set_title('Circuit Depth')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total Gates
        axes[0, 1].plot(df['Polynomial Degree'], df['Total Gates'],
                       's-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Polynomial Degree')
        axes[0, 1].set_ylabel('Total Gates')
        axes[0, 1].set_title('Gate Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # CNOT Count
        axes[0, 2].plot(df['Polynomial Degree'], df['CNOT Count'],
                       '^-', linewidth=2, markersize=8, color='green')
        axes[0, 2].set_xlabel('Polynomial Degree')
        axes[0, 2].set_ylabel('CNOT Count')
        axes[0, 2].set_title('CNOT Gates')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Query Complexity
        axes[1, 0].plot(df['Polynomial Degree'], df['Query Complexity'],
                       'd-', linewidth=2, markersize=8, color='purple')
        axes[1, 0].set_xlabel('Polynomial Degree')
        axes[1, 0].set_ylabel('Query Complexity')
        axes[1, 0].set_title('Query Complexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Estimated Error (log scale)
        axes[1, 1].semilogy(df['Polynomial Degree'], df['Estimated Error'],
                           'v-', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_xlabel('Polynomial Degree')
        axes[1, 1].set_ylabel('Estimated Error')
        axes[1, 1].set_title('Approximation Error (log scale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Qubit Usage
        system_qubits = df['System Qubits']
        ancilla_qubits = df['Ancilla Qubits']
        x_pos = range(len(df))
        axes[1, 2].bar(x_pos, system_qubits, label='System', color='steelblue')
        axes[1, 2].bar(x_pos, ancilla_qubits, bottom=system_qubits,
                      label='Ancilla', color='orange')
        axes[1, 2].set_xlabel('Polynomial Degree')
        axes[1, 2].set_ylabel('Qubits')
        axes[1, 2].set_title('Qubit Usage')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(df['Polynomial Degree'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()
    
    def print_summary_table(self):
        """Print formatted summary table."""
        df = self._create_comparison_dataframe()
        
        print("\n" + "="*120)
        print("BENCHMARK SUMMARY")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")


class SystemSizeBenchmark:
    """Benchmark linear solver scaling with system size."""
    
    def __init__(self):
        """Initialize system size benchmark."""
        self.results = []
    
    def run_size_sweep(
        self,
        sizes: List[int],
        polynomial_degree: int = 10,
        matrix_type: str = "diagonal"
    ):
        """
        Benchmark across different system sizes.
        
        Args:
            sizes: List of system sizes to test
            polynomial_degree: Fixed polynomial degree
            matrix_type: Type of test matrix ("diagonal", "random", "ill_conditioned")
        """
        print("="*80)
        print("SYSTEM SIZE SCALING BENCHMARK")
        print("="*80)
        print(f"Matrix type: {matrix_type}")
        print(f"Polynomial degree: {polynomial_degree}")
        print("="*80)
        
        for n in sizes:
            print(f"\nSystem size: {n}x{n}")
            
            # Create test matrix
            A, b = self._create_test_system(n, matrix_type)
            
            # Run benchmark
            benchmark = LinearSolverBenchmark(A, b)
            result = benchmark.benchmark_with_degree(polynomial_degree)
            
            self.results.append({
                'size': n,
                'result': result
            })
        
        return self._create_size_dataframe()
    
    def _create_test_system(self, n: int, matrix_type: str):
        """Create test linear system."""
        if matrix_type == "diagonal":
            A = np.diag(np.linspace(n, 1, n))
            b = np.ones(n)
        elif matrix_type == "random":
            A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            A = (A + A.conj().T) / 2  # Make Hermitian
            b = np.random.randn(n)
        elif matrix_type == "ill_conditioned":
            # Create matrix with large condition number
            U, _ = np.linalg.qr(np.random.randn(n, n))
            singular_values = np.logspace(0, -3, n)  # κ ≈ 1000
            A = U @ np.diag(singular_values) @ U.T
            b = np.ones(n)
        else:
            # Default: identity-like
            A = np.eye(n) + np.random.randn(n, n) * 0.1
            b = np.ones(n)
        
        return A, b
    
    def _create_size_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from size scaling results."""
        data = []
        for item in self.results:
            result = item['result']
            data.append({
                'System Size': item['size'],
                'Condition Number': result.condition_number,
                'Total Qubits': result.num_qubits,
                'Circuit Depth': result.circuit_depth,
                'Total Gates': result.total_gates,
                'CNOT Count': result.cnot_count,
                'Query Complexity': result.query_complexity,
                'Estimated Error': result.estimated_error,
            })
        
        return pd.DataFrame(data)
    
    def plot_scaling(self, save_path: Optional[str] = None):
        """Plot system size scaling."""
        if not self.results:
            print("No results to plot.")
            return
        
        df = self._create_size_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Linear Solver Scaling with System Size',
                     fontsize=16, fontweight='bold')
        
        # Qubits
        axes[0, 0].plot(df['System Size'], df['Total Qubits'],
                       'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('System Size (n)')
        axes[0, 0].set_ylabel('Total Qubits')
        axes[0, 0].set_title('Qubit Requirements')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Depth
        axes[0, 1].plot(df['System Size'], df['Circuit Depth'],
                       's-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('System Size (n)')
        axes[0, 1].set_ylabel('Circuit Depth')
        axes[0, 1].set_title('Circuit Depth')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gates
        axes[1, 0].plot(df['System Size'], df['Total Gates'],
                       '^-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('System Size (n)')
        axes[1, 0].set_ylabel('Total Gates')
        axes[1, 0].set_title('Gate Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Condition number
        axes[1, 1].semilogy(df['System Size'], df['Condition Number'],
                           'd-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('System Size (n)')
        axes[1, 1].set_ylabel('Condition Number')
        axes[1, 1].set_title('Matrix Condition Number')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")


def run_comprehensive_linear_solver_benchmark():
    """Run comprehensive benchmark of linear solver."""
    print("\n" + "="*80)
    print("COMPREHENSIVE QSVT LINEAR SOLVER BENCHMARK")
    print("="*80 + "\n")
    
    # Test 1: Polynomial degree sweep
    print("\n--- TEST 1: Polynomial Degree Sweep ---")
    A = np.array([[3, 1], [1, 2]], dtype=complex)
    b = np.array([1, 1], dtype=complex)
    
    benchmark = LinearSolverBenchmark(A, b)
    df_degrees = benchmark.run_degree_sweep([5, 10, 15, 20, 25])
    benchmark.print_summary_table()
    benchmark.plot_comparison(save_path='linear_solver_degree_comparison.png')
    
    # Save results
    df_degrees.to_csv('linear_solver_degree_results.csv', index=False)
    print("Results saved to: linear_solver_degree_results.csv")
    
    # Test 2: System size scaling
    print("\n--- TEST 2: System Size Scaling ---")
    size_benchmark = SystemSizeBenchmark()
    df_sizes = size_benchmark.run_size_sweep([2, 4, 8], polynomial_degree=10)
    
    print("\n" + "="*80)
    print("SIZE SCALING RESULTS")
    print("="*80)
    print(df_sizes.to_string(index=False))
    print("="*80)
    
    size_benchmark.plot_scaling(save_path='linear_solver_size_scaling.png')
    df_sizes.to_csv('linear_solver_size_results.csv', index=False)
    print("\nResults saved to: linear_solver_size_results.csv")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_comprehensive_linear_solver_benchmark()
