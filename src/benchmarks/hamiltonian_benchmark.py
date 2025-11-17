"""
Comprehensive benchmarking suite for Hamiltonian simulation algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.trotterization import TrotterizationSimulator
from algorithms.taylor_lcu import TaylorLCUSimulator
from algorithms.qsp import QSPSimulator, QubitizationSimulator
from algorithms.qsvt import QSVTSimulator
from utils.circuit_metrics import analyze_circuit, compare_circuits


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    algorithm_name: str
    num_qubits: int
    circuit_depth: int
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    cnot_count: int
    t_count: int
    construction_time: float
    estimated_error: float
    query_complexity: Optional[int] = None
    ancilla_qubits: Optional[int] = None


class HamiltonianBenchmark:
    """
    Benchmark suite for comparing Hamiltonian simulation algorithms.
    """

    def __init__(self, hamiltonian: SparsePauliOp, time: float):
        """
        Initialize benchmark suite.

        Args:
            hamiltonian: Hamiltonian to simulate
            time: Evolution time
        """
        self.hamiltonian = hamiltonian
        self.time = time
        self.n_qubits = hamiltonian.num_qubits
        self.results = []

    def run_all_benchmarks(
        self,
        trotter_steps: int = 10,
        taylor_order: int = 10,
        qsp_degree: int = 10,
        qsvt_degree: int = 10
    ) -> pd.DataFrame:
        """
        Run all Hamiltonian simulation algorithms and compare.

        Args:
            trotter_steps: Number of Trotter steps
            taylor_order: Taylor series truncation order
            qsp_degree: QSP polynomial degree
            qsvt_degree: QSVT polynomial degree

        Returns:
            DataFrame with comparison results
        """
        print("="*80)
        print("HAMILTONIAN SIMULATION BENCHMARK")
        print("="*80)
        print(f"Hamiltonian: {self.n_qubits} qubits, {len(self.hamiltonian.paulis)} terms")
        print(f"Evolution time: {self.time}")
        print("="*80 + "\n")

        # Run each algorithm
        print("Running First-Order Trotterization...")
        self.benchmark_trotter(order=1, n_steps=trotter_steps)

        print("Running Second-Order Trotterization...")
        self.benchmark_trotter(order=2, n_steps=trotter_steps)

        print("Running Taylor-LCU...")
        self.benchmark_taylor_lcu(truncation_order=taylor_order)

        print("Running QSP...")
        self.benchmark_qsp(degree=qsp_degree)

        print("Running Qubitization...")
        self.benchmark_qubitization()

        print("Running QSVT...")
        self.benchmark_qsvt(degree=qsvt_degree)

        # Create comparison DataFrame
        df = self._create_comparison_dataframe()

        return df

    def benchmark_trotter(self, order: int, n_steps: int):
        """Benchmark Trotterization algorithm."""
        simulator = TrotterizationSimulator(self.hamiltonian, self.time, order=order)

        start_time = time.time()
        circuit = simulator.build_circuit(n_steps)
        construction_time = time.time() - start_time

        metrics = analyze_circuit(circuit)
        estimated_error = simulator.estimate_error(n_steps)

        result = BenchmarkResult(
            algorithm_name=f"Trotter (order {order})",
            num_qubits=metrics['num_qubits'],
            circuit_depth=metrics['depth'],
            total_gates=metrics['total_gates'],
            single_qubit_gates=metrics['single_qubit_gates'],
            two_qubit_gates=metrics['two_qubit_gates'],
            cnot_count=metrics['cnot_count'],
            t_count=metrics['t_count'],
            construction_time=construction_time,
            estimated_error=estimated_error,
            query_complexity=n_steps,
            ancilla_qubits=0
        )

        self.results.append(result)
        self._print_result(result)

    def benchmark_taylor_lcu(self, truncation_order: int):
        """Benchmark Taylor-LCU algorithm."""
        simulator = TaylorLCUSimulator(self.hamiltonian, self.time)

        start_time = time.time()
        circuit = simulator.build_circuit(truncation_order)
        construction_time = time.time() - start_time

        metrics = analyze_circuit(circuit)
        estimated_error = simulator.estimate_error(truncation_order)

        n_ancilla = metrics['num_qubits'] - self.n_qubits

        result = BenchmarkResult(
            algorithm_name="Taylor-LCU",
            num_qubits=metrics['num_qubits'],
            circuit_depth=metrics['depth'],
            total_gates=metrics['total_gates'],
            single_qubit_gates=metrics['single_qubit_gates'],
            two_qubit_gates=metrics['two_qubit_gates'],
            cnot_count=metrics['cnot_count'],
            t_count=metrics['t_count'],
            construction_time=construction_time,
            estimated_error=estimated_error,
            query_complexity=truncation_order,
            ancilla_qubits=n_ancilla
        )

        self.results.append(result)
        self._print_result(result)

    def benchmark_qsp(self, degree: int):
        """Benchmark QSP algorithm."""
        simulator = QSPSimulator(self.hamiltonian, self.time)

        start_time = time.time()
        circuit = simulator.build_circuit(degree)
        construction_time = time.time() - start_time

        metrics = analyze_circuit(circuit)
        estimated_error = simulator.estimate_error(degree)

        n_ancilla = metrics['num_qubits'] - self.n_qubits

        result = BenchmarkResult(
            algorithm_name="QSP",
            num_qubits=metrics['num_qubits'],
            circuit_depth=metrics['depth'],
            total_gates=metrics['total_gates'],
            single_qubit_gates=metrics['single_qubit_gates'],
            two_qubit_gates=metrics['two_qubit_gates'],
            cnot_count=metrics['cnot_count'],
            t_count=metrics['t_count'],
            construction_time=construction_time,
            estimated_error=estimated_error,
            query_complexity=degree,
            ancilla_qubits=n_ancilla
        )

        self.results.append(result)
        self._print_result(result)

    def benchmark_qubitization(self):
        """Benchmark Qubitization algorithm."""
        simulator = QubitizationSimulator(self.hamiltonian, self.time)

        query_complexity = simulator._compute_query_complexity()

        start_time = time.time()
        circuit = simulator.build_circuit(query_complexity)
        construction_time = time.time() - start_time

        metrics = analyze_circuit(circuit)
        estimated_error = simulator.estimate_error(query_complexity)

        n_ancilla = metrics['num_qubits'] - self.n_qubits

        result = BenchmarkResult(
            algorithm_name="Qubitization",
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

    def benchmark_qsvt(self, degree: int):
        """Benchmark QSVT algorithm."""
        simulator = QSVTSimulator(self.hamiltonian, self.time)

        start_time = time.time()
        circuit = simulator.build_circuit(degree)
        construction_time = time.time() - start_time

        metrics = analyze_circuit(circuit)
        estimated_error = simulator.estimate_error(degree)
        query_complexity = simulator.get_query_complexity(degree)

        n_ancilla = metrics['num_qubits'] - self.n_qubits

        result = BenchmarkResult(
            algorithm_name="QSVT",
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

    def _print_result(self, result: BenchmarkResult):
        """Print individual benchmark result."""
        print(f"\n{result.algorithm_name}:")
        print(f"  Qubits: {result.num_qubits} (+ {result.ancilla_qubits or 0} ancilla)")
        print(f"  Depth: {result.circuit_depth}")
        print(f"  Gates: {result.total_gates} ({result.cnot_count} CNOTs)")
        print(f"  Error: {result.estimated_error:.2e}")
        print(f"  Construction time: {result.construction_time:.4f}s")

    def _create_comparison_dataframe(self) -> pd.DataFrame:
        """Create comparison DataFrame from results."""
        data = []
        for result in self.results:
            data.append({
                'Algorithm': result.algorithm_name,
                'Total Qubits': result.num_qubits,
                'System Qubits': self.n_qubits,
                'Ancilla Qubits': result.ancilla_qubits or 0,
                'Circuit Depth': result.circuit_depth,
                'Total Gates': result.total_gates,
                '1-Qubit Gates': result.single_qubit_gates,
                '2-Qubit Gates': result.two_qubit_gates,
                'CNOT Count': result.cnot_count,
                'T Count': result.t_count,
                'Query Complexity': result.query_complexity or 0,
                'Estimated Error': result.estimated_error,
                'Construction Time (s)': result.construction_time,
            })

        df = pd.DataFrame(data)
        return df

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Create comparison plots.

        Args:
            save_path: Path to save the plot (if None, display only)
        """
        if not self.results:
            print("No results to plot. Run benchmarks first.")
            return

        df = self._create_comparison_dataframe()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hamiltonian Simulation Algorithm Comparison', fontsize=16, fontweight='bold')

        # Circuit Depth
        axes[0, 0].bar(df['Algorithm'], df['Circuit Depth'], color='skyblue')
        axes[0, 0].set_title('Circuit Depth')
        axes[0, 0].set_ylabel('Depth')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Total Gates
        axes[0, 1].bar(df['Algorithm'], df['Total Gates'], color='lightcoral')
        axes[0, 1].set_title('Total Gate Count')
        axes[0, 1].set_ylabel('Gates')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # CNOT Count
        axes[0, 2].bar(df['Algorithm'], df['CNOT Count'], color='lightgreen')
        axes[0, 2].set_title('CNOT Gate Count')
        axes[0, 2].set_ylabel('CNOTs')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Qubit Usage (stacked: system + ancilla)
        system_qubits = df['System Qubits']
        ancilla_qubits = df['Ancilla Qubits']
        axes[1, 0].bar(df['Algorithm'], system_qubits, label='System', color='steelblue')
        axes[1, 0].bar(df['Algorithm'], ancilla_qubits, bottom=system_qubits,
                       label='Ancilla', color='orange')
        axes[1, 0].set_title('Qubit Usage')
        axes[1, 0].set_ylabel('Qubits')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Estimated Error (log scale)
        axes[1, 1].bar(df['Algorithm'], df['Estimated Error'], color='mediumpurple')
        axes[1, 1].set_title('Estimated Error')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Query Complexity
        axes[1, 2].bar(df['Algorithm'], df['Query Complexity'], color='gold')
        axes[1, 2].set_title('Query Complexity')
        axes[1, 2].set_ylabel('Queries')
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()

    def print_summary_table(self):
        """Print a formatted summary table."""
        df = self._create_comparison_dataframe()

        print("\n" + "="*120)
        print("BENCHMARK SUMMARY")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")

        # Print winners in each category
        print("Best Performers:")
        print(f"  Lowest Circuit Depth:  {df.loc[df['Circuit Depth'].idxmin(), 'Algorithm']}")
        print(f"  Fewest Total Gates:    {df.loc[df['Total Gates'].idxmin(), 'Algorithm']}")
        print(f"  Fewest CNOTs:          {df.loc[df['CNOT Count'].idxmin(), 'Algorithm']}")
        print(f"  Lowest Error:          {df.loc[df['Estimated Error'].idxmin(), 'Algorithm']}")
        print(f"  Fewest Qubits:         {df.loc[df['Total Qubits'].idxmin(), 'Algorithm']}")
        print()


def run_comprehensive_benchmark(
    hamiltonian: SparsePauliOp,
    time: float,
    plot_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Run comprehensive benchmark of all algorithms.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        plot_path: Optional path to save comparison plot

    Returns:
        DataFrame with benchmark results
    """
    benchmark = HamiltonianBenchmark(hamiltonian, time)
    df = benchmark.run_all_benchmarks()
    benchmark.print_summary_table()
    benchmark.plot_comparison(save_path=plot_path)

    return df
