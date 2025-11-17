"""
Grover's search algorithm implemented via Hamiltonian simulation.

This module implements Grover's algorithm using various Hamiltonian simulation
techniques including Taylor series (LCU) and QSVT.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.taylor_lcu import TaylorLCUSimulator
from algorithms.qsvt import QSVTSimulator
from grover.standard_grover import grover_to_hamiltonian, grover_hamiltonian_pauli


class GroverViaTaylor:
    """
    Grover's search via Hamiltonian simulation using Taylor series (LCU).

    The Grover operator G can be expressed as evolution under a specific
    Hamiltonian. We use Taylor series approximation to implement this evolution.
    """

    def __init__(self, n_qubits: int, marked_states: Optional[List[int]] = None):
        """
        Initialize Grover search via Taylor method.

        Args:
            n_qubits: Number of qubits
            marked_states: List of marked state indices
        """
        self.n_qubits = n_qubits
        self.marked_states = marked_states or [2**n_qubits - 1]

        # Convert Grover's problem to Hamiltonian
        self.hamiltonian = self._construct_grover_hamiltonian()

        # Optimal evolution time for Grover
        # One Grover iteration corresponds to time t = π/4
        self.grover_time = np.pi / 4

    def _construct_grover_hamiltonian(self) -> SparsePauliOp:
        """
        Construct the Hamiltonian for Grover's algorithm.

        Returns:
            SparsePauliOp representing Grover's Hamiltonian
        """
        return grover_hamiltonian_pauli(self.n_qubits, self.marked_states)

    def build_circuit(
        self,
        num_iterations: int,
        truncation_order: int = 10
    ) -> QuantumCircuit:
        """
        Build Grover circuit using Taylor series method.

        Args:
            num_iterations: Number of Grover iterations
            truncation_order: Order of Taylor truncation

        Returns:
            QuantumCircuit implementing Grover via Taylor method
        """
        # Total evolution time
        total_time = num_iterations * self.grover_time

        # Create Taylor simulator
        simulator = TaylorLCUSimulator(self.hamiltonian, total_time)

        # Build circuit
        qc = simulator.build_circuit(truncation_order)

        # Add initial superposition (not included in Hamiltonian evolution)
        qc_full = QuantumCircuit(qc.num_qubits)

        # Initialize system qubits in superposition
        for i in range(self.n_qubits):
            qc_full.h(i)

        # Apply evolution
        qc_full.compose(qc, inplace=True)

        return qc_full

    def estimate_error(self, truncation_order: int, num_iterations: int) -> float:
        """
        Estimate approximation error.

        Args:
            truncation_order: Taylor truncation order
            num_iterations: Number of iterations

        Returns:
            Estimated error
        """
        total_time = num_iterations * self.grover_time
        simulator = TaylorLCUSimulator(self.hamiltonian, total_time)
        return simulator.estimate_error(truncation_order)


class GroverViaQSVT:
    """
    Grover's search via Hamiltonian simulation using QSVT.

    QSVT provides an optimal way to implement polynomial transformations,
    which can be used to implement Grover's algorithm efficiently.
    """

    def __init__(self, n_qubits: int, marked_states: Optional[List[int]] = None):
        """
        Initialize Grover search via QSVT.

        Args:
            n_qubits: Number of qubits
            marked_states: List of marked state indices
        """
        self.n_qubits = n_qubits
        self.marked_states = marked_states or [2**n_qubits - 1]

        # Construct Grover's Hamiltonian
        self.hamiltonian = self._construct_grover_hamiltonian()

        # Grover iteration time
        self.grover_time = np.pi / 4

    def _construct_grover_hamiltonian(self) -> SparsePauliOp:
        """
        Construct the Hamiltonian for Grover's algorithm.

        Returns:
            SparsePauliOp representing Grover's Hamiltonian
        """
        return grover_hamiltonian_pauli(self.n_qubits, self.marked_states)

    def build_circuit(
        self,
        num_iterations: int,
        polynomial_degree: int = 10
    ) -> QuantumCircuit:
        """
        Build Grover circuit using QSVT.

        Args:
            num_iterations: Number of Grover iterations
            polynomial_degree: Degree of polynomial approximation

        Returns:
            QuantumCircuit implementing Grover via QSVT
        """
        # Total evolution time
        total_time = num_iterations * self.grover_time

        # Create QSVT simulator
        simulator = QSVTSimulator(self.hamiltonian, total_time)

        # Build circuit
        qc = simulator.build_circuit(polynomial_degree)

        # Add initial superposition
        qc_full = QuantumCircuit(qc.num_qubits)

        # Initialize system qubits in superposition
        for i in range(self.n_qubits):
            qc_full.h(i)

        # Apply QSVT evolution
        qc_full.compose(qc, inplace=True)

        return qc_full

    def build_optimal_circuit(
        self,
        num_iterations: int,
        target_error: float = 1e-3
    ) -> QuantumCircuit:
        """
        Build QSVT circuit with automatically determined parameters.

        Args:
            num_iterations: Number of Grover iterations
            target_error: Target error tolerance

        Returns:
            QuantumCircuit implementing Grover via QSVT
        """
        total_time = num_iterations * self.grover_time

        simulator = QSVTSimulator(self.hamiltonian, total_time)

        # Determine required degree
        degree = simulator.get_required_degree(target_error)

        return self.build_circuit(num_iterations, degree)

    def estimate_error(self, polynomial_degree: int, num_iterations: int) -> float:
        """
        Estimate approximation error.

        Args:
            polynomial_degree: Polynomial degree
            num_iterations: Number of iterations

        Returns:
            Estimated error
        """
        total_time = num_iterations * self.grover_time
        simulator = QSVTSimulator(self.hamiltonian, total_time)
        return simulator.estimate_error(polynomial_degree)


def grover_taylor_simulation(
    n_qubits: int,
    marked_states: Optional[List[int]] = None,
    num_iterations: Optional[int] = None,
    truncation_order: int = 10
) -> QuantumCircuit:
    """
    Convenience function for Grover via Taylor series.

    Args:
        n_qubits: Number of qubits
        marked_states: List of marked states
        num_iterations: Number of iterations (if None, uses optimal)
        truncation_order: Taylor truncation order

    Returns:
        QuantumCircuit implementing Grover via Taylor method
    """
    if num_iterations is None:
        # Optimal iterations
        search_space = 2 ** n_qubits
        num_marked = len(marked_states) if marked_states else 1
        num_iterations = int(np.round((np.pi / 4) * np.sqrt(search_space / num_marked)))

    grover = GroverViaTaylor(n_qubits, marked_states)
    return grover.build_circuit(num_iterations, truncation_order)


def grover_qsvt_simulation(
    n_qubits: int,
    marked_states: Optional[List[int]] = None,
    num_iterations: Optional[int] = None,
    polynomial_degree: int = 10
) -> QuantumCircuit:
    """
    Convenience function for Grover via QSVT.

    Args:
        n_qubits: Number of qubits
        marked_states: List of marked states
        num_iterations: Number of iterations (if None, uses optimal)
        polynomial_degree: Polynomial degree

    Returns:
        QuantumCircuit implementing Grover via QSVT
    """
    if num_iterations is None:
        # Optimal iterations
        search_space = 2 ** n_qubits
        num_marked = len(marked_states) if marked_states else 1
        num_iterations = int(np.round((np.pi / 4) * np.sqrt(search_space / num_marked)))

    grover = GroverViaQSVT(n_qubits, marked_states)
    return grover.build_circuit(num_iterations, polynomial_degree)


class GroverComparison:
    """
    Compare different implementations of Grover's algorithm.
    """

    def __init__(self, n_qubits: int, marked_states: Optional[List[int]] = None):
        """
        Initialize comparison.

        Args:
            n_qubits: Number of qubits
            marked_states: List of marked states
        """
        self.n_qubits = n_qubits
        self.marked_states = marked_states or [2**n_qubits - 1]

        # Optimal number of iterations
        search_space = 2 ** n_qubits
        num_marked = len(self.marked_states)
        self.optimal_iterations = int(np.round(
            (np.pi / 4) * np.sqrt(search_space / num_marked)
        ))

    def compare_all(
        self,
        num_iterations: Optional[int] = None,
        taylor_order: int = 10,
        qsvt_degree: int = 10
    ):
        """
        Compare all implementations of Grover's algorithm.

        Args:
            num_iterations: Number of iterations (if None, uses optimal)
            taylor_order: Taylor truncation order
            qsvt_degree: QSVT polynomial degree

        Returns:
            Dictionary with circuits and metrics
        """
        from ..utils.circuit_metrics import analyze_circuit

        if num_iterations is None:
            num_iterations = self.optimal_iterations

        results = {}

        # Standard Grover
        print(f"Building standard Grover circuit...")
        from .standard_grover import StandardGrover
        standard_grover = StandardGrover(self.n_qubits, self.marked_states)
        standard_circuit = standard_grover.build_circuit(num_iterations)
        standard_metrics = analyze_circuit(standard_circuit)

        results['Standard'] = {
            'circuit': standard_circuit,
            'metrics': standard_metrics,
            'success_prob': standard_grover.success_probability(num_iterations)
        }

        # Grover via Taylor
        print(f"Building Grover via Taylor-LCU...")
        taylor_grover = GroverViaTaylor(self.n_qubits, self.marked_states)
        taylor_circuit = taylor_grover.build_circuit(num_iterations, taylor_order)
        taylor_metrics = analyze_circuit(taylor_circuit)

        results['Taylor-LCU'] = {
            'circuit': taylor_circuit,
            'metrics': taylor_metrics,
            'error': taylor_grover.estimate_error(taylor_order, num_iterations)
        }

        # Grover via QSVT
        print(f"Building Grover via QSVT...")
        qsvt_grover = GroverViaQSVT(self.n_qubits, self.marked_states)
        qsvt_circuit = qsvt_grover.build_circuit(num_iterations, qsvt_degree)
        qsvt_metrics = analyze_circuit(qsvt_circuit)

        results['QSVT'] = {
            'circuit': qsvt_circuit,
            'metrics': qsvt_metrics,
            'error': qsvt_grover.estimate_error(qsvt_degree, num_iterations)
        }

        return results

    def print_comparison_table(self, results: dict):
        """
        Print comparison table.

        Args:
            results: Results from compare_all()
        """
        import pandas as pd

        data = []
        for method, result in results.items():
            metrics = result['metrics']
            row = {
                'Method': method,
                'Qubits': metrics['num_qubits'],
                'Depth': metrics['depth'],
                'Gates': metrics['total_gates'],
                'CNOTs': metrics['cnot_count'],
            }

            if 'error' in result:
                row['Error'] = f"{result['error']:.2e}"
            elif 'success_prob' in result:
                row['Success Prob'] = f"{result['success_prob']:.4f}"

            data.append(row)

        df = pd.DataFrame(data)

        print("\n" + "="*80)
        print(f"GROVER'S ALGORITHM COMPARISON ({self.n_qubits} qubits)")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        return df
