"""
Utilities for analyzing and comparing quantum circuits.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.converters import circuit_to_dag
from typing import Dict, List
import pandas as pd


def analyze_circuit(circuit: QuantumCircuit) -> Dict:
    """
    Analyze a quantum circuit and extract various metrics.

    Args:
        circuit: QuantumCircuit to analyze

    Returns:
        Dictionary containing circuit metrics
    """
    dag = circuit_to_dag(circuit)

    # Basic metrics
    metrics = {
        'num_qubits': circuit.num_qubits,
        'num_clbits': circuit.num_clbits,
        'depth': circuit.depth(),
        'size': circuit.size(),  # Total number of gates
        'width': circuit.width(),  # num_qubits + num_clbits
    }

    # Gate count by type
    gate_counts = circuit.count_ops()
    metrics['gate_counts'] = gate_counts
    metrics['total_gates'] = sum(gate_counts.values())

    # Single vs two-qubit gates
    single_qubit_gates = 0
    two_qubit_gates = 0
    multi_qubit_gates = 0

    for gate_name, count in gate_counts.items():
        # Common single-qubit gates
        if gate_name in ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg',
                          'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 'p']:
            single_qubit_gates += count
        # Common two-qubit gates
        elif gate_name in ['cx', 'cy', 'cz', 'ch', 'swap', 'iswap',
                           'dcx', 'ecr', 'rxx', 'ryy', 'rzz', 'rzx']:
            two_qubit_gates += count
        # Multi-qubit gates
        else:
            multi_qubit_gates += count

    metrics['single_qubit_gates'] = single_qubit_gates
    metrics['two_qubit_gates'] = two_qubit_gates
    metrics['multi_qubit_gates'] = multi_qubit_gates

    # CNOT count (important metric)
    metrics['cnot_count'] = gate_counts.get('cx', 0)

    # T-gate count (important for fault-tolerant implementations)
    metrics['t_count'] = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)

    # Ancilla qubits (estimate)
    # This is a heuristic - qubits that are initialized and measured but not in output
    metrics['estimated_ancillas'] = 0  # Would need more context to determine

    return metrics


def compare_circuits(circuits: Dict[str, QuantumCircuit], verbose: bool = True) -> pd.DataFrame:
    """
    Compare multiple quantum circuits.

    Args:
        circuits: Dictionary mapping algorithm names to circuits
        verbose: Whether to print comparison table

    Returns:
        DataFrame containing comparison metrics
    """
    comparison_data = []

    for name, circuit in circuits.items():
        metrics = analyze_circuit(circuit)
        row = {
            'Algorithm': name,
            'Qubits': metrics['num_qubits'],
            'Depth': metrics['depth'],
            'Total Gates': metrics['total_gates'],
            'Single-Qubit': metrics['single_qubit_gates'],
            'Two-Qubit': metrics['two_qubit_gates'],
            'CNOT Count': metrics['cnot_count'],
            'T Count': metrics['t_count'],
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    if verbose:
        print("\n" + "="*80)
        print("CIRCUIT COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

    return df


def compute_gate_complexity(circuit: QuantumCircuit) -> Dict[str, int]:
    """
    Compute various complexity measures for a circuit.

    Args:
        circuit: QuantumCircuit to analyze

    Returns:
        Dictionary of complexity metrics
    """
    metrics = analyze_circuit(circuit)

    complexity = {
        'gate_complexity': metrics['total_gates'],
        'depth_complexity': metrics['depth'],
        'cnot_complexity': metrics['cnot_count'],
        'space_complexity': metrics['num_qubits'],
    }

    return complexity


def compute_fidelity(circuit: QuantumCircuit, target_unitary: np.ndarray) -> float:
    """
    Compute the fidelity between a circuit and target unitary.

    Args:
        circuit: QuantumCircuit to compare
        target_unitary: Target unitary matrix

    Returns:
        Fidelity value (0 to 1)
    """
    try:
        circuit_unitary = Operator(circuit).data

        # Process fidelity: F = |Tr(U†V)|² / d²
        d = circuit_unitary.shape[0]
        trace = np.trace(np.conj(target_unitary.T) @ circuit_unitary)
        fidelity = np.abs(trace) ** 2 / (d ** 2)

        return float(fidelity.real)
    except Exception as e:
        print(f"Warning: Could not compute fidelity: {e}")
        return 0.0


def compute_diamond_distance(circuit: QuantumCircuit, target_unitary: np.ndarray) -> float:
    """
    Estimate the diamond distance between a circuit and target unitary.

    This is an approximation using the operator norm.

    Args:
        circuit: QuantumCircuit to compare
        target_unitary: Target unitary matrix

    Returns:
        Estimated diamond distance
    """
    try:
        circuit_unitary = Operator(circuit).data

        # Diamond distance is approximately bounded by operator norm
        diff = circuit_unitary - target_unitary
        operator_norm = np.linalg.norm(diff, ord=2)

        return float(operator_norm)
    except Exception as e:
        print(f"Warning: Could not compute diamond distance: {e}")
        return float('inf')


def print_circuit_summary(circuit: QuantumCircuit, name: str = "Circuit"):
    """
    Print a human-readable summary of a circuit.

    Args:
        circuit: QuantumCircuit to summarize
        name: Name of the circuit
    """
    metrics = analyze_circuit(circuit)

    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Qubits:           {metrics['num_qubits']}")
    print(f"Circuit Depth:    {metrics['depth']}")
    print(f"Total Gates:      {metrics['total_gates']}")
    print(f"  - Single-qubit: {metrics['single_qubit_gates']}")
    print(f"  - Two-qubit:    {metrics['two_qubit_gates']}")
    print(f"  - Multi-qubit:  {metrics['multi_qubit_gates']}")
    print(f"CNOT Count:       {metrics['cnot_count']}")
    print(f"T-gate Count:     {metrics['t_count']}")
    print(f"\nGate breakdown:")
    for gate, count in sorted(metrics['gate_counts'].items()):
        print(f"  {gate:10s}: {count}")
    print(f"{'='*60}\n")


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create a formatted comparison table from benchmark results.

    Args:
        results: List of dictionaries containing benchmark results

    Returns:
        DataFrame with formatted results
    """
    df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = ['Algorithm', 'Time', 'Qubits', 'Depth', 'Gates',
                    'CNOTs', 'Fidelity', 'Error']

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]

    return df[column_order]
