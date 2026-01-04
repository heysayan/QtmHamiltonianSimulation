"""
Utility functions for Hamiltonian operations and manipulations.
"""

import numpy as np
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit import QuantumCircuit
from typing import List, Tuple, Union


def pauli_decomposition(hamiltonian: np.ndarray) -> SparsePauliOp:
    """
    Decompose a Hamiltonian matrix into a sum of Pauli operators.

    Args:
        hamiltonian: Hermitian matrix representing the Hamiltonian

    Returns:
        SparsePauliOp: Pauli decomposition of the Hamiltonian
    """
    n_qubits = int(np.log2(hamiltonian.shape[0]))

    # Create Pauli basis
    pauli_labels = []
    pauli_strings = ['I', 'X', 'Y', 'Z']

    def generate_pauli_strings(n):
        if n == 0:
            return ['']
        smaller = generate_pauli_strings(n - 1)
        return [s + p for p in pauli_strings for s in smaller]

    pauli_labels = generate_pauli_strings(n_qubits)

    # Compute coefficients
    coeffs = []
    labels = []

    for label in pauli_labels:
        pauli_op = SparsePauliOp(label)
        pauli_matrix = pauli_op.to_matrix()

        # Coefficient is Tr(H * P) / 2^n
        coeff = np.trace(hamiltonian @ pauli_matrix) / (2 ** n_qubits)

        if np.abs(coeff) > 1e-10:  # Only keep non-zero terms
            coeffs.append(coeff)
            labels.append(label)

    return SparsePauliOp(labels, coeffs)


def normalize_hamiltonian(hamiltonian: SparsePauliOp) -> Tuple[float, SparsePauliOp]:
    """
    Normalize a Hamiltonian to have spectral norm <= 1.

    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian

    Returns:
        Tuple of (normalization factor, normalized Hamiltonian)
    """
    # Compute spectral norm (largest eigenvalue magnitude)
    h_matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvalsh(h_matrix)
    spectral_norm = np.max(np.abs(eigenvalues))

    if spectral_norm < 1e-10:
        return 1.0, hamiltonian

    normalized_h = hamiltonian / spectral_norm
    return spectral_norm, normalized_h


def get_hamiltonian_terms(hamiltonian: SparsePauliOp) -> List[Tuple[str, float]]:
    """
    Extract individual Pauli terms from a Hamiltonian.

    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian

    Returns:
        List of (pauli_string, coefficient) tuples
    """
    terms = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        terms.append((str(pauli), complex(coeff).real))

    return terms


def create_test_hamiltonian(n_qubits: int, hamiltonian_type: str = "random") -> SparsePauliOp:
    """
    Create test Hamiltonians for benchmarking.

    Args:
        n_qubits: Number of qubits
        hamiltonian_type: Type of Hamiltonian ("random", "heisenberg", "transverse_ising")

    Returns:
        SparsePauliOp representing the Hamiltonian
    """
    if hamiltonian_type == "random":
        # Random Hermitian Hamiltonian
        np.random.seed(42)
        h_matrix = np.random.randn(2**n_qubits, 2**n_qubits) + \
                   1j * np.random.randn(2**n_qubits, 2**n_qubits)
        h_matrix = (h_matrix + h_matrix.conj().T) / 2  # Make Hermitian
        return pauli_decomposition(h_matrix)

    elif hamiltonian_type == "heisenberg":
        # Heisenberg model: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
        pauli_list = []
        coeffs = []

        for i in range(n_qubits - 1):
            # XX term
            xx_string = 'I' * i + 'XX' + 'I' * (n_qubits - i - 2)
            pauli_list.append(xx_string)
            coeffs.append(1.0)

            # YY term
            yy_string = 'I' * i + 'YY' + 'I' * (n_qubits - i - 2)
            pauli_list.append(yy_string)
            coeffs.append(1.0)

            # ZZ term
            zz_string = 'I' * i + 'ZZ' + 'I' * (n_qubits - i - 2)
            pauli_list.append(zz_string)
            coeffs.append(1.0)

        return SparsePauliOp(pauli_list, coeffs)

    elif hamiltonian_type == "transverse_ising":
        # Transverse field Ising model: H = -sum_i Z_i Z_{i+1} - g * sum_i X_i
        g = 0.5  # Transverse field strength
        pauli_list = []
        coeffs = []

        # ZZ interactions
        for i in range(n_qubits - 1):
            zz_string = 'I' * i + 'ZZ' + 'I' * (n_qubits - i - 2)
            pauli_list.append(zz_string)
            coeffs.append(-1.0)

        # X fields
        for i in range(n_qubits):
            x_string = 'I' * i + 'X' + 'I' * (n_qubits - i - 1)
            pauli_list.append(x_string)
            coeffs.append(-g)

        return SparsePauliOp(pauli_list, coeffs)

    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")


def hamiltonian_to_matrix(hamiltonian: SparsePauliOp) -> np.ndarray:
    """
    Convert SparsePauliOp to dense matrix.

    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian

    Returns:
        Dense matrix representation
    """
    return hamiltonian.to_matrix()


def compute_evolution_operator(hamiltonian: SparsePauliOp, time: float) -> np.ndarray:
    """
    Compute the exact evolution operator exp(-iHt) for comparison.

    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian
        time: Evolution time

    Returns:
        Evolution operator as a matrix
    """
    h_matrix = hamiltonian.to_matrix()
    return scipy.linalg.expm(-1j * time * h_matrix)


def one_norm(hamiltonian: SparsePauliOp) -> float:
    """
    Compute the 1-norm of a Hamiltonian (sum of absolute values of coefficients).

    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian

    Returns:
        1-norm value
    """
    return np.sum(np.abs(hamiltonian.coeffs))


def spectral_norm(hamiltonian: SparsePauliOp) -> float:
    """
    Compute the spectral norm (largest eigenvalue magnitude) of a Hamiltonian.

    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian

    Returns:
        Spectral norm value
    """
    h_matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvalsh(h_matrix)
    return np.max(np.abs(eigenvalues))
