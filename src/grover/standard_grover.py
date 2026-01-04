"""
Standard Grover's search algorithm implementation.

Grover's algorithm searches an unstructured database of N=2^n items
in O(√N) queries, providing quadratic speedup over classical search.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from typing import List, Optional, Callable


class StandardGrover:
    """
    Standard implementation of Grover's search algorithm.
    """

    def __init__(self, n_qubits: int, marked_states: Optional[List[int]] = None):
        """
        Initialize Grover's search.

        Args:
            n_qubits: Number of qubits (search space size = 2^n)
            marked_states: List of marked state indices (if None, defaults to |11...1⟩)
        """
        self.n_qubits = n_qubits
        self.search_space_size = 2 ** n_qubits

        if marked_states is None:
            # Default: mark the all-ones state
            self.marked_states = [self.search_space_size - 1]
        else:
            self.marked_states = marked_states

        self.num_marked = len(self.marked_states)

    def build_circuit(self, num_iterations: Optional[int] = None) -> QuantumCircuit:
        """
        Build Grover's search circuit.

        Args:
            num_iterations: Number of Grover iterations (if None, uses optimal)

        Returns:
            QuantumCircuit implementing Grover's algorithm
        """
        if num_iterations is None:
            num_iterations = self._optimal_iterations()

        qc = QuantumCircuit(self.n_qubits)

        # Initialize superposition
        qc.h(range(self.n_qubits))

        # Apply Grover iterations
        for _ in range(num_iterations):
            # Oracle
            self._apply_oracle(qc)

            # Diffusion operator
            self._apply_diffusion(qc)

        return qc

    def _optimal_iterations(self) -> int:
        """
        Compute optimal number of Grover iterations.

        Returns:
            Optimal number of iterations ≈ π/4 * √(N/M)
            where N is search space size and M is number of marked items
        """
        if self.num_marked == 0:
            return 0

        optimal = int(np.round(
            (np.pi / 4) * np.sqrt(self.search_space_size / self.num_marked)
        ))

        return max(1, optimal)

    def _apply_oracle(self, qc: QuantumCircuit):
        """
        Apply oracle that marks target states.

        Oracle: O|x⟩ = (-1)^f(x)|x⟩ where f(x)=1 for marked states

        Args:
            qc: QuantumCircuit to append oracle to
        """
        for state in self.marked_states:
            # Create oracle for this marked state
            self._apply_state_oracle(qc, state)

    def _apply_state_oracle(self, qc: QuantumCircuit, state: int):
        """
        Apply oracle marking a specific state.

        Args:
            qc: QuantumCircuit to append to
            state: State index to mark
        """
        # Convert state to binary
        binary = format(state, f'0{self.n_qubits}b')

        # Flip qubits where bit is 0
        for i, bit in enumerate(reversed(binary)):
            if bit == '0':
                qc.x(i)

        # Apply multi-controlled Z
        if self.n_qubits == 1:
            qc.z(0)
        elif self.n_qubits == 2:
            qc.cz(0, 1)
        else:
            # Multi-controlled Z using MCX and phase
            qc.h(self.n_qubits - 1)
            qc.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            qc.h(self.n_qubits - 1)

        # Unflip qubits
        for i, bit in enumerate(reversed(binary)):
            if bit == '0':
                qc.x(i)

    def _apply_diffusion(self, qc: QuantumCircuit):
        """
        Apply Grover diffusion operator.

        Diffusion: D = 2|s⟩⟨s| - I where |s⟩ = H^⊗n|0⟩

        Args:
            qc: QuantumCircuit to append diffusion to
        """
        # H^⊗n
        qc.h(range(self.n_qubits))

        # Apply X to all qubits
        qc.x(range(self.n_qubits))

        # Multi-controlled Z
        if self.n_qubits == 1:
            qc.z(0)
        elif self.n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(self.n_qubits - 1)
            qc.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            qc.h(self.n_qubits - 1)

        # Apply X to all qubits
        qc.x(range(self.n_qubits))

        # H^⊗n
        qc.h(range(self.n_qubits))

    def success_probability(self, num_iterations: int) -> float:
        """
        Compute success probability after given iterations.

        Args:
            num_iterations: Number of Grover iterations

        Returns:
            Success probability
        """
        if self.num_marked == 0:
            return 0.0

        theta = np.arcsin(np.sqrt(self.num_marked / self.search_space_size))
        prob = np.sin((2 * num_iterations + 1) * theta) ** 2

        return float(prob)


def create_grover_circuit(
    n_qubits: int,
    marked_states: Optional[List[int]] = None,
    num_iterations: Optional[int] = None
) -> QuantumCircuit:
    """
    Convenience function to create Grover's search circuit.

    Args:
        n_qubits: Number of qubits
        marked_states: List of marked state indices
        num_iterations: Number of iterations (if None, uses optimal)

    Returns:
        QuantumCircuit implementing Grover's algorithm
    """
    grover = StandardGrover(n_qubits, marked_states)
    return grover.build_circuit(num_iterations)


def grover_to_hamiltonian(n_qubits: int, marked_states: Optional[List[int]] = None):
    """
    Convert Grover's problem to Hamiltonian formulation.

    Grover's algorithm can be expressed as Hamiltonian evolution with:
    H = |s⟩⟨s| - |w⟩⟨w|
    where |s⟩ is uniform superposition and |w⟩ is the marked state.

    This is equivalent to H = (I - 2|w⟩⟨w|) + (I - 2|s⟩⟨s|)
                             = 2I - 2|w⟩⟨w| - 2|s⟩⟨s|

    Args:
        n_qubits: Number of qubits
        marked_states: List of marked state indices

    Returns:
        Hamiltonian matrix for Grover's problem
    """
    from qiskit.quantum_info import SparsePauliOp

    if marked_states is None:
        marked_states = [2**n_qubits - 1]

    N = 2 ** n_qubits

    # Create Hamiltonian matrix
    H = np.zeros((N, N), dtype=complex)

    # |s⟩ = uniform superposition
    s = np.ones(N) / np.sqrt(N)

    # Add -2|s⟩⟨s| term (diffusion)
    H -= 2 * np.outer(s, s)

    # Add -2|w⟩⟨w| term for each marked state (oracle)
    for w_idx in marked_states:
        w = np.zeros(N)
        w[w_idx] = 1.0
        H -= 2 * np.outer(w, w)

    # Add 2I to make it proper
    H += 2 * np.eye(N)

    return H


def grover_hamiltonian_pauli(n_qubits: int, marked_states: Optional[List[int]] = None):
    """
    Express Grover's Hamiltonian in Pauli basis.

    Args:
        n_qubits: Number of qubits
        marked_states: List of marked state indices

    Returns:
        SparsePauliOp representing Grover's Hamiltonian
    """
    from qiskit.quantum_info import SparsePauliOp
    from src.utils.hamiltonian_utils import pauli_decomposition

    H_matrix = grover_to_hamiltonian(n_qubits, marked_states)
    return pauli_decomposition(H_matrix)
