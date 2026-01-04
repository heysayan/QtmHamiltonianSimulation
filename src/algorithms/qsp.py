"""
Quantum Signal Processing (QSP) for Hamiltonian simulation.

QSP implements polynomial transformations of a unitary matrix using a sequence
of controlled rotations. It can be used to approximate functions like exp(-iHt)
by polynomial approximation.

Reference: "Quantum singular value transformation and beyond: exponential
improvements for quantum matrix arithmetics" (Gilyén et al., 2019)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple, Optional
import scipy.linalg
import scipy.special


class QSPSimulator:
    """
    Hamiltonian simulation using Quantum Signal Processing.
    """

    def __init__(self, hamiltonian: SparsePauliOp, time: float):
        """
        Initialize QSP simulator.

        Args:
            hamiltonian: The Hamiltonian to simulate (as SparsePauliOp)
            time: Evolution time
        """
        self.hamiltonian = hamiltonian
        self.time = time
        self.n_qubits = hamiltonian.num_qubits

        # Compute Hamiltonian norm for scaling
        self.h_norm = self._compute_spectral_norm()

    def _compute_spectral_norm(self) -> float:
        """Compute spectral norm of Hamiltonian."""
        h_matrix = self.hamiltonian.to_matrix()
        eigenvalues = np.linalg.eigvalsh(h_matrix)
        return float(np.max(np.abs(eigenvalues)))

    def build_circuit(self, degree: int) -> QuantumCircuit:
        """
        Build QSP circuit for Hamiltonian simulation.

        Args:
            degree: Degree of polynomial approximation

        Returns:
            QuantumCircuit implementing QSP
        """
        # Compute phase angles for QSP sequence
        phase_angles = self._compute_qsp_phases(degree)

        # Create circuit
        qc = QuantumCircuit(self.n_qubits + 1)  # +1 for ancilla

        # Build QSP sequence
        self._build_qsp_sequence(qc, phase_angles)

        return qc

    def _compute_qsp_phases(self, degree: int) -> np.ndarray:
        """
        Compute phase angles for QSP sequence to approximate exp(-iHt).

        This uses Jacobi-Anger expansion and polynomial approximation.

        Args:
            degree: Polynomial degree

        Returns:
            Array of phase angles
        """
        # Scaled time parameter
        lambda_t = self.h_norm * self.time

        # For small lambda_t, use Taylor series
        if lambda_t < 1:
            return self._taylor_qsp_phases(degree, lambda_t)
        else:
            # Use Jacobi-Anger expansion
            return self._jacobi_anger_phases(degree, lambda_t)

    def _taylor_qsp_phases(self, degree: int, lambda_t: float) -> np.ndarray:
        """
        Compute QSP phases using Taylor series approximation.

        exp(-i*lambda_t*x) ≈ sum_{k=0}^d (-i*lambda_t)^k / k! * x^k

        Args:
            degree: Polynomial degree
            lambda_t: Scaled time parameter

        Returns:
            Phase angles for QSP
        """
        # Simplified: use uniform phases
        # In practice, these should be computed via optimization
        phases = np.zeros(degree + 1)

        for k in range(degree + 1):
            # Approximate phase for Taylor term
            taylor_coeff = ((-1j * lambda_t) ** k) / scipy.special.factorial(k)
            phases[k] = np.angle(taylor_coeff)

        return phases

    def _jacobi_anger_phases(self, degree: int, lambda_t: float) -> np.ndarray:
        """
        Compute QSP phases using Jacobi-Anger expansion.

        exp(-i*lambda_t*x) = sum_{k=-inf}^{inf} (-i)^k J_k(lambda_t) T_k(x)

        where J_k are Bessel functions and T_k are Chebyshev polynomials.

        Args:
            degree: Polynomial degree
            lambda_t: Scaled time parameter

        Returns:
            Phase angles for QSP
        """
        phases = np.zeros(degree + 1)

        for k in range(degree + 1):
            # Bessel function coefficient
            bessel_coeff = scipy.special.jv(k, lambda_t)
            phases[k] = np.angle((-1j) ** k * bessel_coeff)

        return phases

    def _build_qsp_sequence(self, qc: QuantumCircuit, phases: np.ndarray):
        """
        Build the QSP sequence of rotations.

        The QSP sequence alternates between signal rotations (R_z) and
        signal processing rotations (controlled operations).

        Args:
            qc: QuantumCircuit to build on
            phases: Phase angles for the sequence
        """
        ancilla_qubit = self.n_qubits  # Last qubit is ancilla
        system_qubits = list(range(self.n_qubits))

        # Initial rotation
        qc.h(ancilla_qubit)

        # QSP sequence
        for i, phase in enumerate(phases):
            # Signal rotation
            qc.rz(phase, ancilla_qubit)

            # Signal operator (block-encoded Hamiltonian)
            if i < len(phases) - 1:
                self._apply_signal_operator(qc, system_qubits, ancilla_qubit)

        # Final measurement basis rotation
        qc.h(ancilla_qubit)

    def _apply_signal_operator(
        self,
        qc: QuantumCircuit,
        system_qubits: List[int],
        ancilla_qubit: int
    ):
        """
        Apply the signal operator (block-encoded Hamiltonian).

        Args:
            qc: QuantumCircuit to append to
            system_qubits: List of system qubit indices
            ancilla_qubit: Ancilla qubit index
        """
        # Apply controlled-Hamiltonian
        # This is a simplified version; full implementation requires
        # proper block encoding

        for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            pauli_str = str(pauli)

            # Apply controlled Pauli operation
            self._apply_controlled_pauli(qc, system_qubits, ancilla_qubit, pauli_str)

    def _apply_controlled_pauli(
        self,
        qc: QuantumCircuit,
        system_qubits: List[int],
        control_qubit: int,
        pauli_str: str
    ):
        """
        Apply a controlled Pauli string operation.

        Args:
            qc: QuantumCircuit to append to
            system_qubits: System qubit indices
            control_qubit: Control qubit index
            pauli_str: Pauli string (e.g., "IXYZ")
        """
        # Convert to Z basis
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(system_qubits):
                break
            qubit = system_qubits[i]
            if p == 'X':
                qc.h(qubit)
            elif p == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)

        # Find active qubits
        active_qubits = []
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(system_qubits):
                break
            if p != 'I':
                active_qubits.append(system_qubits[i])

        # Apply controlled-Z operations
        if len(active_qubits) == 1:
            qc.cz(control_qubit, active_qubits[0])
        elif len(active_qubits) > 1:
            # Multi-controlled Z
            # Create CNOT ladder
            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

            qc.cz(control_qubit, active_qubits[-1])

            # Uncompute
            for i in range(len(active_qubits) - 2, -1, -1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

        # Convert back from Z basis
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(system_qubits):
                break
            qubit = system_qubits[i]
            if p == 'X':
                qc.h(qubit)
            elif p == 'Y':
                qc.h(qubit)
                qc.s(qubit)

    def estimate_error(self, degree: int) -> float:
        """
        Estimate approximation error for QSP.

        Args:
            degree: Polynomial degree

        Returns:
            Estimated error bound
        """
        lambda_t = self.h_norm * self.time

        # Error from polynomial approximation
        # Using Taylor series error bound
        error = (lambda_t ** (degree + 1)) / scipy.special.factorial(degree + 1)

        return float(error)

    def get_required_degree(self, target_error: float) -> int:
        """
        Estimate required polynomial degree for target error.

        Args:
            target_error: Desired error bound

        Returns:
            Required polynomial degree
        """
        lambda_t = self.h_norm * self.time

        # Find minimum degree
        for d in range(1, 1000):
            error = (lambda_t ** (d + 1)) / scipy.special.factorial(d + 1)
            if error < target_error:
                return d

        return 1000


class QubitizationSimulator:
    """
    Hamiltonian simulation using Qubitization.

    Qubitization is a special case of QSP that provides optimal scaling
    for Hamiltonian simulation.

    Reference: "Hamiltonian Simulation by Qubitization" (Low & Chuang, 2017)
    """

    def __init__(self, hamiltonian: SparsePauliOp, time: float):
        """
        Initialize Qubitization simulator.

        Args:
            hamiltonian: The Hamiltonian to simulate
            time: Evolution time
        """
        self.hamiltonian = hamiltonian
        self.time = time
        self.n_qubits = hamiltonian.num_qubits

        # Decompose Hamiltonian for LCU
        self.pauli_terms = []
        self.coeffs = []
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            self.pauli_terms.append(str(pauli))
            self.coeffs.append(complex(coeff))

        self.n_terms = len(self.pauli_terms)
        self.n_ancilla = int(np.ceil(np.log2(self.n_terms)))

        # Compute alpha (sum of absolute values)
        self.alpha = np.sum(np.abs(self.coeffs))

    def build_circuit(self, query_complexity: Optional[int] = None) -> QuantumCircuit:
        """
        Build qubitization circuit.

        Args:
            query_complexity: Number of queries to oracle (if None, computed automatically)

        Returns:
            QuantumCircuit implementing qubitization
        """
        if query_complexity is None:
            query_complexity = self._compute_query_complexity()

        # Create circuit with system and ancilla qubits
        qc = QuantumCircuit(self.n_qubits + self.n_ancilla)

        # Build walk operator
        for _ in range(query_complexity):
            self._apply_walk_operator(qc)

        return qc

    def _apply_walk_operator(self, qc: QuantumCircuit):
        """
        Apply one step of the quantum walk operator W = R·S.

        W encodes the Hamiltonian spectrum in its eigenphases.

        Args:
            qc: QuantumCircuit to append to
        """
        # Apply SELECT operation
        self._apply_select(qc)

        # Apply REFLECT operation
        self._apply_reflect(qc)

    def _apply_select(self, qc: QuantumCircuit):
        """
        Apply SELECT operation: applies unitaries controlled on ancilla state.

        Args:
            qc: QuantumCircuit to append to
        """
        system_qubits = list(range(self.n_qubits))
        ancilla_qubits = list(range(self.n_qubits, self.n_qubits + self.n_ancilla))

        # For each Pauli term, apply it controlled on corresponding ancilla state
        for idx, pauli_str in enumerate(self.pauli_terms):
            # This is simplified; full implementation requires multi-controlled operations
            self._apply_pauli_to_qubits(qc, system_qubits, pauli_str)

    def _apply_reflect(self, qc: QuantumCircuit):
        """
        Apply REFLECT operation: reflects about the prepared state.

        Args:
            qc: QuantumCircuit to append to
        """
        ancilla_qubits = list(range(self.n_qubits, self.n_qubits + self.n_ancilla))

        # Reflect about |0⟩ state
        for qubit in ancilla_qubits:
            qc.x(qubit)

        # Multi-controlled Z
        if len(ancilla_qubits) > 1:
            qc.h(ancilla_qubits[-1])
            qc.mcx(ancilla_qubits[:-1], ancilla_qubits[-1])
            qc.h(ancilla_qubits[-1])
        else:
            qc.z(ancilla_qubits[0])

        for qubit in ancilla_qubits:
            qc.x(qubit)

    def _apply_pauli_to_qubits(self, qc: QuantumCircuit, qubits: List[int], pauli_str: str):
        """Apply Pauli string to specified qubits."""
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(qubits):
                break
            qubit = qubits[i]
            if p == 'X':
                qc.x(qubit)
            elif p == 'Y':
                qc.y(qubit)
            elif p == 'Z':
                qc.z(qubit)

    def _compute_query_complexity(self) -> int:
        """
        Compute the number of queries needed for desired accuracy.

        Returns:
            Number of walk operator applications
        """
        # Query complexity: O(alpha * t)
        # where alpha is the 1-norm of coefficients
        query_complexity = int(np.ceil(self.alpha * self.time))
        return max(1, query_complexity)

    def estimate_error(self, query_complexity: int) -> float:
        """
        Estimate simulation error.

        Args:
            query_complexity: Number of queries used

        Returns:
            Estimated error
        """
        # Error scales as O((alpha*t)^2 / queries)
        if query_complexity == 0:
            return float('inf')

        error = (self.alpha * self.time) ** 2 / query_complexity
        return float(error)


def qsp_simulation(hamiltonian: SparsePauliOp, time: float, degree: int) -> QuantumCircuit:
    """
    Convenience function for QSP simulation.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        degree: Polynomial degree

    Returns:
        QuantumCircuit implementing QSP simulation
    """
    simulator = QSPSimulator(hamiltonian, time)
    return simulator.build_circuit(degree)


def qubitization_simulation(hamiltonian: SparsePauliOp, time: float) -> QuantumCircuit:
    """
    Convenience function for Qubitization simulation.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time

    Returns:
        QuantumCircuit implementing qubitization
    """
    simulator = QubitizationSimulator(hamiltonian, time)
    return simulator.build_circuit()
