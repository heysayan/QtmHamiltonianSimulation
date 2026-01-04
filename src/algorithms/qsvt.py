"""
Quantum Singular Value Transformation (QSVT) for Hamiltonian simulation.

QSVT is a general framework that unifies many quantum algorithms including
QSP, amplitude amplification, and Hamiltonian simulation. It applies polynomial
transformations to the singular values of a block-encoded matrix.

Reference: "Quantum singular value transformation and beyond: exponential
improvements for quantum matrix arithmetics" (Gilyén et al., 2019)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple, Optional, Callable
import scipy.special
import scipy.linalg


class QSVTSimulator:
    """
    Hamiltonian simulation using Quantum Singular Value Transform.

    QSVT generalizes QSP to rectangular matrices and provides a unified
    framework for quantum algorithms.
    """

    def __init__(self, hamiltonian: SparsePauliOp, time: float):
        """
        Initialize QSVT simulator.

        Args:
            hamiltonian: The Hamiltonian to simulate (as SparsePauliOp)
            time: Evolution time
        """
        self.hamiltonian = hamiltonian
        self.time = time
        self.n_qubits = hamiltonian.num_qubits

        # Decompose Hamiltonian
        self.pauli_terms = []
        self.coeffs = []
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            self.pauli_terms.append(str(pauli))
            self.coeffs.append(complex(coeff))

        # Compute normalization constants
        self.alpha = np.sum(np.abs(self.coeffs))  # 1-norm
        self.spectral_norm = self._compute_spectral_norm()

        # Number of ancilla qubits for block encoding
        self.n_terms = len(self.pauli_terms)
        self.n_ancilla = int(np.ceil(np.log2(max(2, self.n_terms))))

    def _compute_spectral_norm(self) -> float:
        """Compute spectral norm of Hamiltonian."""
        try:
            h_matrix = self.hamiltonian.to_matrix()
            eigenvalues = np.linalg.eigvalsh(h_matrix)
            return float(np.max(np.abs(eigenvalues)))
        except:
            return self.alpha

    def build_circuit(self, degree: int, projection: bool = True) -> QuantumCircuit:
        """
        Build QSVT circuit for Hamiltonian simulation.

        Args:
            degree: Degree of polynomial approximation
            projection: Whether to include projection (measurement) step

        Returns:
            QuantumCircuit implementing QSVT
        """
        # Create quantum registers
        system_reg = QuantumRegister(self.n_qubits, 'system')
        ancilla_reg = AncillaRegister(self.n_ancilla + 1, 'ancilla')  # +1 for signal qubit
        qc = QuantumCircuit(system_reg, ancilla_reg)

        # Compute phase angles for QSVT sequence
        phase_angles = self._compute_qsvt_phases(degree)

        # Build QSVT sequence
        self._build_qsvt_sequence(qc, system_reg, ancilla_reg, phase_angles)

        # Projection step (measure ancilla to be |0⟩)
        if projection:
            # In practice, this is done via post-selection
            # We just mark it with a barrier
            qc.barrier()

        return qc

    def _compute_qsvt_phases(self, degree: int) -> np.ndarray:
        """
        Compute phase angles for QSVT to approximate exp(-iHt).

        Uses polynomial approximation of the exponential function.

        Args:
            degree: Polynomial degree

        Returns:
            Array of phase angles for QSVT sequence
        """
        # Normalize time by spectral norm
        if self.spectral_norm > 0:
            normalized_time = self.time * self.spectral_norm
        else:
            normalized_time = self.time

        # Use Jacobi-Anger expansion for exponential
        phases = self._jacobi_anger_qsvt_phases(degree, normalized_time)

        return phases

    def _jacobi_anger_qsvt_phases(self, degree: int, lambda_t: float) -> np.ndarray:
        """
        Compute QSVT phases using Jacobi-Anger expansion.

        exp(-i*lambda_t*x) = J_0(lambda_t) + 2*sum_{k=1}^infty (-i)^k J_k(lambda_t) T_k(x)

        where J_k are Bessel functions and T_k are Chebyshev polynomials.

        Args:
            degree: Polynomial degree
            lambda_t: Scaled time parameter

        Returns:
            QSVT phase angles
        """
        phases = np.zeros(2 * degree + 1)

        # Compute Bessel coefficients
        for k in range(degree + 1):
            bessel = scipy.special.jv(k, lambda_t)

            if k == 0:
                # Central phase
                phases[degree] = np.angle(bessel)
            else:
                # Symmetric phases for positive and negative k
                phase_val = np.angle((-1j) ** k * bessel)
                phases[degree + k] = phase_val
                phases[degree - k] = -phase_val

        return phases

    def _taylor_qsvt_phases(self, degree: int, lambda_t: float) -> np.ndarray:
        """
        Compute QSVT phases using Taylor series.

        Args:
            degree: Polynomial degree
            lambda_t: Scaled time parameter

        Returns:
            QSVT phase angles
        """
        phases = np.zeros(degree + 1)

        for k in range(degree + 1):
            # Taylor coefficient: (-i*lambda_t)^k / k!
            coeff = ((-1j * lambda_t) ** k) / scipy.special.factorial(k)
            phases[k] = np.angle(coeff)

        return phases

    def _build_qsvt_sequence(
        self,
        qc: QuantumCircuit,
        system_reg: QuantumRegister,
        ancilla_reg: AncillaRegister,
        phases: np.ndarray
    ):
        """
        Build the QSVT sequence of operations.

        The QSVT sequence consists of alternating signal processing rotations
        and block-encoded operator applications.

        Args:
            qc: QuantumCircuit to build on
            system_reg: System qubit register
            ancilla_reg: Ancilla qubit register
            phases: Phase angles for the sequence
        """
        signal_qubit = ancilla_reg[-1]  # Last ancilla is signal qubit
        prep_ancilla = list(ancilla_reg[:-1])  # Rest are for state preparation

        # Initialize signal qubit
        qc.h(signal_qubit)

        # QSVT iteration
        for i, phase in enumerate(phases):
            # Signal processing rotation
            qc.rz(2 * phase, signal_qubit)

            # Apply block-encoded operator (if not last iteration)
            if i < len(phases) - 1:
                self._apply_block_encoding(qc, system_reg, prep_ancilla, signal_qubit)

        # Final basis rotation
        qc.h(signal_qubit)

    def _apply_block_encoding(
        self,
        qc: QuantumCircuit,
        system_reg: QuantumRegister,
        prep_ancilla: List,
        signal_qubit: int
    ):
        """
        Apply block encoding of the Hamiltonian.

        Block encoding creates a unitary U such that:
        ⟨0|_anc U |0⟩_anc = H/alpha

        Args:
            qc: QuantumCircuit to append to
            system_reg: System qubit register
            prep_ancilla: Ancilla qubits for state preparation
            signal_qubit: Signal qubit for QSVT
        """
        # PREPARE: Create superposition encoding coefficients
        self._prepare_lcu_state(qc, prep_ancilla)

        # SELECT: Apply Pauli operators controlled on ancilla
        self._apply_select_operation(qc, system_reg, prep_ancilla, signal_qubit)

        # PREPARE†: Unprepare ancilla state
        self._prepare_lcu_state(qc, prep_ancilla, inverse=True)

    def _prepare_lcu_state(
        self,
        qc: QuantumCircuit,
        ancilla_qubits: List,
        inverse: bool = False
    ):
        """
        Prepare ancilla state encoding LCU coefficients.

        Creates state: |0⟩ -> sum_j sqrt(|alpha_j|/alpha) |j⟩

        Args:
            qc: QuantumCircuit to append to
            ancilla_qubits: Ancilla qubits
            inverse: Whether to apply inverse
        """
        if len(ancilla_qubits) == 0:
            return

        # Normalized coefficients
        normalized_coeffs = np.abs(self.coeffs) / self.alpha

        # For simplicity, create uniform superposition
        # Full implementation would use exact amplitude encoding
        if not inverse:
            for qubit in ancilla_qubits:
                qc.h(qubit)
        else:
            for qubit in ancilla_qubits:
                qc.h(qubit)

    def _apply_select_operation(
        self,
        qc: QuantumCircuit,
        system_reg: QuantumRegister,
        prep_ancilla: List,
        signal_qubit: int
    ):
        """
        Apply SELECT operation: controlled Pauli operators.

        Args:
            qc: QuantumCircuit to append to
            system_reg: System qubit register
            prep_ancilla: Preparation ancilla qubits
            signal_qubit: Signal qubit
        """
        # Apply each Pauli term controlled on ancilla state
        for idx, pauli_str in enumerate(self.pauli_terms):
            # Apply controlled Pauli
            # This is simplified; full implementation needs multi-controlled operations
            self._apply_controlled_pauli_string(
                qc, system_reg, signal_qubit, pauli_str
            )

    def _apply_controlled_pauli_string(
        self,
        qc: QuantumCircuit,
        system_reg: QuantumRegister,
        control_qubit: int,
        pauli_str: str
    ):
        """
        Apply controlled Pauli string operation.

        Args:
            qc: QuantumCircuit to append to
            system_reg: System qubit register
            control_qubit: Control qubit
            pauli_str: Pauli string to apply
        """
        # Convert to computational basis
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(system_reg):
                break
            qubit = system_reg[i]
            if p == 'X':
                qc.h(qubit)
            elif p == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)

        # Apply controlled-Z operations
        active_qubits = []
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(system_reg):
                break
            if p != 'I':
                active_qubits.append(system_reg[i])

        if len(active_qubits) == 1:
            qc.cz(control_qubit, active_qubits[0])
        elif len(active_qubits) > 1:
            # CNOT ladder
            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i], active_qubits[i + 1])
            qc.cz(control_qubit, active_qubits[-1])
            for i in range(len(active_qubits) - 2, -1, -1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

        # Convert back from computational basis
        for i, p in enumerate(reversed(pauli_str)):
            if i >= len(system_reg):
                break
            qubit = system_reg[i]
            if p == 'X':
                qc.h(qubit)
            elif p == 'Y':
                qc.h(qubit)
                qc.s(qubit)

    def estimate_error(self, degree: int) -> float:
        """
        Estimate approximation error for QSVT.

        Args:
            degree: Polynomial degree

        Returns:
            Estimated error bound
        """
        lambda_t = self.spectral_norm * self.time

        # Error from polynomial approximation of exponential
        # Using truncated Jacobi-Anger expansion error
        if lambda_t < 1:
            # Taylor series error
            error = (lambda_t ** (degree + 1)) / scipy.special.factorial(degree + 1)
        else:
            # Jacobi-Anger truncation error (approximate)
            # Error ~ exp(-degree * log(degree / (e * lambda_t)))
            if degree > lambda_t:
                error = np.exp(-degree * np.log(degree / (np.e * lambda_t)))
            else:
                error = 1.0  # Conservative bound

        return float(error)

    def get_required_degree(self, target_error: float) -> int:
        """
        Estimate required polynomial degree for target error.

        Args:
            target_error: Desired error bound

        Returns:
            Required polynomial degree
        """
        lambda_t = self.spectral_norm * self.time

        if lambda_t < 1:
            # Use Taylor series
            for d in range(1, 1000):
                error = (lambda_t ** (d + 1)) / scipy.special.factorial(d + 1)
                if error < target_error:
                    return d
        else:
            # Use Jacobi-Anger (approximate)
            for d in range(int(np.ceil(lambda_t)), 1000):
                if d > lambda_t:
                    error = np.exp(-d * np.log(d / (np.e * lambda_t)))
                    if error < target_error:
                        return d

        return 1000

    def get_query_complexity(self, degree: int) -> int:
        """
        Compute query complexity (number of block encoding calls).

        Args:
            degree: Polynomial degree

        Returns:
            Number of queries to block encoding
        """
        # QSVT requires 2*degree + 1 queries
        return 2 * degree + 1


def qsvt_simulation(
    hamiltonian: SparsePauliOp,
    time: float,
    degree: int,
    projection: bool = True
) -> QuantumCircuit:
    """
    Convenience function for QSVT Hamiltonian simulation.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        degree: Polynomial degree
        projection: Whether to include projection step

    Returns:
        QuantumCircuit implementing QSVT simulation
    """
    simulator = QSVTSimulator(hamiltonian, time)
    return simulator.build_circuit(degree, projection)


def qsvt_optimal_simulation(
    hamiltonian: SparsePauliOp,
    time: float,
    target_error: float
) -> Tuple[QuantumCircuit, dict]:
    """
    Build QSVT circuit with automatically determined parameters for target error.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        target_error: Target approximation error

    Returns:
        Tuple of (circuit, metadata dict with degree and query complexity)
    """
    simulator = QSVTSimulator(hamiltonian, time)

    # Determine required degree
    degree = simulator.get_required_degree(target_error)

    # Build circuit
    circuit = simulator.build_circuit(degree)

    # Compute metrics
    metadata = {
        'degree': degree,
        'query_complexity': simulator.get_query_complexity(degree),
        'estimated_error': simulator.estimate_error(degree),
        'spectral_norm': simulator.spectral_norm,
        'alpha': simulator.alpha,
    }

    return circuit, metadata
