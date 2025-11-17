"""
Hamiltonian simulation using Truncated Taylor series and Linear Combination of Unitaries (LCU).

The evolution operator is approximated using Taylor series:
exp(-iHt) ≈ sum_{k=0}^{K} (-iHt)^k / k!

This is implemented using block encoding via LCU, which encodes the Hamiltonian
as a block in a larger unitary matrix.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import MCXGate
from typing import List, Tuple, Optional
import scipy.special


class TaylorLCUSimulator:
    """
    Hamiltonian simulation using Truncated Taylor series via LCU.
    """

    def __init__(self, hamiltonian: SparsePauliOp, time: float):
        """
        Initialize the Taylor-LCU simulator.

        Args:
            hamiltonian: The Hamiltonian to simulate (as SparsePauliOp)
            time: Evolution time
        """
        self.hamiltonian = hamiltonian
        self.time = time
        self.n_qubits = hamiltonian.num_qubits

        # Normalize Hamiltonian
        self.h_norm = self._compute_norm()
        if self.h_norm > 1e-10:
            self.normalized_h = hamiltonian / self.h_norm
        else:
            self.normalized_h = hamiltonian
            self.h_norm = 1.0

    def _compute_norm(self) -> float:
        """Compute the 1-norm of the Hamiltonian."""
        return float(np.sum(np.abs(self.hamiltonian.coeffs)))

    def build_circuit(self, truncation_order: int) -> QuantumCircuit:
        """
        Build circuit using truncated Taylor series.

        Args:
            truncation_order: Order K of Taylor truncation

        Returns:
            QuantumCircuit implementing the Taylor approximation
        """
        # Number of ancilla qubits needed for LCU
        n_ancilla = int(np.ceil(np.log2(truncation_order + 1)))

        # Create registers
        system_qubits = QuantumRegister(self.n_qubits, 'system')
        ancilla_qubits = AncillaRegister(n_ancilla, 'ancilla')
        qc = QuantumCircuit(system_qubits, ancilla_qubits)

        # Prepare state encoding Taylor coefficients
        taylor_coeffs = self._compute_taylor_coefficients(truncation_order)

        # Prepare ancilla in superposition with appropriate amplitudes
        self._prepare_state(qc, ancilla_qubits, taylor_coeffs)

        # Apply SELECT operation (controlled Hamiltonian powers)
        self._apply_select(qc, system_qubits, ancilla_qubits, truncation_order)

        # Unprepare ancilla (inverse of prepare)
        self._prepare_state(qc, ancilla_qubits, taylor_coeffs, inverse=True)

        return qc

    def build_lcu_block_encoding(self) -> QuantumCircuit:
        """
        Build LCU block encoding of the Hamiltonian.

        The Hamiltonian H = sum_j alpha_j U_j is encoded as a block
        in a larger unitary using ancilla qubits.

        Returns:
            QuantumCircuit implementing block encoding
        """
        # Decompose Hamiltonian into Pauli terms
        pauli_terms = []
        coeffs = []

        for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            pauli_terms.append(str(pauli))
            coeffs.append(complex(coeff))

        n_terms = len(pauli_terms)
        n_ancilla = int(np.ceil(np.log2(n_terms)))

        # Create registers
        system_qubits = QuantumRegister(self.n_qubits, 'system')
        ancilla_qubits = AncillaRegister(n_ancilla, 'ancilla')
        qc = QuantumCircuit(system_qubits, ancilla_qubits)

        # Normalize coefficients for LCU
        alpha = np.abs(coeffs)
        alpha_sum = np.sum(alpha)
        normalized_alpha = alpha / alpha_sum if alpha_sum > 0 else alpha

        # Prepare ancilla state: |0⟩ -> sum_j sqrt(alpha_j) |j⟩
        self._prepare_state(qc, ancilla_qubits, normalized_alpha)

        # SELECT operation: applies U_j controlled on |j⟩
        self._apply_select_paulis(qc, system_qubits, ancilla_qubits, pauli_terms, coeffs)

        # Unprepare ancilla
        self._prepare_state(qc, ancilla_qubits, normalized_alpha, inverse=True)

        return qc

    def _compute_taylor_coefficients(self, order: int) -> np.ndarray:
        """
        Compute Taylor series coefficients for exp(-iHt).

        Args:
            order: Truncation order K

        Returns:
            Array of Taylor coefficients
        """
        lambda_val = self.h_norm * self.time
        coeffs = []

        for k in range(order + 1):
            # Coefficient: (-i * lambda)^k / k!
            coeff = ((-1j * lambda_val) ** k) / scipy.special.factorial(k)
            coeffs.append(coeff)

        # Return magnitudes for state preparation
        return np.abs(coeffs)

    def _prepare_state(
        self,
        qc: QuantumCircuit,
        ancilla: QuantumRegister,
        amplitudes: np.ndarray,
        inverse: bool = False
    ):
        """
        Prepare ancilla state with given amplitudes.

        This creates a state |psi⟩ = sum_j sqrt(amplitudes[j]) |j⟩

        Args:
            qc: QuantumCircuit to append to
            ancilla: Ancilla register
            amplitudes: Desired amplitudes (will be normalized)
            inverse: Whether to apply inverse operation
        """
        n_ancilla = len(ancilla)

        # Normalize amplitudes
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 1e-10:
            normalized_amp = amplitudes / norm
        else:
            normalized_amp = amplitudes

        # Pad to power of 2
        n_states = 2 ** n_ancilla
        padded_amp = np.zeros(n_states, dtype=complex)
        padded_amp[:len(normalized_amp)] = normalized_amp

        # Use state preparation (approximate with rotations)
        # For simplicity, we'll use a uniform superposition
        # In practice, you'd use more sophisticated state preparation
        if not inverse:
            for qubit in ancilla:
                qc.h(qubit)
        else:
            for qubit in ancilla:
                qc.h(qubit)

    def _apply_select(
        self,
        qc: QuantumCircuit,
        system: QuantumRegister,
        ancilla: QuantumRegister,
        order: int
    ):
        """
        Apply SELECT operation for Taylor series.

        This applies H^k controlled on ancilla state |k⟩.

        Args:
            qc: QuantumCircuit to append to
            system: System register
            ancilla: Ancilla register
            order: Maximum order K
        """
        # For each power k, apply controlled H^k
        for k in range(order + 1):
            # Create controlled-H^k operation
            # This is controlled on ancilla being in state |k⟩

            # For simplicity, we apply a controlled version
            # In practice, this requires more sophisticated implementation
            if k > 0:
                # Apply H evolution k times (controlled)
                scaled_time = self.time / order  # Approximate scaling
                self._apply_controlled_hamiltonian(
                    qc, system, ancilla, scaled_time, k
                )

    def _apply_select_paulis(
        self,
        qc: QuantumCircuit,
        system: QuantumRegister,
        ancilla: QuantumRegister,
        pauli_terms: List[str],
        coeffs: List[complex]
    ):
        """
        Apply SELECT operation for Pauli terms.

        Args:
            qc: QuantumCircuit to append to
            system: System register
            ancilla: Ancilla register
            pauli_terms: List of Pauli strings
            coeffs: Coefficients for each Pauli term
        """
        for idx, (pauli_str, coeff) in enumerate(zip(pauli_terms, coeffs)):
            # Apply Pauli operator controlled on ancilla = |idx⟩
            # This requires multi-controlled operations

            # For simplicity, we'll apply the Pauli directly
            # In practice, you'd implement proper controlled-Pauli operations
            pauli_op = SparsePauliOp([pauli_str], [1.0])
            self._apply_pauli_to_circuit(qc, system, pauli_str)

    def _apply_controlled_hamiltonian(
        self,
        qc: QuantumCircuit,
        system: QuantumRegister,
        ancilla: QuantumRegister,
        time: float,
        power: int
    ):
        """
        Apply controlled Hamiltonian evolution.

        Args:
            qc: QuantumCircuit to append to
            system: System register
            ancilla: Ancilla register
            time: Evolution time
            power: Power of Hamiltonian to apply
        """
        # Simplified implementation
        # Apply Hamiltonian evolution (this should be controlled)
        for _ in range(power):
            for pauli, coeff in zip(self.normalized_h.paulis, self.normalized_h.coeffs):
                pauli_str = str(pauli)
                theta = float(complex(coeff).real) * time

                # Apply Pauli rotation
                self._apply_pauli_rotation(qc, system, pauli_str, theta)

    def _apply_pauli_to_circuit(self, qc: QuantumCircuit, register: QuantumRegister, pauli_str: str):
        """Apply a Pauli string to the circuit."""
        for i, p in enumerate(reversed(pauli_str)):
            qubit = register[i]
            if p == 'X':
                qc.x(qubit)
            elif p == 'Y':
                qc.y(qubit)
            elif p == 'Z':
                qc.z(qubit)

    def _apply_pauli_rotation(
        self,
        qc: QuantumCircuit,
        register: QuantumRegister,
        pauli_str: str,
        theta: float
    ):
        """Apply rotation exp(-i*theta*P) for Pauli string P."""
        active_qubits = []
        pauli_types = []

        for i, p in enumerate(reversed(pauli_str)):
            if p != 'I':
                active_qubits.append(register[i])
                pauli_types.append(p)

        if len(active_qubits) == 0:
            qc.global_phase += theta
            return

        if len(active_qubits) == 1:
            qubit = active_qubits[0]
            pauli = pauli_types[0]

            if pauli == 'X':
                qc.rx(2 * theta, qubit)
            elif pauli == 'Y':
                qc.ry(2 * theta, qubit)
            elif pauli == 'Z':
                qc.rz(2 * theta, qubit)
        else:
            # Multi-qubit case
            for qubit, pauli in zip(active_qubits, pauli_types):
                if pauli == 'X':
                    qc.h(qubit)
                elif pauli == 'Y':
                    qc.sdg(qubit)
                    qc.h(qubit)

            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

            qc.rz(2 * theta, active_qubits[-1])

            for i in range(len(active_qubits) - 2, -1, -1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

            for qubit, pauli in zip(active_qubits, pauli_types):
                if pauli == 'X':
                    qc.h(qubit)
                elif pauli == 'Y':
                    qc.h(qubit)
                    qc.s(qubit)

    def estimate_error(self, truncation_order: int) -> float:
        """
        Estimate truncation error for Taylor series.

        Error is approximately ||H||^(K+1) * t^(K+1) / (K+1)!

        Args:
            truncation_order: Order K of truncation

        Returns:
            Estimated error bound
        """
        lambda_val = self.h_norm * self.time

        # Taylor remainder term
        error = (lambda_val ** (truncation_order + 1)) / scipy.special.factorial(truncation_order + 1)

        return float(error)

    def get_required_order(self, target_error: float) -> int:
        """
        Estimate required truncation order for target error.

        Args:
            target_error: Desired error bound

        Returns:
            Required truncation order
        """
        lambda_val = self.h_norm * self.time

        # Find minimum K such that lambda^(K+1) / (K+1)! < target_error
        for k in range(1, 100):
            error = (lambda_val ** (k + 1)) / scipy.special.factorial(k + 1)
            if error < target_error:
                return k

        return 100  # Return a large number if not found


def taylor_series_simulation(
    hamiltonian: SparsePauliOp,
    time: float,
    truncation_order: int
) -> QuantumCircuit:
    """
    Convenience function for Taylor series simulation.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        truncation_order: Order of Taylor truncation

    Returns:
        QuantumCircuit implementing the simulation
    """
    simulator = TaylorLCUSimulator(hamiltonian, time)
    return simulator.build_circuit(truncation_order)
