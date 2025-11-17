"""
First-order and higher-order Trotterization for Hamiltonian simulation.

The Trotter-Suzuki product formula approximates the evolution operator:
exp(-iHt) ≈ [exp(-iH₁t/r) exp(-iH₂t/r) ... exp(-iHₘt/r)]^r

where H = H₁ + H₂ + ... + Hₘ is a sum of Pauli terms, and r is the number of Trotter steps.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple, Optional
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter


class TrotterizationSimulator:
    """
    Hamiltonian simulation using Trotter-Suzuki product formulas.
    """

    def __init__(self, hamiltonian: SparsePauliOp, time: float, order: int = 1):
        """
        Initialize the Trotterization simulator.

        Args:
            hamiltonian: The Hamiltonian to simulate (as SparsePauliOp)
            time: Evolution time
            order: Order of the Trotter formula (1, 2, 4, etc.)
        """
        self.hamiltonian = hamiltonian
        self.time = time
        self.order = order
        self.n_qubits = hamiltonian.num_qubits

    def build_circuit(self, n_trotter_steps: int = 1) -> QuantumCircuit:
        """
        Build the Trotterized circuit for Hamiltonian simulation.

        Args:
            n_trotter_steps: Number of Trotter steps (r)

        Returns:
            QuantumCircuit implementing the Trotterized evolution
        """
        qc = QuantumCircuit(self.n_qubits)

        # Choose synthesis method based on order
        if self.order == 1:
            synthesis = LieTrotter()
        elif self.order == 2:
            synthesis = SuzukiTrotter(order=2)
        elif self.order == 4:
            synthesis = SuzukiTrotter(order=4)
        else:
            # Default to first-order
            synthesis = LieTrotter()

        # Create the evolution gate
        evolution_gate = PauliEvolutionGate(
            self.hamiltonian,
            time=self.time,
            synthesis=synthesis
        )

        # For multiple Trotter steps, we need to scale the time
        if n_trotter_steps > 1:
            evolution_gate = PauliEvolutionGate(
                self.hamiltonian,
                time=self.time / n_trotter_steps,
                synthesis=synthesis
            )
            for _ in range(n_trotter_steps):
                qc.append(evolution_gate, range(self.n_qubits))
        else:
            qc.append(evolution_gate, range(self.n_qubits))

        return qc

    def build_first_order_circuit(self, n_trotter_steps: int = 1) -> QuantumCircuit:
        """
        Build first-order Trotter circuit with explicit decomposition.

        This manually constructs the circuit by evolving each Pauli term.

        Args:
            n_trotter_steps: Number of Trotter steps

        Returns:
            QuantumCircuit implementing first-order Trotterization
        """
        qc = QuantumCircuit(self.n_qubits)

        # Time step per Trotter step
        dt = self.time / n_trotter_steps

        # Apply Trotter steps
        for _ in range(n_trotter_steps):
            # Evolve each Pauli term
            for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
                # Create evolution for single Pauli term
                pauli_op = SparsePauliOp([pauli], [coeff])
                evolution_time = dt * complex(coeff).real

                # Apply evolution exp(-i * coeff * pauli * dt)
                self._apply_pauli_evolution(qc, str(pauli), evolution_time)

        return qc

    def _apply_pauli_evolution(self, qc: QuantumCircuit, pauli_string: str, theta: float):
        """
        Apply evolution under a single Pauli string: exp(-i * theta * P).

        Args:
            qc: QuantumCircuit to append to
            pauli_string: Pauli string (e.g., "IXYZ")
            theta: Evolution angle
        """
        n = len(pauli_string)

        # Find non-identity Pauli operators
        active_qubits = []
        pauli_types = []

        for i, p in enumerate(reversed(pauli_string)):  # Qiskit uses little-endian
            if p != 'I':
                active_qubits.append(i)
                pauli_types.append(p)

        if len(active_qubits) == 0:
            # Identity operator, just add a global phase
            qc.global_phase += theta
            return

        if len(active_qubits) == 1:
            # Single-qubit Pauli rotation
            qubit = active_qubits[0]
            pauli = pauli_types[0]

            if pauli == 'X':
                qc.rx(2 * theta, qubit)
            elif pauli == 'Y':
                qc.ry(2 * theta, qubit)
            elif pauli == 'Z':
                qc.rz(2 * theta, qubit)
        else:
            # Multi-qubit Pauli string
            # Convert to Z basis
            for qubit, pauli in zip(active_qubits, pauli_types):
                if pauli == 'X':
                    qc.h(qubit)
                elif pauli == 'Y':
                    qc.sdg(qubit)
                    qc.h(qubit)

            # Create CNOT ladder
            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

            # Apply RZ rotation on last qubit
            qc.rz(2 * theta, active_qubits[-1])

            # Uncompute CNOT ladder
            for i in range(len(active_qubits) - 2, -1, -1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

            # Convert back from Z basis
            for qubit, pauli in zip(active_qubits, pauli_types):
                if pauli == 'X':
                    qc.h(qubit)
                elif pauli == 'Y':
                    qc.h(qubit)
                    qc.s(qubit)

    def estimate_error(self, n_trotter_steps: int) -> float:
        """
        Estimate the Trotter error for given number of steps.

        For first-order Trotter: error ~ O(t²/r) where r is the number of steps
        For second-order: error ~ O(t³/r²)

        Args:
            n_trotter_steps: Number of Trotter steps

        Returns:
            Estimated error bound
        """
        # Compute commutator bounds (simplified)
        # ||H||² term for error estimation
        h_norm = np.sum(np.abs(self.hamiltonian.coeffs))

        if self.order == 1:
            # First-order error: O((||H|| t)² / r)
            error = (h_norm * self.time) ** 2 / n_trotter_steps
        elif self.order == 2:
            # Second-order error: O((||H|| t)³ / r²)
            error = (h_norm * self.time) ** 3 / (n_trotter_steps ** 2)
        else:
            # General higher-order (approximate)
            error = (h_norm * self.time) ** (self.order + 1) / (n_trotter_steps ** self.order)

        return error

    def get_required_steps(self, target_error: float) -> int:
        """
        Estimate the number of Trotter steps needed for a target error.

        Args:
            target_error: Desired error bound

        Returns:
            Number of Trotter steps required
        """
        h_norm = np.sum(np.abs(self.hamiltonian.coeffs))

        if self.order == 1:
            r = int(np.ceil((h_norm * self.time) ** 2 / target_error))
        elif self.order == 2:
            r = int(np.ceil(np.sqrt((h_norm * self.time) ** 3 / target_error)))
        else:
            r = int(np.ceil((h_norm * self.time) ** (self.order + 1) / target_error) ** (1 / self.order))

        return max(1, r)


def first_order_trotter(
    hamiltonian: SparsePauliOp,
    time: float,
    n_trotter_steps: int = 1
) -> QuantumCircuit:
    """
    Convenience function for first-order Trotterization.

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        n_trotter_steps: Number of Trotter steps

    Returns:
        QuantumCircuit implementing the evolution
    """
    simulator = TrotterizationSimulator(hamiltonian, time, order=1)
    return simulator.build_circuit(n_trotter_steps)


def second_order_trotter(
    hamiltonian: SparsePauliOp,
    time: float,
    n_trotter_steps: int = 1
) -> QuantumCircuit:
    """
    Convenience function for second-order Trotterization (Suzuki formula).

    Args:
        hamiltonian: Hamiltonian to simulate
        time: Evolution time
        n_trotter_steps: Number of Trotter steps

    Returns:
        QuantumCircuit implementing the evolution
    """
    simulator = TrotterizationSimulator(hamiltonian, time, order=2)
    return simulator.build_circuit(n_trotter_steps)
