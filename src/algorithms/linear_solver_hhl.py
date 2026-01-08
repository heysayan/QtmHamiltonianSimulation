"""
HHL Algorithm for Linear System Solving.

The Harrow-Hassidim-Lloyd (HHL) algorithm is a quantum algorithm for solving
linear systems Ax=b, providing exponential speedup for certain matrices.

Reference: "Quantum algorithm for linear systems of equations" (Harrow, Hassidim, Lloyd, 2009)

Key idea:
1. Prepare state |b⟩
2. Apply Quantum Phase Estimation (QPE) to A
3. Controlled rotation based on eigenvalues (implements A^{-1})
4. Uncompute QPE
5. Measure to obtain |x⟩ ∝ A^{-1}|b⟩

Complexity: O(log(N) s^2 κ^2 / ε) where:
- N: dimension of matrix
- s: sparsity
- κ: condition number  
- ε: precision
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from typing import Tuple, Optional
import warnings


class HHLSolver:
    """
    Harrow-Hassidim-Lloyd (HHL) algorithm for linear system solving.
    
    Solves Ax = b for Hermitian matrix A using quantum phase estimation
    and controlled rotations.
    """
    
    def __init__(self, matrix_A: np.ndarray, vector_b: np.ndarray):
        """
        Initialize HHL solver.
        
        Args:
            matrix_A: Coefficient matrix A (should be Hermitian)
            vector_b: Right-hand side vector b
            
        Raises:
            ValueError: If dimensions are incompatible
        """
        self.A = np.array(matrix_A, dtype=complex)
        self.b = np.array(vector_b, dtype=complex).flatten()
        
        # Validate dimensions
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.A.shape[0] != len(self.b):
            raise ValueError("Matrix and vector dimensions must match")
        
        self.n = self.A.shape[0]
        self.n_qubits = int(np.ceil(np.log2(self.n)))
        
        # Check if matrix is Hermitian
        if not np.allclose(self.A, self.A.conj().T):
            warnings.warn("Matrix A is not Hermitian. HHL requires Hermitian matrices.")
        
        # Compute matrix properties
        self.eigenvalues = self._compute_eigenvalues()
        self.condition_number = self._compute_condition_number()
        
        # Normalize b vector
        self.b_norm = np.linalg.norm(self.b)
        if self.b_norm > 0:
            self.b_normalized = self.b / self.b_norm
        else:
            self.b_normalized = self.b
        
        # Number of qubits for QPE (controls precision)
        self.n_qpe_qubits = max(4, int(np.ceil(np.log2(self.condition_number))) + 2)
    
    def _compute_eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of matrix A."""
        eigenvalues = np.linalg.eigvalsh(self.A)
        return eigenvalues
    
    def _compute_condition_number(self) -> float:
        """Compute condition number κ = λ_max / λ_min."""
        eig = self.eigenvalues
        eig_nonzero = eig[np.abs(eig) > 1e-10]
        if len(eig_nonzero) > 0:
            kappa = np.max(np.abs(eig_nonzero)) / np.min(np.abs(eig_nonzero))
            return float(kappa)
        return float('inf')
    
    def build_circuit(self, use_qpe: bool = True) -> QuantumCircuit:
        """
        Build HHL quantum circuit.
        
        Args:
            use_qpe: Whether to use full QPE (simplified version if False)
            
        Returns:
            QuantumCircuit implementing HHL
        """
        # Create quantum registers
        # System qubits for |b⟩ and solution |x⟩
        system_reg = QuantumRegister(self.n_qubits, 'system')
        
        # QPE qubits (clock register)
        qpe_reg = QuantumRegister(self.n_qpe_qubits, 'qpe')
        
        # Ancilla qubit for controlled rotation
        ancilla_reg = AncillaRegister(1, 'ancilla')
        
        # Classical register for measurement
        classical_reg = ClassicalRegister(1, 'c')
        
        qc = QuantumCircuit(system_reg, qpe_reg, ancilla_reg, classical_reg)
        
        # Step 1: Prepare initial state |b⟩
        self._prepare_state_b(qc, system_reg)
        
        # Step 2: Quantum Phase Estimation
        if use_qpe:
            self._apply_qpe(qc, system_reg, qpe_reg)
        else:
            # Simplified version without full QPE
            self._apply_simplified_eigenvalue_estimation(qc, system_reg, qpe_reg)
        
        # Step 3: Controlled rotation (implements A^{-1})
        self._apply_controlled_rotation(qc, qpe_reg, ancilla_reg)
        
        # Step 4: Inverse QPE
        if use_qpe:
            self._apply_inverse_qpe(qc, system_reg, qpe_reg)
        
        # Step 5: Measure ancilla (post-selection for success)
        qc.measure(ancilla_reg[0], classical_reg[0])
        
        return qc
    
    def _prepare_state_b(self, qc: QuantumCircuit, system_reg: QuantumRegister):
        """
        Prepare initial state |b⟩.
        
        Args:
            qc: QuantumCircuit to append to
            system_reg: System qubit register
        """
        # Pad vector to power of 2
        padded_size = 2 ** self.n_qubits
        b_padded = np.zeros(padded_size, dtype=complex)
        b_padded[:len(self.b_normalized)] = self.b_normalized
        
        # Normalize
        norm = np.linalg.norm(b_padded)
        if norm > 0:
            b_padded = b_padded / norm
        
        # Use state initialization
        qc.initialize(b_padded, system_reg)
    
    def _apply_qpe(self, qc: QuantumCircuit, system_reg: QuantumRegister, 
                   qpe_reg: QuantumRegister):
        """
        Apply Quantum Phase Estimation.
        
        Args:
            qc: QuantumCircuit to append to
            system_reg: System qubit register
            qpe_reg: QPE qubit register (clock)
        """
        # Initialize QPE qubits in superposition
        for q in qpe_reg:
            qc.h(q)
        
        # Controlled-U^{2^k} operations
        # This is simplified - full implementation requires efficient
        # implementation of controlled time evolution
        for k, q in enumerate(qpe_reg):
            # Controlled-A^{2^k} operation
            power = 2**k
            self._apply_controlled_hamiltonian_evolution(qc, system_reg, q, power)
        
        # Inverse QFT
        qft = QFT(num_qubits=self.n_qpe_qubits, inverse=True, do_swaps=False)
        qc.append(qft, qpe_reg)
    
    def _apply_simplified_eigenvalue_estimation(
        self, qc: QuantumCircuit, system_reg: QuantumRegister,
        qpe_reg: QuantumRegister
    ):
        """
        Simplified eigenvalue estimation (not full QPE).
        
        Args:
            qc: QuantumCircuit
            system_reg: System register
            qpe_reg: QPE register
        """
        # Simplified version: prepare superposition and apply controlled evolution
        for q in qpe_reg:
            qc.h(q)
        
        # Apply a few controlled evolutions
        for k in range(min(3, len(qpe_reg))):
            self._apply_controlled_hamiltonian_evolution(qc, system_reg, qpe_reg[k], 2**k)
    
    def _apply_controlled_hamiltonian_evolution(
        self, qc: QuantumCircuit, system_reg: QuantumRegister,
        control_qubit: int, power: int
    ):
        """
        Apply controlled-U^{power} where U = e^{iAt}.
        
        This is a simplified placeholder. Full implementation would require
        Hamiltonian simulation techniques.
        
        Args:
            qc: QuantumCircuit
            system_reg: System register
            control_qubit: Control qubit
            power: Power of evolution operator
        """
        # Simplified: apply some controlled rotations
        # In practice, this requires Hamiltonian simulation (e.g., Trotterization)
        for i, q in enumerate(system_reg):
            # Approximate controlled evolution with rotations
            angle = 2 * np.pi * power / (2**self.n_qpe_qubits)
            qc.cp(angle, control_qubit, q)
    
    def _apply_controlled_rotation(
        self, qc: QuantumCircuit, qpe_reg: QuantumRegister,
        ancilla_reg: AncillaRegister
    ):
        """
        Apply controlled rotation to implement A^{-1}.
        
        The rotation angle is inversely proportional to the eigenvalue.
        
        Args:
            qc: QuantumCircuit
            qpe_reg: QPE register encoding eigenvalue
            ancilla_reg: Ancilla for rotation
        """
        # For each eigenvalue λ encoded in QPE register,
        # rotate ancilla by angle ∝ 1/λ
        
        # Simplified implementation using multi-controlled rotations
        # Full implementation requires more sophisticated approach
        
        # Apply Ry rotation controlled on QPE register
        # Angle ∝ C/λ where C is normalization constant
        C = 1.0  # Normalization constant
        
        # Multi-controlled rotation (simplified)
        # In practice, need to decompose into native gates
        for i, q in enumerate(qpe_reg):
            # Approximate rotation based on qubit significance
            angle = C * np.pi / (2**(i+1))
            qc.cry(angle, q, ancilla_reg[0])
    
    def _apply_inverse_qpe(
        self, qc: QuantumCircuit, system_reg: QuantumRegister,
        qpe_reg: QuantumRegister
    ):
        """
        Apply inverse QPE to uncompute eigenvalue.
        
        Args:
            qc: QuantumCircuit
            system_reg: System register
            qpe_reg: QPE register
        """
        # Forward QFT
        qft = QFT(num_qubits=self.n_qpe_qubits, inverse=False, do_swaps=False)
        qc.append(qft, qpe_reg)
        
        # Inverse controlled operations
        for k in range(len(qpe_reg)-1, -1, -1):
            power = 2**k
            self._apply_controlled_hamiltonian_evolution(qc, system_reg, qpe_reg[k], -power)
        
        # Hadamards
        for q in qpe_reg:
            qc.h(q)
    
    def solve(self) -> np.ndarray:
        """
        Solve the linear system classically for verification.
        
        Returns:
            Solution vector x
        """
        try:
            x_classical = np.linalg.solve(self.A, self.b)
            return x_classical
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            x_classical = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
            return x_classical
    
    def estimate_error(self, n_qpe_qubits: int = None) -> float:
        """
        Estimate approximation error.
        
        Args:
            n_qpe_qubits: Number of QPE qubits (default: self.n_qpe_qubits)
            
        Returns:
            Estimated error bound
        """
        if n_qpe_qubits is None:
            n_qpe_qubits = self.n_qpe_qubits
        
        # Error from QPE
        qpe_error = 2 * np.pi / (2**n_qpe_qubits)
        
        # Error from rotation approximation
        rotation_error = 1.0 / (2**n_qpe_qubits)
        
        # Total error (approximate)
        total_error = self.condition_number * (qpe_error + rotation_error)
        
        return float(total_error)
    
    def get_query_complexity(self, n_qpe_qubits: int = None) -> int:
        """
        Compute query complexity.
        
        For HHL: O(log(N) s^2 κ^2 / ε)
        
        Args:
            n_qpe_qubits: Number of QPE qubits
            
        Returns:
            Estimated number of queries
        """
        if n_qpe_qubits is None:
            n_qpe_qubits = self.n_qpe_qubits
        
        # Simplified estimate
        # QPE requires O(2^n_qpe) controlled operations
        qpe_queries = 2 * n_qpe_qubits
        
        # Each controlled operation requires Hamiltonian simulation
        # which depends on sparsity s and condition number κ
        # Simplified: assume sparse matrix with s = O(1)
        
        # Total queries (simplified estimate)
        queries = qpe_queries * int(np.ceil(self.condition_number))
        
        return queries
    
    def get_circuit_metrics(self, circuit: QuantumCircuit) -> dict:
        """
        Extract metrics from HHL circuit.
        
        Args:
            circuit: The quantum circuit
            
        Returns:
            Dictionary of circuit metrics
        """
        from utils.circuit_metrics import analyze_circuit
        
        metrics = analyze_circuit(circuit)
        
        # Add HHL-specific metrics
        metrics['condition_number'] = self.condition_number
        metrics['matrix_dimension'] = self.n
        metrics['system_qubits'] = self.n_qubits
        metrics['qpe_qubits'] = self.n_qpe_qubits
        metrics['ancilla_qubits'] = 1
        
        return metrics


def solve_linear_system_hhl(
    A: np.ndarray,
    b: np.ndarray,
    use_qpe: bool = True
) -> Tuple[QuantumCircuit, dict]:
    """
    Convenience function to solve linear system using HHL.
    
    Args:
        A: Coefficient matrix (should be Hermitian)
        b: Right-hand side vector
        use_qpe: Whether to use full QPE
        
    Returns:
        Tuple of (quantum circuit, metadata dict)
    """
    solver = HHLSolver(A, b)
    
    circuit = solver.build_circuit(use_qpe)
    
    metadata = {
        'condition_number': solver.condition_number,
        'query_complexity': solver.get_query_complexity(),
        'estimated_error': solver.estimate_error(),
        'matrix_dimension': solver.n,
        'system_qubits': solver.n_qubits,
        'qpe_qubits': solver.n_qpe_qubits,
        'ancilla_qubits': 1,
    }
    
    return circuit, metadata
