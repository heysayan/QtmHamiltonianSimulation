# Quantum Hamiltonian Simulation Framework

A comprehensive implementation and comparison of quantum algorithms for Hamiltonian simulation, with applications to Grover's search algorithm and quantum linear system solving.

## Overview

This repository implements and compares multiple state-of-the-art quantum algorithms for simulating Hamiltonian dynamics:

1. **Trotterization** (Suzuki-Trotter Product Formulas)
2. **Truncated Taylor Series** (Linear Combination of Unitaries - LCU)
3. **Quantum Signal Processing** (QSP)
4. **Qubitization**
5. **Quantum Singular Value Transform** (QSVT)

Additionally, it demonstrates how these Hamiltonian simulation techniques can be applied to:
- **Grover's search algorithm**: Novel implementations using Taylor-LCU and QSVT
- **Quantum linear system solver**: QSVT-based algorithm for solving Ax=b

## Features

- **Multiple Algorithm Implementations**: Complete implementations of 5 different Hamiltonian simulation algorithms
- **Quantum Linear System Solver**: QSVT-based implementation for solving linear equations Ax=b
- **Comprehensive Benchmarking**: Detailed comparison metrics including:
  - Circuit depth
  - Gate count (total, single-qubit, two-qubit, CNOT, T)
  - Qubit requirements (system + ancilla)
  - Query complexity
  - Error estimates
- **Grover's Search via Hamiltonian Simulation**: Novel implementations of Grover's algorithm using Taylor-LCU and QSVT
- **Visualization Tools**: Automated generation of comparison plots and scaling analyses
- **Multiple Test Hamiltonians**: Pre-built test cases (Heisenberg, Transverse Ising, Random)

## Installation

### Prerequisites

- Python 3.8+
- Qiskit 1.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/heysayan/QtmHamiltonianSimulation.git
cd QtmHamiltonianSimulation

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
QtmHamiltonianSimulation/
├── src/
│   ├── algorithms/          # Hamiltonian simulation algorithms
│   │   ├── trotterization.py
│   │   ├── taylor_lcu.py
│   │   ├── qsp.py           # QSP and Qubitization
│   │   ├── qsvt.py
│   │   └── linear_solver_qsvt.py  # Linear system solver
│   ├── grover/              # Grover's algorithm implementations
│   │   ├── standard_grover.py
│   │   └── hamiltonian_grover.py
│   ├── utils/               # Utility functions
│   │   ├── hamiltonian_utils.py
│   │   └── circuit_metrics.py
│   └── benchmarks/          # Benchmarking tools
│       ├── hamiltonian_benchmark.py
│       └── linear_solver_benchmark.py
├── examples/                # Example scripts
│   ├── hamiltonian_simulation_example.py
│   ├── grover_comparison_example.py
│   └── linear_solver_example.py
├── literature/             # Reference papers
└── tests/                  # Unit tests
```

## Quick Start

### Hamiltonian Simulation

```python
from qiskit.quantum_info import SparsePauliOp
from src.algorithms.trotterization import TrotterizationSimulator
from src.algorithms.qsvt import QSVTSimulator
from src.utils.hamiltonian_utils import create_test_hamiltonian

# Create a test Hamiltonian
hamiltonian = create_test_hamiltonian(n_qubits=3, hamiltonian_type="heisenberg")
time = 1.0

# Trotterization
trotter = TrotterizationSimulator(hamiltonian, time, order=1)
trotter_circuit = trotter.build_circuit(n_trotter_steps=10)

# QSVT
qsvt = QSVTSimulator(hamiltonian, time)
qsvt_circuit = qsvt.build_circuit(degree=10)
```

### Comprehensive Benchmark

```python
from src.utils.hamiltonian_utils import create_test_hamiltonian
from src.benchmarks.hamiltonian_benchmark import run_comprehensive_benchmark

hamiltonian = create_test_hamiltonian(n_qubits=3, hamiltonian_type="heisenberg")
df = run_comprehensive_benchmark(hamiltonian, time=1.0, plot_path="comparison.png")
```

### Grover's Algorithm Comparison

```python
from src.grover.hamiltonian_grover import GroverComparison

# Compare standard vs Hamiltonian simulation methods
comparison = GroverComparison(n_qubits=3, marked_states=[7])
results = comparison.compare_all(taylor_order=10, qsvt_degree=10)
comparison.print_comparison_table(results)
```

### Quantum Linear System Solver

```python
from src.algorithms.linear_solver_qsvt import QSVTLinearSolver
import numpy as np

# Define linear system Ax = b
A = np.array([[2, 1], [1, 2]])
b = np.array([3, 3])

# Create solver and build circuit
solver = QSVTLinearSolver(A, b)
circuit = solver.build_circuit(polynomial_degree=10)

# Get classical solution for verification
x = solver.solve()
print(f"Solution: {x}")  # Output: [1, 1]
```

## Running Examples

### Full Hamiltonian Simulation Benchmark

```bash
cd examples
python hamiltonian_simulation_example.py
```

This will:
- Run all 5 algorithms on different Hamiltonians
- Generate comparison tables and plots
- Analyze complexity scaling
- Save results to CSV files

### Grover's Algorithm Comparison

```bash
cd examples
python grover_comparison_example.py
```

This will:
- Compare standard Grover with Hamiltonian simulation variants
- Analyze scaling with problem size
- Generate visualization plots
- Save detailed metrics

### Quantum Linear System Solver

```bash
cd examples
python linear_solver_example.py
```

This will:
- Demonstrate solving linear systems Ax=b using QSVT
- Compare performance across different polynomial degrees
- Analyze scaling with system size
- Generate comparison plots
- Verify solution correctness

## Algorithms

### 1. Trotterization

**Description**: Approximates evolution operator using product formulas.

**Formula**: `exp(-iHt) ≈ [exp(-iH₁t/r)...exp(-iHₘt/r)]^r`

**Complexity**:
- First-order: O((||H||t)²/ε) queries
- Second-order: O((||H||t)^(3/2)/√ε) queries

**Use Case**: Simple, resource-efficient for short time evolution

### 2. Truncated Taylor Series (LCU)

**Description**: Uses Linear Combination of Unitaries to implement truncated Taylor series of exp(-iHt).

**Formula**: `exp(-iHt) ≈ Σ_{k=0}^K (-iHt)^k/k!`

**Complexity**: O(||H||t + log(1/ε)/log log(1/ε)) queries

**Use Case**: Good for small evolution times

### 3. Quantum Signal Processing (QSP)

**Description**: Implements polynomial transformations using signal rotations.

**Complexity**: O(||H||t + log(1/ε)) queries

**Use Case**: Near-optimal for general Hamiltonians

### 4. Qubitization

**Description**: Optimal Hamiltonian simulation using quantum walks.

**Complexity**: O(||H||t + log(1/ε)) queries

**Use Case**: Optimal scaling, requires block encoding

### 5. Quantum Singular Value Transform (QSVT)

**Description**: Most general framework unifying quantum algorithms.

**Complexity**: O(||H||t + log(1/ε)) queries

**Use Case**: Most versatile, optimal scaling, foundation for advanced algorithms

## Comparison Metrics

The framework compares algorithms across multiple dimensions:

| Metric | Description |
|--------|-------------|
| **Circuit Depth** | Critical path length through circuit |
| **Total Gates** | Overall gate count |
| **CNOT Count** | Two-qubit gate count (most expensive) |
| **T-gate Count** | Fault-tolerant implementation cost |
| **Qubit Count** | System + ancilla qubits required |
| **Query Complexity** | Number of oracle/block encoding calls |
| **Error Estimate** | Theoretical approximation error |

## Grover's Search via Hamiltonian Simulation

### Theory

Grover's algorithm can be formulated as Hamiltonian evolution:
- Standard Grover operator G = -D·O where D is diffusion, O is oracle
- Can be expressed as G = exp(-iHt) for specific H
- Evolution time t = π/4 corresponds to one Grover iteration

### Implementations

1. **Standard Grover**: Classical implementation using oracle and diffusion
2. **Taylor-LCU Grover**: Uses truncated Taylor series to approximate G
3. **QSVT Grover**: Uses QSVT to implement polynomial approximation of G

### Comparison Results

Standard Grover is more efficient for the specific task, but Hamiltonian simulation methods demonstrate:
- Generality: Same framework works for any problem expressible as Hamiltonian
- Theoretical importance: Connection between search and simulation
- Advanced techniques: Block encoding, signal processing concepts

## Quantum Linear System Solver

### Theory

The QSVT-based linear system solver solves Ax = b by:
1. Block-encoding matrix A into a unitary operator
2. Using QSVT to apply polynomial P(A) ≈ A^{-1}
3. Preparing state |b⟩ and applying the transformation
4. Obtaining |x⟩ ≈ A^{-1}|b⟩ through measurement

**Complexity**: O(κ log(1/ε)) queries where κ is the condition number and ε is target precision.

### Usage Example

```python
from algorithms.linear_solver_qsvt import QSVTLinearSolver
import numpy as np

# Define linear system Ax = b
A = np.array([[2, 1], [1, 2]])
b = np.array([3, 3])

# Create solver
solver = QSVTLinearSolver(A, b)

# Build quantum circuit
circuit = solver.build_circuit(polynomial_degree=10)

# Get classical solution for verification
x = solver.solve()
print(f"Solution: {x}")  # [1, 1]

# Analyze circuit
print(f"Qubits: {circuit.num_qubits}")
print(f"Depth: {circuit.depth()}")
print(f"Condition number: {solver.condition_number}")
```

### Key Features

- **Automatic degree estimation**: Based on condition number and target error
- **Error bounds**: Theoretical estimates of approximation quality
- **Query complexity tracking**: O(d) for polynomial degree d
- **Regularization**: Cutoff parameter for ill-conditioned matrices
- **Comprehensive metrics**: Circuit depth, gate counts, resource requirements

### Benchmark Results

For a 2×2 well-conditioned system (κ ≈ 3):

| Degree | Qubits | Depth | Gates | Query Complexity | Error |
|--------|--------|-------|-------|------------------|-------|
| 5      | 3      | 10    | 20    | 5                | 8e-1  |
| 10     | 3      | 20    | 35    | 10               | 5e-1  |
| 15     | 3      | 30    | 50    | 15               | 4e-1  |
| 20     | 3      | 40    | 65    | 20               | 4e-1  |

The solver scales efficiently with system size and polynomial degree, maintaining near-optimal query complexity.

## References

### Papers

See `literature/` folder for PDF references:

1. Feynman - "Simulating Physics with Computers" (1982)
2. Lloyd - "Universal Quantum Simulators" (1996)
3. Berry et al. - "Simulating Hamiltonian Dynamics with Truncated Taylor Series" (2015)
4. Low & Chuang - "Optimal Hamiltonian Simulation by Quantum Signal Processing" (2017)
5. Low & Chuang - "Hamiltonian Simulation by Qubitization" (2017)
6. Gilyén et al. - "Quantum Singular Value Transformation" (2019)

### Video Lectures

- [Grand Unification of Quantum Algorithms (QSVT)](https://youtu.be/GFRojXdrVXI?si=ahezO8ljCQnUGMZF) - Isaac Chuang
- [Recent Results in Hamiltonian Simulation](https://youtu.be/PerdRJ-offU?si=OjHU-LaIS-zAo_Qv) - Robin Kothari (2018)
- [General Overview of Quantum Algorithms](https://youtu.be/4jJswyS9ieg?si=NkDejCHK7AAFxwWD) - Robin Kothari
- [Linear Combinations of Unitaries and QSP](https://youtu.be/mWg56DxtDy0?si=HoXIm3edXjM0ZnR7) - Robin Kothari

## Results

### Sample Benchmark Output

```
ALGORITHM COMPARISON
====================================================================================
Algorithm          Qubits  Depth   Total Gates  CNOTs   Query Complexity  Error
====================================================================================
Trotter (order 1)     3      120      450         180         10          2.5e-02
Trotter (order 2)     3      240      890         360         10          6.3e-04
Taylor-LCU            6      180      620         245         10          8.2e-05
QSP                   4      150      580         230         10          8.2e-05
Qubitization          6      200      710         285         15          4.1e-03
QSVT                  7      165      640         255         21          8.2e-05
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional Hamiltonian simulation algorithms
- More sophisticated error analysis
- Quantum resource estimation
- Experimental validation on real quantum hardware
- Additional applications beyond Grover's search

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qtm_hamiltonian_simulation,
  title={Quantum Hamiltonian Simulation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/heysayan/QtmHamiltonianSimulation}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a research and educational framework. For production quantum computing applications, consider using optimized libraries like Qiskit's built-in simulation methods or specialized packages.
