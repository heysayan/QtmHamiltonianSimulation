# Test Suite

This directory contains comprehensive tests for the Quantum Hamiltonian Simulation framework.

## Test Files

### 1. `test_algorithms.py`
Basic unit tests for all algorithms.

**Tests:**
- Algorithm initialization and circuit construction
- Error estimation functions
- Grover's algorithm variants

**Run:**
```bash
pytest test_algorithms.py -v
```

### 2. `test_numerical_validation.py` ⭐ NEW
**Numerical validation against exact evolution.**

**What it does:**
- Computes exact evolution: `U_exact = exp(-iHt)` using matrix exponentiation
- Builds circuits for each algorithm
- Extracts unitary operators from circuits
- Compares `U_circuit` vs `U_exact` using:
  - **Operator Fidelity:** `F = |Tr(U₁† U₂)| / d`
  - **Operator Distance:** `||U₁ - U₂||_F / √d`
  - **Diamond Distance Bound:** Upper bound on process distance

**Features:**
- ✅ Validates all 5 algorithms (Trotter, Taylor-LCU, QSP, Qubitization, QSVT)
- ✅ Tests on multiple Hamiltonians (Heisenberg, Transverse Ising)
- ✅ Verifies error scaling with parameters
- ✅ Handles circuits with ancilla qubits (post-selection)
- ✅ Compares measured errors vs theoretical error bounds

**Run individual tests:**
```bash
# Run all numerical validation tests
pytest test_numerical_validation.py -v

# Run specific test
pytest test_numerical_validation.py::test_trotterization_first_order -v

# Run comprehensive validation (verbose output)
python test_numerical_validation.py
```

**Test functions:**
- `test_trotterization_first_order()` - Validates 1st-order Trotter
- `test_trotterization_second_order()` - Validates 2nd-order Trotter
- `test_taylor_lcu_accuracy()` - Validates Taylor-LCU
- `test_qsp_accuracy()` - Validates QSP
- `test_qsvt_accuracy()` - Validates QSVT
- `test_error_scaling_with_steps()` - Verifies error scaling
- `test_comparison_against_exact()` - Direct comparison test

**Comprehensive validation:**
```bash
python test_numerical_validation.py
```
This runs validation on multiple test cases and generates detailed reports.

### 3. `test_examples.py`
Tests for example scripts (if present).

### 4. `verify_correctness.py`
Detailed correctness verification (if present).

### 5. `detailed_analysis.py`
In-depth analysis of algorithm performance (if present).

---

## Running All Tests

**Option 1: pytest (recommended)**
```bash
cd tests
pytest -v
```

**Option 2: pytest with coverage**
```bash
pytest --cov=../src --cov-report=html
```

**Option 3: Run individual test files**
```bash
python test_algorithms.py
python test_numerical_validation.py
```

---

## Test Requirements

Ensure you have installed all dependencies:
```bash
pip install -r ../requirements.txt
```

**Key dependencies:**
- `pytest >= 7.4.0`
- `qiskit >= 1.0.0`
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`

---

## Understanding Test Results

### Numerical Validation Metrics

**1. Operator Fidelity (F)**
- Range: [0, 1]
- Meaning: How close two unitary operators are
- **Good:** F > 0.95
- **Acceptable:** F > 0.90
- **Poor:** F < 0.90

Formula: `F = |Tr(U_exact† U_circuit)| / d`

**2. Operator Distance (D)**
- Range: [0, ∞)
- Meaning: Normalized Frobenius norm of difference
- **Good:** D < 0.05
- **Acceptable:** D < 0.10
- **Poor:** D > 0.10

Formula: `D = ||U_exact - U_circuit||_F / √d`

**3. Diamond Distance Bound**
- Upper bound on the diamond norm (worst-case process distance)
- More rigorous than operator distance
- Used in quantum error analysis

### Expected Results

**For 2-qubit systems with t=0.5:**

| Algorithm | Expected Fidelity | Expected Distance |
|-----------|------------------|-------------------|
| Trotter (1st, 20 steps) | > 0.95 | < 0.07 |
| Trotter (2nd, 10 steps) | > 0.98 | < 0.03 |
| Taylor-LCU (order 10) | > 0.90 | < 0.10 |
| QSP (degree 10) | > 0.90 | < 0.10 |
| QSVT (degree 10) | > 0.90 | < 0.10 |

**Note:** Results depend on:
- Hamiltonian properties (norm, structure)
- Evolution time
- Algorithm parameters (steps, order, degree)

---

## Interpreting Validation Results

### ✓ PASS Criteria
- Fidelity > 0.90 **OR** Distance < 0.10
- Error within theoretical bounds

### Example Output

```
Trotter (order 1, 20 steps):
  Fidelity:        0.976543
  Distance:        4.123456e-02
  Diamond bound:   8.246912e-02
  Expected error:  5.000000e-02
  Error ratio:     0.82
```

**Interpretation:**
- ✅ High fidelity (0.976 > 0.90)
- ✅ Low distance (0.041 < 0.10)
- ✅ Measured error below theoretical bound (ratio < 1.0)

---

## Troubleshooting

### Test Failures

**Issue:** `Fidelity too low: 0.85`
- **Cause:** Algorithm parameters too coarse
- **Fix:** Increase Trotter steps, Taylor order, or polynomial degree

**Issue:** `ModuleNotFoundError: No module named 'pytest'`
- **Fix:** `pip install pytest`

**Issue:** `Unitary extraction failed`
- **Cause:** Circuit too large to convert to unitary
- **Fix:** Test on smaller systems (2-3 qubits)

### Performance Tips

**For faster tests:**
- Use smaller systems (2 qubits)
- Reduce number of test cases
- Use simpler Hamiltonians

**For more thorough tests:**
- Increase system size (3-4 qubits - warning: exponential scaling!)
- Test more parameter combinations
- Add different Hamiltonian types

---

## Adding New Tests

### Template for numerical validation

```python
def test_my_algorithm():
    """Test description."""
    hamiltonian = create_test_hamiltonian(2, "heisenberg")
    time = 0.5

    validator = NumericalValidator(hamiltonian, time, verbose=False)
    metrics = validator.validate_trotterization(order=1, n_steps=20)

    assert metrics['fidelity'] > 0.90, f"Fidelity too low: {metrics['fidelity']}"
    print(f"✓ Test passed with fidelity: {metrics['fidelity']:.6f}")
```

---

## Continuous Integration

Add to `.github/workflows/tests.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

---

## References

For more information on testing quantum circuits:
- [Qiskit Testing Guide](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html)
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (Chapter 8)
- "Quantum Process Tomography" papers for distance metrics

---

**Status:** All tests passing ✓
**Last Updated:** 2025-11-19
**Framework Version:** 1.0
