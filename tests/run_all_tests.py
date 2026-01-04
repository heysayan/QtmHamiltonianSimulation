#!/usr/bin/env python3
"""
Comprehensive test runner for Quantum Hamiltonian Simulation Framework.
Runs all tests when pytest is not available.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_test_function(test_func, test_name):
    """Run a single test function and report results."""
    try:
        test_func()
        print(f"  ✓ {test_name}")
        return True
    except AssertionError as e:
        print(f"  ✗ {test_name}")
        print(f"    Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ {test_name} (Exception)")
        print(f"    Error: {type(e).__name__}: {e}")
        return False


def run_basic_unit_tests():
    """Run basic unit tests from test_algorithms.py"""
    print("\n" + "="*80)
    print("RUNNING BASIC UNIT TESTS")
    print("="*80)

    from test_algorithms import (
        test_create_test_hamiltonians,
        test_trotterization,
        test_taylor_lcu,
        test_qsp,
        test_qubitization,
        test_qsvt,
        test_grover_standard,
        test_grover_via_taylor,
        test_grover_via_qsvt
    )

    tests = [
        (test_create_test_hamiltonians, "test_create_test_hamiltonians"),
        (test_trotterization, "test_trotterization"),
        (test_taylor_lcu, "test_taylor_lcu"),
        (test_qsp, "test_qsp"),
        (test_qubitization, "test_qubitization"),
        (test_qsvt, "test_qsvt"),
        (test_grover_standard, "test_grover_standard"),
        (test_grover_via_taylor, "test_grover_via_taylor"),
        (test_grover_via_qsvt, "test_grover_via_qsvt"),
    ]

    passed = 0
    failed = 0

    for test_func, test_name in tests:
        if run_test_function(test_func, test_name):
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def run_numerical_validation_tests():
    """Run numerical validation tests."""
    print("\n" + "="*80)
    print("RUNNING NUMERICAL VALIDATION TESTS")
    print("="*80)

    from test_numerical_validation import (
        test_trotterization_first_order,
        test_trotterization_second_order,
        test_taylor_lcu_accuracy,
        test_qsp_accuracy,
        test_qsvt_accuracy,
        test_error_scaling_with_steps,
        test_comparison_against_exact
    )

    tests = [
        (test_trotterization_first_order, "test_trotterization_first_order"),
        (test_trotterization_second_order, "test_trotterization_second_order"),
        (test_taylor_lcu_accuracy, "test_taylor_lcu_accuracy"),
        (test_qsp_accuracy, "test_qsp_accuracy"),
        (test_qsvt_accuracy, "test_qsvt_accuracy"),
        (test_error_scaling_with_steps, "test_error_scaling_with_steps"),
        (test_comparison_against_exact, "test_comparison_against_exact"),
    ]

    passed = 0
    failed = 0

    for test_func, test_name in tests:
        if run_test_function(test_func, test_name):
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def run_comprehensive_validation():
    """Run comprehensive numerical validation."""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE NUMERICAL VALIDATION")
    print("="*80)

    try:
        from test_numerical_validation import run_comprehensive_validation
        run_comprehensive_validation()
        print("\n✓ Comprehensive validation completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Comprehensive validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    print("\n" + "="*80)
    print("QUANTUM HAMILTONIAN SIMULATION - TEST SUITE")
    print("="*80)

    all_passed = True

    # Run basic unit tests
    if not run_basic_unit_tests():
        all_passed = False

    # Run numerical validation tests
    if not run_numerical_validation_tests():
        all_passed = False

    # Run comprehensive validation (verbose)
    if not run_comprehensive_validation():
        all_passed = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*80 + "\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
