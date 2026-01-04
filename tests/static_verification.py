#!/usr/bin/env python3
"""
Static verification of test suite structure and completeness.
Verifies test files without requiring dependencies to be installed.
"""

import os
import ast
import sys

def analyze_test_file(filepath):
    """Analyze a test file and extract test functions."""
    with open(filepath, 'r') as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

    test_functions = []
    classes = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('test_'):
                test_functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return {
        'test_functions': test_functions,
        'classes': classes,
        'imports': imports
    }, None


def verify_numerical_validation():
    """Verify numerical validation test file."""
    print("\n" + "="*80)
    print("VERIFYING: test_numerical_validation.py")
    print("="*80)

    filepath = 'tests/test_numerical_validation.py'
    info, error = analyze_test_file(filepath)

    if error:
        print(f"✗ Failed: {error}")
        return False

    print(f"✓ Syntax: Valid")
    print(f"✓ Test functions found: {len(info['test_functions'])}")

    expected_tests = [
        'test_trotterization_first_order',
        'test_trotterization_second_order',
        'test_taylor_lcu_accuracy',
        'test_qsp_accuracy',
        'test_qsvt_accuracy',
        'test_error_scaling_with_steps',
        'test_comparison_against_exact'
    ]

    for test in expected_tests:
        if test in info['test_functions']:
            print(f"  ✓ {test}")
        else:
            print(f"  ✗ Missing: {test}")
            return False

    expected_classes = ['NumericalValidator']
    for cls in expected_classes:
        if cls in info['classes']:
            print(f"✓ Class: {cls}")
        else:
            print(f"✗ Missing class: {cls}")
            return False

    expected_functions = [
        'compute_exact_evolution',
        'extract_unitary_from_circuit',
        'compute_operator_fidelity',
        'compute_operator_distance',
        'compute_diamond_distance_bound'
    ]

    # Check if main function exists
    with open(filepath, 'r') as f:
        content = f.read()

    for func in expected_functions:
        if f"def {func}(" in content:
            print(f"✓ Function: {func}")
        else:
            print(f"✗ Missing function: {func}")
            return False

    # Check for comprehensive validation function
    if 'run_comprehensive_validation' in content:
        print(f"✓ Function: run_comprehensive_validation")
    else:
        print(f"✗ Missing function: run_comprehensive_validation")
        return False

    return True


def verify_basic_tests():
    """Verify basic test file."""
    print("\n" + "="*80)
    print("VERIFYING: test_algorithms.py")
    print("="*80)

    filepath = 'tests/test_algorithms.py'
    info, error = analyze_test_file(filepath)

    if error:
        print(f"✗ Failed: {error}")
        return False

    print(f"✓ Syntax: Valid")
    print(f"✓ Test functions found: {len(info['test_functions'])}")

    expected_tests = [
        'test_create_test_hamiltonians',
        'test_trotterization',
        'test_taylor_lcu',
        'test_qsp',
        'test_qubitization',
        'test_qsvt'
    ]

    for test in expected_tests:
        if test in info['test_functions']:
            print(f"  ✓ {test}")
        else:
            print(f"  ✗ Missing: {test}")
            return False

    return True


def verify_test_documentation():
    """Verify test documentation exists."""
    print("\n" + "="*80)
    print("VERIFYING: Test Documentation")
    print("="*80)

    readme_path = 'tests/README.md'
    if os.path.exists(readme_path):
        print(f"✓ {readme_path} exists")

        with open(readme_path, 'r') as f:
            content = f.read()

        required_sections = [
            'test_numerical_validation.py',
            'Operator Fidelity',
            'Operator Distance',
            'Running All Tests',  # Usage section
            'Expected Results'
        ]

        for section in required_sections:
            if section in content:
                print(f"  ✓ Contains: {section}")
            else:
                print(f"  ✗ Missing: {section}")
                return False

        return True
    else:
        print(f"✗ {readme_path} not found")
        return False


def verify_algorithm_implementations():
    """Verify all algorithm files are present and valid."""
    print("\n" + "="*80)
    print("VERIFYING: Algorithm Implementations")
    print("="*80)

    algorithms = [
        'src/algorithms/trotterization.py',
        'src/algorithms/taylor_lcu.py',
        'src/algorithms/qsp.py',
        'src/algorithms/qsvt.py'
    ]

    all_valid = True

    for algo_file in algorithms:
        if os.path.exists(algo_file):
            info, error = analyze_test_file(algo_file)
            if error:
                print(f"✗ {algo_file}: {error}")
                all_valid = False
            else:
                print(f"✓ {algo_file}: Valid syntax, {len(info['classes'])} classes")
        else:
            print(f"✗ {algo_file}: Not found")
            all_valid = False

    return all_valid


def verify_utils():
    """Verify utility files."""
    print("\n" + "="*80)
    print("VERIFYING: Utility Modules")
    print("="*80)

    utils = [
        'src/utils/hamiltonian_utils.py',
        'src/utils/circuit_metrics.py'
    ]

    all_valid = True

    for util_file in utils:
        if os.path.exists(util_file):
            info, error = analyze_test_file(util_file)
            if error:
                print(f"✗ {util_file}: {error}")
                all_valid = False
            else:
                print(f"✓ {util_file}: Valid syntax")
        else:
            print(f"✗ {util_file}: Not found")
            all_valid = False

    # Check for scipy import in hamiltonian_utils.py
    with open('src/utils/hamiltonian_utils.py', 'r') as f:
        content = f.read()

    if 'import scipy.linalg' in content:
        print("✓ hamiltonian_utils.py: scipy.linalg imported")
    else:
        print("✗ hamiltonian_utils.py: Missing scipy.linalg import")
        all_valid = False

    return all_valid


def count_lines_of_code():
    """Count lines of code in the project."""
    print("\n" + "="*80)
    print("CODE STATISTICS")
    print("="*80)

    stats = {
        'Algorithm implementations': 0,
        'Test files': 0,
        'Utility modules': 0,
        'Examples': 0
    }

    for root, dirs, files in os.walk('.'):
        for file in files:
            if not file.endswith('.py'):
                continue

            filepath = os.path.join(root, file)

            try:
                with open(filepath, 'r') as f:
                    lines = len(f.readlines())

                if 'src/algorithms' in filepath:
                    stats['Algorithm implementations'] += lines
                elif 'tests' in filepath:
                    stats['Test files'] += lines
                elif 'src/utils' in filepath:
                    stats['Utility modules'] += lines
                elif 'examples' in filepath:
                    stats['Examples'] += lines
            except:
                pass

    for category, lines in stats.items():
        print(f"{category:<30} {lines:>6} lines")

    print(f"{'Total':<30} {sum(stats.values()):>6} lines")


def main():
    """Main verification routine."""
    print("\n" + "="*80)
    print("STATIC VERIFICATION OF TEST SUITE")
    print("="*80)

    results = []

    # Verify algorithm implementations
    results.append(("Algorithm Implementations", verify_algorithm_implementations()))

    # Verify utils
    results.append(("Utility Modules", verify_utils()))

    # Verify basic tests
    results.append(("Basic Unit Tests", verify_basic_tests()))

    # Verify numerical validation
    results.append(("Numerical Validation", verify_numerical_validation()))

    # Verify documentation
    results.append(("Test Documentation", verify_test_documentation()))

    # Statistics
    count_lines_of_code()

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<30} {status}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n✓ ALL VERIFICATIONS PASSED")
        print("  - All files have valid syntax")
        print("  - All required tests are present")
        print("  - All algorithm implementations exist")
        print("  - Documentation is complete")
        print("  - Code is ready for testing (pending dependency installation)")
        return 0
    else:
        print("\n✗ SOME VERIFICATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
