"""
Generate diagrams and visualizations for the presentation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11


def create_algorithm_timeline():
    """Create timeline of Hamiltonian simulation algorithms."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Timeline data
    algorithms = [
        (1982, "Feynman\nQuantum Simulation Idea", 'lightblue'),
        (1996, "Lloyd\nUniversal Quantum Simulator", 'lightgreen'),
        (2006, "Berry et al.\nTrotterization Analysis", 'lightyellow'),
        (2015, "Berry et al.\nTaylor Series (LCU)", 'lightcoral'),
        (2017, "Low & Chuang\nQSP & Qubitization", 'plum'),
        (2019, "Gilyén et al.\nQSVT", 'gold'),
    ]

    years = [alg[0] for alg in algorithms]
    y_positions = [0, 1, 0, 1, 0, 1]  # Alternate heights

    # Draw timeline
    ax.plot([1980, 2020], [0.5, 0.5], 'k-', linewidth=2, zorder=1)

    # Add algorithms
    for (year, name, color), y_pos in zip(algorithms, y_positions):
        # Draw vertical line
        ax.plot([year, year], [0.5, y_pos], 'k--', linewidth=1, alpha=0.5)

        # Draw box
        box = FancyBboxPatch((year - 1.5, y_pos - 0.15), 3, 0.3,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)

        # Add text
        ax.text(year, y_pos, name, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Styling
    ax.set_xlim(1978, 2022)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    ax.set_title('Evolution of Hamiltonian Simulation Algorithms',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('algorithm_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Created algorithm_timeline.png")
    plt.close()


def create_complexity_comparison():
    """Create complexity comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 7))

    algorithms = ['Trotter\n(1st order)', 'Trotter\n(2nd order)',
                  'Taylor-LCU', 'QSP', 'Qubitization', 'QSVT']

    # Query complexity (big-O notation values for t=1, epsilon=0.01)
    # These are approximate relative values
    complexity = [100, 30, 20, 15, 12, 12]

    colors = ['#3498db', '#2980b9', '#e74c3c', '#9b59b6', '#1abc9c', '#f39c12']

    bars = ax.bar(algorithms, complexity, color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, complexity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Relative Query Complexity', fontsize=13, fontweight='bold')
    ax.set_title('Query Complexity Comparison\n(Lower is Better)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(complexity) * 1.2)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('complexity_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created complexity_comparison.png")
    plt.close()


def create_algorithm_comparison_radar():
    """Create radar chart comparing algorithm properties."""
    from math import pi

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Categories
    categories = ['Circuit\nDepth', 'Gate\nCount', 'Qubit\nEfficiency',
                  'Error\nBounds', 'Generality']
    N = len(categories)

    # Algorithm scores (0-10 scale, higher is better)
    algorithms_data = {
        'Trotter': [7, 8, 9, 5, 7],
        'Taylor-LCU': [6, 6, 6, 7, 8],
        'QSP': [8, 7, 7, 8, 9],
        'QSVT': [9, 8, 8, 9, 10],
    }

    colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12']

    # Compute angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Plot each algorithm
    for (name, values), color in zip(algorithms_data.items(), colors):
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Algorithm Properties Comparison',
              fontsize=16, fontweight='bold', pad=20, y=1.08)

    plt.tight_layout()
    plt.savefig('algorithm_radar.png', dpi=300, bbox_inches='tight')
    print("✓ Created algorithm_radar.png")
    plt.close()


def create_trotter_diagram():
    """Create diagram explaining Trotterization."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Title
    ax.text(0.5, 0.95, 'Trotterization: Product Formula Approximation',
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Exact evolution
    box1 = FancyBboxPatch((0.05, 0.6), 0.35, 0.25,
                          boxstyle="round,pad=0.02",
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(0.225, 0.725, 'Exact Evolution', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.225, 0.65, r'$e^{-iHt} = e^{-i(H_1+H_2+...+H_m)t}$',
            ha='center', fontsize=11)

    # Arrow
    arrow = FancyArrowPatch((0.42, 0.725), (0.58, 0.725),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color='red')
    ax.add_patch(arrow)
    ax.text(0.5, 0.78, 'Approximate', ha='center', fontsize=11,
            fontweight='bold', color='red')

    # Trotter approximation
    box2 = FancyBboxPatch((0.6, 0.6), 0.35, 0.25,
                          boxstyle="round,pad=0.02",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(0.775, 0.725, 'Trotter Formula', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.775, 0.65, r'$\approx [e^{-iH_1t/r}e^{-iH_2t/r}...e^{-iH_mt/r}]^r$',
            ha='center', fontsize=11)

    # Steps visualization
    ax.text(0.5, 0.45, 'Trotter Steps (r repetitions):',
            ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Draw step boxes
    step_width = 0.12
    start_x = 0.14
    for i in range(5):
        box = FancyBboxPatch((start_x + i*0.16, 0.15), step_width, 0.15,
                            boxstyle="round,pad=0.01",
                            facecolor='lightyellow', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(start_x + i*0.16 + step_width/2, 0.225, f'Step {i+1}',
                ha='center', fontsize=9)

    ax.text(0.9, 0.225, '...', ha='center', fontsize=14, fontweight='bold')

    # Error note
    ax.text(0.5, 0.05, r'Error: $O(t^2/r)$ for 1st order, $O(t^3/r^2)$ for 2nd order',
            ha='center', fontsize=11, style='italic',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('trotter_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Created trotter_diagram.png")
    plt.close()


def create_lcu_diagram():
    """Create diagram explaining Linear Combination of Unitaries."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Title
    ax.text(0.5, 0.95, 'Linear Combination of Unitaries (LCU)',
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Hamiltonian decomposition
    ax.text(0.5, 0.85, r'Hamiltonian: $H = \sum_{j=1}^m \alpha_j U_j$',
            ha='center', fontsize=13, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Block encoding
    ax.text(0.1, 0.7, 'Block Encoding:', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Large unitary matrix
    large_box = Rectangle((0.15, 0.35), 0.25, 0.25,
                          facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(large_box)

    # H block inside
    h_block = Rectangle((0.15, 0.47), 0.1, 0.1,
                        facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(h_block)

    ax.text(0.275, 0.475, r'Block: $H/\alpha$ + padding',
            ha='center', va='center', fontsize=11, transform=ax.transAxes)

    ax.text(0.275, 0.3, 'Unitary U', ha='center', fontsize=11,
            fontweight='bold', transform=ax.transAxes)

    # Arrow
    arrow = FancyArrowPatch((0.42, 0.475), (0.53, 0.475),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='blue',
                           transform=ax.transAxes)
    ax.add_patch(arrow)

    # PREPARE-SELECT-PREPARE† structure
    ax.text(0.7, 0.7, 'LCU Circuit Structure:', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Circuit boxes
    boxes = [
        (0.55, 0.5, 'PREPARE', 'lightgreen'),
        (0.65, 0.5, 'SELECT', 'lightyellow'),
        (0.75, 0.5, 'PREPARE†', 'lightgreen'),
    ]

    for x, y, label, color in boxes:
        box = FancyBboxPatch((x, y), 0.08, 0.12,
                            boxstyle="round,pad=0.01",
                            facecolor=color, edgecolor='black', linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + 0.04, y + 0.06, label, ha='center', va='center',
                fontsize=9, fontweight='bold', transform=ax.transAxes)

    # Annotations
    ax.text(0.59, 0.42, 'Encode\ncoefficients', ha='center', fontsize=8,
            transform=ax.transAxes)
    ax.text(0.69, 0.42, 'Apply\nunitaries', ha='center', fontsize=8,
            transform=ax.transAxes)
    ax.text(0.79, 0.42, 'Uncompute', ha='center', fontsize=8,
            transform=ax.transAxes)

    # Taylor series application
    ax.text(0.5, 0.25, 'Taylor Series Approximation:', fontsize=12,
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.18, r'$e^{-iHt} \approx \sum_{k=0}^K \frac{(-iHt)^k}{k!}$',
            ha='center', fontsize=13, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Advantages
    ax.text(0.5, 0.08, 'Advantages: Good for sparse Hamiltonians, systematic error control',
            ha='center', fontsize=10, style='italic', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('lcu_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Created lcu_diagram.png")
    plt.close()


def create_qsvt_diagram():
    """Create diagram explaining QSVT framework."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Title
    ax.text(0.5, 0.96, 'Quantum Singular Value Transform (QSVT)',
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.92, 'The Grand Unification of Quantum Algorithms',
            ha='center', fontsize=12, style='italic', transform=ax.transAxes)

    # Central QSVT box
    qsvt_box = FancyBboxPatch((0.35, 0.65), 0.3, 0.15,
                              boxstyle="round,pad=0.02",
                              facecolor='gold', edgecolor='black', linewidth=3)
    ax.add_patch(qsvt_box)
    ax.text(0.5, 0.775, 'QSVT', ha='center', va='center',
            fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.71, 'Polynomial transformation\nof singular values',
            ha='center', va='center', fontsize=10, transform=ax.transAxes)

    # Applications around QSVT
    applications = [
        (0.15, 0.75, 'Hamiltonian\nSimulation', 'lightblue'),
        (0.15, 0.55, 'Amplitude\nAmplification', 'lightgreen'),
        (0.5, 0.5, 'Linear Systems', 'lightcoral'),
        (0.85, 0.55, 'Quantum\nSearch', 'plum'),
        (0.85, 0.75, 'Matrix\nInversion', 'lightyellow'),
    ]

    for x, y, name, color in applications:
        # Draw arrow from application to QSVT
        if x < 0.5:
            arrow = FancyArrowPatch((x + 0.08, y), (0.35, 0.725),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='gray',
                                   transform=ax.transAxes)
        elif x > 0.5:
            arrow = FancyArrowPatch((x - 0.08, y), (0.65, 0.725),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='gray',
                                   transform=ax.transAxes)
        else:
            arrow = FancyArrowPatch((x, y + 0.08), (0.5, 0.65),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='gray',
                                   transform=ax.transAxes)
        ax.add_patch(arrow)

        # Draw application box
        box = FancyBboxPatch((x - 0.07, y - 0.05), 0.14, 0.1,
                            boxstyle="round,pad=0.01",
                            facecolor=color, edgecolor='black', linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=9, fontweight='bold', transform=ax.transAxes)

    # QSVT sequence explanation
    ax.text(0.5, 0.38, 'QSVT Sequence:', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Draw phase sequence
    phases = ['φ₀', 'U', 'φ₁', 'U†', 'φ₂', 'U', 'φ₃', '...', 'φₖ']
    x_start = 0.15
    for i, phase in enumerate(phases):
        if phase.startswith('φ'):
            color = 'lightcoral'
            label = phase
        elif phase == 'U' or phase == 'U†':
            color = 'lightblue'
            label = phase
        else:
            color = 'white'
            label = phase

        box = FancyBboxPatch((x_start + i*0.08, 0.26), 0.06, 0.08,
                            boxstyle="round,pad=0.005",
                            facecolor=color, edgecolor='black', linewidth=1.5,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x_start + i*0.08 + 0.03, 0.3, label, ha='center', va='center',
                fontsize=9, transform=ax.transAxes)

    # Legend
    ax.text(0.15, 0.18, 'φᵢ: Signal processing rotations',
            fontsize=10, transform=ax.transAxes)
    ax.text(0.15, 0.14, 'U: Block-encoded operator',
            fontsize=10, transform=ax.transAxes)

    # Key properties
    properties = [
        'Optimal query complexity: O(d) where d is polynomial degree',
        'Achieves Heisenberg-limited precision',
        'Unifies QSP, Qubitization, Amplitude Amplification',
    ]

    y_pos = 0.08
    for i, prop in enumerate(properties):
        ax.text(0.5, y_pos - i*0.04, f'• {prop}',
                ha='center', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('qsvt_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Created qsvt_diagram.png")
    plt.close()


def create_grover_hamiltonian_diagram():
    """Create diagram showing Grover as Hamiltonian simulation."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Title
    ax.text(0.5, 0.95, "Grover's Algorithm via Hamiltonian Simulation",
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Standard Grover
    box1 = FancyBboxPatch((0.05, 0.7), 0.25, 0.15,
                          boxstyle="round,pad=0.02",
                          facecolor='lightblue', edgecolor='black', linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(box1)
    ax.text(0.175, 0.82, 'Standard Grover', ha='center', fontsize=12,
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.175, 0.765, 'G = -D·O', ha='center', fontsize=11,
            transform=ax.transAxes)
    ax.text(0.175, 0.72, 'Oracle + Diffusion', ha='center', fontsize=9,
            style='italic', transform=ax.transAxes)

    # Equivalence arrow
    arrow1 = FancyArrowPatch((0.32, 0.775), (0.43, 0.775),
                            arrowstyle='<->', mutation_scale=25,
                            linewidth=3, color='red',
                            transform=ax.transAxes)
    ax.add_patch(arrow1)
    ax.text(0.375, 0.82, 'Equivalent', ha='center', fontsize=11,
            fontweight='bold', color='red', transform=ax.transAxes)

    # Hamiltonian formulation
    box2 = FancyBboxPatch((0.45, 0.7), 0.25, 0.15,
                          boxstyle="round,pad=0.02",
                          facecolor='lightgreen', edgecolor='black', linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(box2)
    ax.text(0.575, 0.82, 'Hamiltonian Form', ha='center', fontsize=12,
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.575, 0.765, r'$G = e^{-iHt}$', ha='center', fontsize=11,
            transform=ax.transAxes)
    ax.text(0.575, 0.72, r'$t = \pi/4$ per iteration', ha='center', fontsize=9,
            style='italic', transform=ax.transAxes)

    # Implementation methods
    ax.text(0.5, 0.58, 'Implementation via Hamiltonian Simulation:',
            ha='center', fontsize=13, fontweight='bold', transform=ax.transAxes)

    # Three methods
    methods = [
        (0.2, 'Taylor-LCU', 'lightcoral',
         'Truncated Taylor series\nwith block encoding'),
        (0.5, 'QSVT', 'gold',
         'Polynomial transformation\nvia signal processing'),
        (0.8, 'Standard', 'lightblue',
         'Direct oracle\nimplementation'),
    ]

    for x, name, color, desc in methods:
        box = FancyBboxPatch((x - 0.12, 0.35), 0.24, 0.15,
                            boxstyle="round,pad=0.02",
                            facecolor=color, edgecolor='black', linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x, 0.455, name, ha='center', fontsize=11,
                fontweight='bold', transform=ax.transAxes)
        ax.text(x, 0.4, desc, ha='center', fontsize=8,
                transform=ax.transAxes)

    # Comparison metrics
    ax.text(0.5, 0.25, 'Comparison Metrics:', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    metrics = [
        'Circuit Depth: Standard < Taylor-LCU ≈ QSVT',
        'Gate Count: Standard < Taylor-LCU < QSVT',
        'Ancilla Qubits: Standard (0) < Taylor-LCU < QSVT',
        'Generality: Standard < Taylor-LCU ≈ QSVT',
    ]

    y_pos = 0.18
    for i, metric in enumerate(metrics):
        ax.text(0.5, y_pos - i*0.04, f'• {metric}',
                ha='center', fontsize=9, transform=ax.transAxes)

    # Key insight
    ax.text(0.5, 0.02,
            'Key Insight: Hamiltonian simulation methods trade efficiency for generality',
            ha='center', fontsize=10, style='italic', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('grover_hamiltonian_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Created grover_hamiltonian_diagram.png")
    plt.close()


def create_benchmark_results_visualization():
    """Create mock benchmark results visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hamiltonian Simulation: Comprehensive Benchmark Results',
                 fontsize=18, fontweight='bold', y=0.995)

    algorithms = ['Trotter\n(1st)', 'Trotter\n(2nd)', 'Taylor-LCU',
                  'QSP', 'Qubitization', 'QSVT']
    colors = ['#3498db', '#2980b9', '#e74c3c', '#9b59b6', '#1abc9c', '#f39c12']

    # Mock data
    depth = [85, 165, 142, 128, 156, 138]
    gates = [324, 612, 485, 412, 528, 468]
    cnots = [128, 248, 196, 164, 212, 188]
    qubits = [3, 3, 6, 4, 6, 7]
    queries = [10, 10, 10, 10, 15, 21]
    errors = [2.5e-2, 6.3e-4, 8.2e-5, 8.2e-5, 4.1e-3, 8.2e-5]

    # Circuit Depth
    axes[0, 0].bar(algorithms, depth, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_title('Circuit Depth', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Depth', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Total Gates
    axes[0, 1].bar(algorithms, gates, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('Total Gate Count', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Gates', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # CNOT Count
    axes[0, 2].bar(algorithms, cnots, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 2].set_title('CNOT Count', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('CNOTs', fontsize=12)
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(axis='y', alpha=0.3)

    # Qubit Count
    axes[1, 0].bar(algorithms, qubits, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_title('Total Qubits', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Qubits', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Query Complexity
    axes[1, 1].bar(algorithms, queries, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_title('Query Complexity', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Queries', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Error (log scale)
    axes[1, 2].bar(algorithms, errors, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 2].set_title('Estimated Error', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Error', fontsize=12)
    axes[1, 2].set_yscale('log')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("✓ Created benchmark_results.png")
    plt.close()


def main():
    """Generate all presentation diagrams."""
    print("\nGenerating presentation diagrams...\n")

    create_algorithm_timeline()
    create_complexity_comparison()
    create_algorithm_comparison_radar()
    create_trotter_diagram()
    create_lcu_diagram()
    create_qsvt_diagram()
    create_grover_hamiltonian_diagram()
    create_benchmark_results_visualization()

    print("\n✓ All diagrams generated successfully!")
    print(f"  Saved in: {os.getcwd()}\n")


if __name__ == "__main__":
    main()
