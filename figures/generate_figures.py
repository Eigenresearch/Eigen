"""Generate all figures from results/benchmark_summary.csv."""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read summary CSV
summary_path = os.path.join("results", "benchmark_summary.csv")
rows = []
with open(summary_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['size'] = int(row['size'])
        row['mean_s'] = float(row['mean_s'])
        row['std_s'] = float(row['std_s'])
        row['min_s'] = float(row['min_s'])
        row['max_s'] = float(row['max_s'])
        row['ci95_s'] = float(row['ci95_s'])
        rows.append(row)

os.makedirs("figures/output", exist_ok=True)

# Color scheme
COLORS = {'eigen_vm': '#2563eb', 'python': '#dc2626'}

# ---------------------------------------------------------------------------
# Figure 1: Architecture diagram (matplotlib-rendered)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# Boxes
boxes = [
    (1, 4.5, "Source (.eig)", '#e0e7ff'),
    (4, 4.5, "Lexer → Parser\n→ AST", '#c7d2fe'),
    (7, 4.5, "Type Checker\n→ MLIR → EQIR", '#a5b4fc'),
    (1, 2.5, "Optimizer\n(7 passes)", '#ddd6fe'),
    (4, 2.5, "EBC Compiler\n→ Bytecode", '#ede9fe'),
    (7, 2.5, "VM Execution\n(JIT + inline cache)", '#f3e8ff'),
    (4, 0.5, "Quantum Simulator\n(Dense/Sparse/MPS/Stab)", '#fae8ff'),
]
for x, y, text, color in boxes:
    rect = plt.Rectangle((x-0.9, y-0.4), 1.8, 0.8,
                            facecolor=color, edgecolor='black',
                            linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
             fontsize=8, fontweight='bold', zorder=3)

# Arrows
arrows = [(1.9, 4.5, 3.1, 4.5), (4.9, 4.5, 6.1, 4.5),
           (7, 4.1, 7, 2.9), (6.1, 2.5, 4.9, 2.5),
           (3.1, 2.5, 1.9, 2.5), (4, 2.1, 4, 0.9),
           (4.9, 2.5, 6.1, 2.5)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

ax.set_title('Figure 1: Eigen 2.7 System Architecture',
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/output/fig1_architecture.png', dpi=150)
plt.close()
print("Figure 1 saved")

# ---------------------------------------------------------------------------
# Figure 2: Main results — mean time vs workload size (log-log)
# ---------------------------------------------------------------------------
workloads = sorted(set(r['workload'] for r in rows))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, wl in enumerate(workloads):
    ax = axes[idx]
    for impl in ['eigen_vm', 'python']:
        wl_rows = sorted(
            [r for r in rows if r['workload'] == wl
             and r['implementation'] == impl],
            key=lambda r: r['size'])
        if not wl_rows:
            continue
        sizes = [r['size'] for r in wl_rows]
        means = [r['mean_s'] * 1000 for r in wl_rows]  # ms
        stds = [r['std_s'] * 1000 for r in wl_rows]
        ax.errorbar(sizes, means, yerr=stds, marker='o',
                      label=impl, color=COLORS[impl], capsize=3, linewidth=2)
    ax.set_xlabel('Workload Size (N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(wl.replace('_', ' ').title())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 2: Mean Execution Time vs Workload Size (Eigen VM vs Python)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/output/fig2_main_results.png', dpi=150)
plt.close()
print("Figure 2 saved")

# ---------------------------------------------------------------------------
# Figure 3: Speedup ratio (Python/Eigen) per workload
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
speedup_data = []
speedup_labels = []
for wl in workloads:
    for size in sorted(set(r['size'] for r in rows if r['workload'] == wl)):
        eigen_mean = next((r['mean_s'] for r in rows
                            if r['workload'] == wl
                            and r['implementation'] == 'eigen_vm'
                            and r['size'] == size), None)
        py_mean = next((r['mean_s'] for r in rows
                         if r['workload'] == wl
                         and r['implementation'] == 'python'
                         and r['size'] == size), None)
        if eigen_mean and py_mean and eigen_mean > 0:
            speedup = py_mean / eigen_mean
            speedup_data.append(speedup)
            speedup_labels.append(f"{wl}\nN={size}")

colors = ['#dc2626' if s < 1 else '#2563eb' for s in speedup_data]
ax.barh(range(len(speedup_data)), speedup_data, color=colors, height=0.6)
ax.set_yticks(range(len(speedup_data)))
ax.set_yticklabels(speedup_labels, fontsize=7)
ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Speedup (Python time / Eigen VM time)')
ax.set_title('Figure 3: Speedup Ratio — Positive = Eigen Faster, Negative = Python Faster',
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/output/fig3_speedup.png', dpi=150)
plt.close()
print("Figure 3 saved")

# ---------------------------------------------------------------------------
# Figure 4: Scaling behavior — arithmetic sum
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
wl = 'arithmetic_sum'
for impl in ['eigen_vm', 'python']:
    wl_rows = sorted(
        [r for r in rows if r['workload'] == wl
         and r['implementation'] == impl],
        key=lambda r: r['size'])
    sizes = [r['size'] for r in wl_rows]
    means = [r['mean_s'] * 1000 for r in wl_rows]
    ax.plot(sizes, means, marker='s', label=impl,
             color=COLORS[impl], linewidth=2, markersize=6)

ax.set_xlabel('N (iterations)')
ax.set_ylabel('Time (ms)')
ax.set_title('Figure 4: Scaling Behavior — Arithmetic Sum Workload',
              fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/output/fig4_scaling.png', dpi=150)
plt.close()
print("Figure 4 saved")

# ---------------------------------------------------------------------------
# Figure 5: Quantum workload comparison (bell_state)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
wl = 'bell_state'
for impl in ['eigen_vm', 'python']:
    wl_rows = sorted(
        [r for r in rows if r['workload'] == wl
         and r['implementation'] == impl],
        key=lambda r: r['size'])
    sizes = [r['size'] for r in wl_rows]
    means = [r['mean_s'] * 1000 for r in wl_rows]
    stds = [r['std_s'] * 1000 for r in wl_rows]
    ax.errorbar(sizes, means, yerr=stds, marker='D',
                 label=impl, color=COLORS[impl],
                 capsize=4, linewidth=2, markersize=8)

ax.set_xlabel('Shots')
ax.set_ylabel('Time (ms)')
ax.set_title('Figure 5: Quantum Bell State — Eigen VM vs Python (numpy)',
              fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/output/fig5_bell_state.png', dpi=150)
plt.close()
print("Figure 5 saved")

# ---------------------------------------------------------------------------
# Figure 6: Gate chain throughput
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
wl = 'gate_chain'
for impl in ['eigen_vm', 'python']:
    wl_rows = sorted(
        [r for r in rows if r['workload'] == wl
         and r['implementation'] == impl],
        key=lambda r: r['size'])
    sizes = [r['size'] for r in wl_rows]
    means = [r['mean_s'] * 1000 for r in wl_rows]
    throughputs = [s / m for s, m in zip(sizes, means, strict=False)]
    ax.plot(sizes, throughputs, marker='^',
             label=f"{impl} (gates/ms)", color=COLORS[impl],
             linewidth=2, markersize=8)

ax.set_xlabel('N (gates)')
ax.set_ylabel('Throughput (gates/ms)')
ax.set_title('Figure 6: Gate Chain Throughput — Quantum Gate Application Rate',
              fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/output/fig6_gate_throughput.png', dpi=150)
plt.close()
print("Figure 6 saved")

# ---------------------------------------------------------------------------
# Figure 7: Variance / stability analysis
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = []
cv_labels = []
for wl in workloads:
    for impl in ['eigen_vm', 'python']:
        wl_rows = [r for r in rows if r['workload'] == wl
                    and r['implementation'] == impl]
        for r in wl_rows:
            if r['mean_s'] > 0:
                cv = r['std_s'] / r['mean_s']
                cv_data.append(cv)
                cv_labels.append(f"{wl[:8]}\n{impl[:5]}\nN={r['size']}")

ax.barh(range(len(cv_data)), cv_data, color='#6366f1', height=0.7)
ax.set_yticks(range(len(cv_data)))
ax.set_yticklabels(cv_labels, fontsize=6)
ax.set_xlabel('Coefficient of Variation (std/mean)')
ax.set_title('Figure 7: Measurement Stability — Coefficient of Variation Across All Runs',
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/output/fig7_stability.png', dpi=150)
plt.close()
print("Figure 7 saved")

# ---------------------------------------------------------------------------
# Figure 8: Appendix figure — raw trial distribution
# ---------------------------------------------------------------------------
raw_path = os.path.join("results", "benchmark_raw.csv")
raw_rows = []
with open(raw_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['elapsed_s'] = float(row['elapsed_s'])
        row['size'] = int(row['size'])
        raw_rows.append(row)

fig, ax = plt.subplots(figsize=(10, 6))
# Show distribution for arithmetic_sum N=10000
target_wl = 'arithmetic_sum'
target_size = 10000
for impl in ['eigen_vm', 'python']:
    trials = [r['elapsed_s'] * 1000 for r in raw_rows
               if r['workload'] == target_wl
               and r['implementation'] == impl
               and r['size'] == target_size]
    if trials:
        ax.hist(trials, bins=10, alpha=0.6, label=impl,
                 color=COLORS[impl], edgecolor='black')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Figure 8: Trial Distribution — Arithmetic Sum (N=10000, 10 Trials)',
              fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/output/fig8_distribution.png', dpi=150)
plt.close()
print("Figure 8 saved")

print("\nAll 8 figures generated in figures/output/")
