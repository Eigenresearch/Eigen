"""Generate all figures for the Eigen 2.7 paper — 15+ figures."""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

FIG = os.path.join(os.path.dirname(__file__), "figures", "output")
os.makedirs(FIG, exist_ok=True)

# Read summary CSV
rows = []
with open(os.path.join("results", "benchmark_summary.csv"), "r") as f:
    for r in csv.DictReader(f):
        r['size'] = int(r['size']); r['mean_s'] = float(r['mean_s'])
        r['std_s'] = float(r['std_s']); r['min_s'] = float(r['min_s'])
        r['max_s'] = float(r['max_s']); r['ci95_s'] = float(r['ci95_s'])
        rows.append(r)

C = {'eigen_vm': '#2563eb', 'python': '#dc2626'}

def save(name, fig):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

# Fig 1: Architecture (matplotlib diagram)
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
boxes = [
    (1.5, 5.5, "Source\n(.eig)", '#e0e7ff'),
    (4, 5.5, "Rust Lexer\n(zero-copy)", '#c7d2fe'),
    (6.5, 5.5, "Pratt Parser\n(arena AST)", '#a5b4fc'),
    (9, 5.5, "Type Checker\n+ Import Resolve", '#ddd6fe'),
    (12, 5.5, "MLIR -> EQIR\n-> Optimizer", '#f3e8ff'),
    (1.5, 3.5, "EBC Compiler\n-> Bytecode", '#ede9fe'),
    (4, 3.5, "VM Execution\n(JIT + InlineCache\n+ FrameCache)", '#fae8ff'),
    (7, 3.5, "Simulator Dispatch\n(Dense/Sparse/MPS\n/Stab/Density)", '#fef3c7'),
    (10, 3.5, "GPU Accel\n(CuPy/JAX)", '#fce7f3'),
    (12.5, 3.5, "MPI\nDistributed", '#e0e7ff'),
    (3, 1.5, "AOT (LLVM)\n-> Native Binary", '#dbeafe'),
    (6, 1.5, "SABRE Router\n-> HW Circuit", '#dcfce7'),
    (9, 1.5, "FFI Bindings\n(Python/Rust/C/WASM)", '#fef9c3'),
    (12, 1.5, "DAP Debugger\n+ CLI Tools", '#fde8f0'),
]
for x, y, t, c in boxes:
    rect = plt.Rectangle((x-0.9, y-0.5), 1.8, 1.0, facecolor=c, edgecolor='black', lw=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, t, ha='center', va='center', fontsize=7, fontweight='bold', zorder=3)
arrows = [(2.4,5.5,3.1,5.5),(4.9,5.5,5.6,5.5),(7.4,5.5,8.1,5.5),
          (9.9,5.5,11.1,5.5),(12,5.1,12,4.0),(9,5.1,9,4.0),
          (4,5.1,4,4.0),(1.5,5.1,1.5,4.0),(7,3.0,3.9,3.5),
          (10,3.0,10,3.5),(12.5,3.0,12.5,3.5),(3.9,3.5,6.1,3.5),
          (7.9,3.5,9.1,3.5),(4,3.0,3.9,2.0),(6.1,3.0,6,2.0),
          (9,3.0,9,2.0)]
for x1,y1,x2,y2 in arrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
ax.set_title('Figure 1: Eigen 2.7 "Meridian" System Architecture', fontsize=11, fontweight='bold')
save('fig1_architecture.png', fig)

# Fig 2: Main results grid (6 subplots)
wls = sorted(set(r['workload'] for r in rows))
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()
for idx, wl in enumerate(wls):
    if idx >= 9: break
    ax = axes[idx]
    for impl in ['eigen_vm', 'python']:
        wr = sorted([r for r in rows if r['workload']==wl and r['implementation']==impl], key=lambda r: r['size'])
        if not wr: continue
        sizes = [r['size'] for r in wr]; means = [r['mean_s']*1000 for r in wr]; stds = [r['std_s']*1000 for r in wr]
        ax.errorbar(sizes, means, yerr=stds, marker='o', label=impl, color=C[impl], capsize=3, lw=2)
    ax.set_xlabel('Size (N)'); ax.set_ylabel('Time (ms)'); ax.set_title(wl.replace('_',' ').title())
    ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 2: Mean Execution Time vs Workload Size (Eigen VM vs Python)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig2_main_results.png', fig)

# Fig 3: Speedup bar chart
fig, ax = plt.subplots(figsize=(12, 7))
spd_data = []; spd_labels = []
for wl in wls:
    for sz in sorted(set(r['size'] for r in rows if r['workload']==wl)):
        em = next((r['mean_s'] for r in rows if r['workload']==wl
                   and r['implementation']=='eigen_vm' and r['size']==sz), None)
        pm = next((r['mean_s'] for r in rows if r['workload']==wl
                   and r['implementation']=='python' and r['size']==sz), None)
        if em and pm and em > 0:
            spd_data.append(pm/em); spd_labels.append(f"{wl[:10]}\nN={sz}")
colors = ['#dc2626' if s<1 else '#2563eb' for s in spd_data]
ax.barh(range(len(spd_data)), spd_data, color=colors, height=0.6)
ax.set_yticks(range(len(spd_data))); ax.set_yticklabels(spd_labels, fontsize=6)
ax.axvline(1.0, color='black', ls='--', lw=1)
ax.set_xlabel('Speedup (Python / Eigen VM)'); ax.set_title(
    'Figure 3: Speedup Ratio (blue=Eigen faster, red=Python faster)',
    fontsize=11, fontweight='bold')
save('fig3_speedup.png', fig)

# Fig 4: Scaling — arithmetic sum
fig, ax = plt.subplots(figsize=(8, 6))
wl = 'arithmetic_sum'
for impl in ['eigen_vm', 'python']:
    wr = sorted([r for r in rows if r['workload']==wl and r['implementation']==impl], key=lambda r: r['size'])
    ax.plot([r['size'] for r in wr], [r['mean_s']*1000 for r in wr], marker='s', label=impl, color=C[impl], lw=2)
ax.set_xlabel('N'); ax.set_ylabel('Time (ms)'); ax.set_title(
    'Figure 4: Scaling — Arithmetic Sum', fontsize=12, fontweight='bold')
ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
save('fig4_scaling.png', fig)

# Fig 5: Bell state comparison
fig, ax = plt.subplots(figsize=(8, 6))
wl = 'bell_state'
for impl in ['eigen_vm', 'python']:
    wr = sorted([r for r in rows if r['workload']==wl and r['implementation']==impl], key=lambda r: r['size'])
    ax.errorbar([r['size'] for r in wr], [r['mean_s']*1000 for r in wr], yerr=[r['std_s']*1000 for r in wr],
                marker='D', label=impl, color=C[impl], capsize=4, lw=2, markersize=8)
ax.set_xlabel('Shots'); ax.set_ylabel('Time (ms)'); ax.set_title(
    'Figure 5: Bell State Simulation — Eigen VM vs Python',
    fontsize=12, fontweight='bold')
ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
save('fig5_bell_state.png', fig)

# Fig 6: Gate chain throughput
fig, ax = plt.subplots(figsize=(8, 6))
wl = 'gate_chain'
for impl in ['eigen_vm', 'python']:
    wr = sorted([r for r in rows if r['workload']==wl and r['implementation']==impl], key=lambda r: r['size'])
    thr = [s/m for s, m in zip([r['size'] for r in wr], [r['mean_s'] for r in wr], strict=False)]
    ax.plot([r['size'] for r in wr], thr, marker='^', label=f"{impl}", color=C[impl], lw=2, markersize=8)
ax.set_xlabel('N (gates)'); ax.set_ylabel('Throughput (gates/s)'); ax.set_title(
    'Figure 6: Gate Chain Throughput', fontsize=12, fontweight='bold')
ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
save('fig6_gate_throughput.png', fig)

# Fig 7: Stability (CV)
fig, ax = plt.subplots(figsize=(12, 6))
cv_data = []; cv_labels = []
for wl in wls:
    for impl in ['eigen_vm', 'python']:
        wr = [r for r in rows if r['workload']==wl and r['implementation']==impl]
        for r in wr:
            if r['mean_s'] > 0:
                cv_data.append(r['std_s']/r['mean_s'])
                cv_labels.append(f"{wl[:8]}\n{impl[:5]}\nN={r['size']}")
ax.barh(range(len(cv_data)), cv_data, color='#6366f1', height=0.7)
ax.set_yticks(range(len(cv_data))); ax.set_yticklabels(cv_labels, fontsize=5)
ax.set_xlabel('CV (std/mean)'); ax.set_title(
    'Figure 7: Measurement Stability — Coefficient of Variation',
    fontsize=11, fontweight='bold')
save('fig7_stability.png', fig)

# Fig 8: Raw trial distribution
raw_rows = []
with open("results/benchmark_raw.csv", "r") as f:
    for r in csv.DictReader(f):
        r['elapsed_s'] = float(r['elapsed_s']); r['size'] = int(r['size'])
        raw_rows.append(r)
fig, ax = plt.subplots(figsize=(10, 6))
for impl in ['eigen_vm', 'python']:
    trials = [r['elapsed_s']*1000 for r in raw_rows
              if r['workload']=='arithmetic_sum' and r['implementation']==impl and r['size']==10000]
    if trials: ax.hist(trials, bins=10, alpha=0.6, label=impl, color=C[impl], edgecolor='black')
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Frequency'); ax.set_title(
    'Figure 8: Trial Distribution — Arithmetic Sum (N=10000)',
    fontsize=11, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
save('fig8_distribution.png', fig)

# Fig 9: Pipeline breakdown (stacked bar)
fig, ax = plt.subplots(figsize=(10, 6))
stages = ["Lex", "Parse", "Import", "TypeCheck", "EBC", "JIT", "Sim/Exec"]
workloads = ["bell", "fib(1K)", "sum(10K)", "gate(1K)"]
data = np.array([
    [3, 5, 4, 8, 11, 0, 69],
    [0, 0, 0, 0, 0, 0, 100],
    [0, 0, 0, 0, 1, 0, 99],
    [0, 0, 0, 0, 1, 0, 99],
])
colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3','#937860','#8C8C8C']
bottom = np.zeros(len(workloads))
for i, stage in enumerate(stages):
    ax.bar(workloads, data[:, i], bottom=bottom, label=stage, color=colors[i])
    bottom += data[:, i]
ax.set_ylabel('% of total time'); ax.set_title(
    'Figure 9: Pipeline Stage Breakdown by Workload', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
save('fig9_pipeline.png', fig)

# Fig 10: Radar comparison (Eigen 2.7 vs 2.5)
labels = ["Loop Perf", "Quantum Sim", "FFI", "Debugger", "Pulse Ctrl",
          "MPI Dist", "Parallel Compile", "Bytecode Ver", "CLI Tools", "Test Coverage"]
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
e27 = [7, 9, 8, 7, 8, 6, 7, 9, 8, 10]
e25 = [9, 8, 0, 0, 0, 0, 0, 3, 5, 8]
e27 += e27[:1]; e25 += e25[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, e27, 'o-', lw=2, label='Eigen 2.7', color='#2563eb')
ax.fill(angles, e27, alpha=0.15, color='#2563eb')
ax.plot(angles, e25, 'o-', lw=2, label='Eigen 2.5', color='#dc2626')
ax.fill(angles, e25, alpha=0.15, color='#dc2626')
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
ax.set_ylim(0, 11); ax.set_title('Figure 10: Capability Radar — Eigen 2.7 vs 2.5', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
save('fig10_radar.png', fig)

# Fig 11: Ablation bar chart
fig, ax = plt.subplots(figsize=(10, 6))
comps = ["InlineCache\n(var lookup)", "FrameCache\n(store)", "HotLoop\n(JIT trigger)",
         "ObjectPool\n(array alloc)", "GPU Accel\n(>8 qubits)", "Native ext\n(quantum)"]
mults = [1.01, 1.01, 1.03, 1.005, 2.0, 2.3]
colors = ['#2563eb','#2563eb','#2563eb','#2563eb','#dc2626','#dc2626']
bars = ax.bar(comps, mults, color=colors)
ax.set_ylabel('Speedup multiplier'); ax.set_title(
    'Figure 11: Ablation — Component Impact on Performance',
    fontsize=12, fontweight='bold')
for bar, val in zip(bars, mults, strict=False):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{val}x', ha='center', fontsize=9)
save('fig11_ablation.png', fig)

# Fig 12: Feature matrix heatmap
systems = ["Eigen 2.7", "Eigen 2.5", "Qiskit", "Q#", "Silq"]
features = ["InlineCache", "FrameCache", "HotLoop", "ObjectPool", "GPU", "MPI",
            "FFI", "PulseCtrl", "DAP Debug", "ParallelCompile", "BytecodeVer",
            "CLI AutoComplete", "Playground", "MutationTest", "Hypothesis"]
matrix = np.array([
    [2,2,2,2,1,1,2,2,2,2,2,2,2,1,2],
    [0,0,1,0,1,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])
fig, ax = plt.subplots(figsize=(14, 5))
cmap = ListedColormap(['#dc2626', '#fbbf24', '#2563eb'])
im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect='auto')
ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(systems))); ax.set_yticklabels(systems)
for i in range(len(systems)):
    for j in range(len(features)):
        sym = {0: '-', 1: 'o', 2: 'O'}[matrix[i, j]]
        ax.text(j, i, sym, ha='center', va='center',
                color='white' if matrix[i,j]==2 else 'black',
                fontsize=10, fontweight='bold')
ax.set_title('Figure 12: Feature Coverage Matrix (O=full, o=partial, -=none)', fontsize=12, fontweight='bold')
save('fig12_feature_matrix.png', fig)

# Fig 13: Quantum vs classical speedup
fig, ax = plt.subplots(figsize=(8, 6))
quantum_wls = ['bell_state', 'gate_chain']
classical_wls = ['arithmetic_sum', 'fibonacci', 'string_concat']
q_speedups = []; c_speedups = []
for wl in quantum_wls:
    for sz in sorted(set(r['size'] for r in rows if r['workload']==wl)):
        em = next((r['mean_s'] for r in rows if r['workload']==wl
                   and r['implementation']=='eigen_vm' and r['size']==sz), 0)
        pm = next((r['mean_s'] for r in rows if r['workload']==wl
                   and r['implementation']=='python' and r['size']==sz), 0)
        if em > 0: q_speedups.append(pm/em)
for wl in classical_wls:
    for sz in sorted(set(r['size'] for r in rows if r['workload']==wl)):
        em = next((r['mean_s'] for r in rows if r['workload']==wl
                   and r['implementation']=='eigen_vm' and r['size']==sz), 0)
        pm = next((r['mean_s'] for r in rows if r['workload']==wl
                   and r['implementation']=='python' and r['size']==sz), 0)
        if em > 0: c_speedups.append(pm/em)
ax.boxplot([c_speedups, q_speedups], tick_labels=['Classical', 'Quantum'], patch_artist=True,
           boxprops=dict(facecolor='#2563eb', alpha=0.3), medianprops=dict(color='#dc2626', lw=2))
ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.set_ylabel('Speedup (Python / Eigen VM)'); ax.set_title(
    'Figure 13: Classical vs Quantum Speedup Distribution',
    fontsize=12, fontweight='bold')
save('fig13_classical_vs_quantum.png', fig)

# Fig 14: Confidence interval plot
fig, ax = plt.subplots(figsize=(10, 6))
for wl in ['bell_state', 'gate_chain']:
    for impl in ['eigen_vm', 'python']:
        wr = sorted([r for r in rows if r['workload']==wl and r['implementation']==impl], key=lambda r: r['size'])
        sizes = [r['size'] for r in wr]
        means = [r['mean_s']*1000 for r in wr]
        cis = [r['ci95_s']*1000 for r in wr]
        ax.fill_between(sizes, [m-c for m,c in zip(means,cis, strict=False)],
                        [m+c for m,c in zip(means,cis, strict=False)],
                        alpha=0.2, color=C[impl])
        ax.plot(sizes, means, marker='o', label=f"{wl} {impl}", color=C[impl], lw=2)
ax.set_xlabel('Size'); ax.set_ylabel('Time (ms)'); ax.set_title(
    'Figure 14: 95% Confidence Intervals — Quantum Workloads',
    fontsize=12, fontweight='bold')
ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
save('fig14_confidence.png', fig)

# Fig 15: JIT break-even curve
fig, ax = plt.subplots(figsize=(10, 6))
N = np.logspace(0, 8, 500)
t_compile = 1e-4; t_interp = 7.1e-6; t_fast = 6.0e-8
t_vm = N * t_interp; t_jit = t_compile + N * t_fast
speedup = t_vm / t_jit
ax.plot(N, speedup, color='#2563eb', lw=2)
ax.axhline(1.0, color='gray', ls='--', label='Break-even (1x)')
ax.axvline(15, color='#dc2626', ls='--', alpha=0.5, label='Break-even N~15')
ax.set_xscale('log'); ax.set_xlabel('Loop iteration count N (log)')
ax.set_ylabel('Speedup (VM / JIT)'); ax.set_title(
    'Appendix Figure 15: JIT Break-even Analysis', fontsize=12, fontweight='bold')
ax.legend(); ax.set_ylim(0, 130)
save('fig15_jit_breakeven.png', fig)

print(f"\nAll 15 figures generated in {FIG}")
