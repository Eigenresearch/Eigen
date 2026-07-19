
class CircuitSVGGenerator:
    """Generates modern, high-quality SVG visualizations of EQIR quantum circuits."""

    def __init__(self, graph):
        self.graph = graph

    def get_layout(self):
        sorted_nodes = self.graph.topological_sort()
        qubits = []
        gates = []
        for node in sorted_nodes:
            if node.type == 'ALLOC':
                qname = node.targets[0]
                if qname not in qubits:
                    qubits.append(qname)
            elif node.type == 'GATE':
                gates.append({
                    'gate': node.gate_name,
                    'targets': list(node.targets),
                    'args': list(node.args)
                })

        qubit_next_col = {q: 0 for q in qubits}
        columns = []

        for gate in gates:
            targets = gate['targets']
            if not targets:
                continue

            col_idx = max(qubit_next_col[t] for t in targets)

            target_indices = [qubits.index(t) for t in targets]
            min_idx = min(target_indices)
            max_idx = max(target_indices)
            span_qubits = qubits[min_idx : max_idx + 1]

            col_idx = max(col_idx, max(qubit_next_col[q] for q in span_qubits))

            while len(columns) <= col_idx:
                columns.append([])
            columns[col_idx].append(gate)

            for q in span_qubits:
                qubit_next_col[q] = col_idx + 1

        return qubits, columns

    def generate(self) -> str:
        qubits, columns = self.get_layout()

        qubit_height = 50
        col_width = 60
        padding_x = 40
        padding_y = 30

        num_qubits = len(qubits)
        num_cols = len(columns)

        width = padding_x * 2 + max(1, num_cols) * col_width
        height = padding_y * 2 + max(0, num_qubits - 1) * qubit_height

        svg = []
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                   f'height="{height}" viewBox="0 0 {width} {height}">')

        svg.append("""<style>
            .bg { fill: #0f111a; }
            .qubit-line { stroke: #2a2e42; stroke-width: 2; stroke-dasharray: 2, 2; }
            .qubit-label { fill: #8f93a2; font-family: 'Outfit', 'Inter', sans-serif;
            font-size: 14px; font-weight: bold; }
            .gate-rect { fill: url(#gate-grad); stroke: #82aaff; stroke-width: 1.5; rx: 6px; ry: 6px; }
            .gate-text { fill: #ffffff; font-family: 'Outfit', 'Inter', sans-serif;
            font-size: 13px; font-weight: bold; text-anchor: middle;
            dominant-baseline: middle; }
            .control-dot { fill: #82aaff; stroke: #0f111a; stroke-width: 1.5; }
            .cnot-target-outer { fill: #ff5370; stroke: #ff5370; stroke-width: 1.5; }
            .cnot-target-inner { stroke: #ffffff; stroke-width: 2; }
            .swap-x { stroke: #f78c6c; stroke-width: 2; stroke-linecap: round; }
            .connection-line { stroke: #82aaff; stroke-width: 2; }
        </style>""")

        svg.append("""<defs>
            <linearGradient id="gate-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#4a3b8c" />
                <stop offset="100%" stop-color="#2a1b6c" />
            </linearGradient>
        </defs>""")

        svg.append(f'<rect width="{width}" height="{height}" class="bg" />')

        for i, q in enumerate(qubits):
            y = padding_y + i * qubit_height
            svg.append(f'<text x="{padding_x - 15}" y="{y}" class="qubit-label" '
                       f'text-anchor="end" dominant-baseline="middle">{q}</text>')
            svg.append(f'<line x1="{padding_x}" y1="{y}" x2="{width - padding_x}" y2="{y}" class="qubit-line" />')

        for col_idx, col_gates in enumerate(columns):
            cx = padding_x + col_idx * col_width + col_width / 2
            for gate in col_gates:
                gate_name = gate['gate']
                targets = gate['targets']
                target_indices = [qubits.index(t) for t in targets]

                if len(targets) == 1:
                    y = padding_y + target_indices[0] * qubit_height
                    gw, gh = 36, 36
                    svg.append(f'<rect x="{cx - gw/2}" y="{y - gh/2}" width="{gw}" height="{gh}" class="gate-rect" />')
                    svg.append(f'<text x="{cx}" y="{y}" class="gate-text">{gate_name}</text>')

                elif len(targets) == 2 and gate_name in ('CNOT', 'CX'):
                    ctrl_y = padding_y + target_indices[0] * qubit_height
                    tgt_y = padding_y + target_indices[1] * qubit_height

                    svg.append(f'<line x1="{cx}" y1="{ctrl_y}" x2="{cx}" y2="{tgt_y}" class="connection-line" />')
                    svg.append(f'<circle cx="{cx}" cy="{ctrl_y}" r="5" class="control-dot" />')
                    
                    r = 8
                    svg.append(f'<circle cx="{cx}" cy="{tgt_y}" r="{r}" class="cnot-target-outer" />')
                    svg.append(f'<line x1="{cx - r + 3}" y1="{tgt_y}" x2="{cx + r - 3}" '
                               f'y2="{tgt_y}" class="cnot-target-inner" />')
                    svg.append(f'<line x1="{cx}" y1="{tgt_y - r + 3}" x2="{cx}" '
                               f'y2="{tgt_y + r - 3}" class="cnot-target-inner" />')

                elif len(targets) == 2 and gate_name == 'SWAP':
                    y1 = padding_y + target_indices[0] * qubit_height
                    y2 = padding_y + target_indices[1] * qubit_height

                    svg.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" class="connection-line" />')
                    for y in (y1, y2):
                        d = 5
                        svg.append(f'<line x1="{cx - d}" y1="{y - d}" x2="{cx + d}" y2="{y + d}" class="swap-x" />')
                        svg.append(f'<line x1="{cx - d}" y1="{y + d}" x2="{cx + d}" y2="{y - d}" class="swap-x" />')

                elif len(targets) == 2 and gate_name == 'CZ':
                    y1 = padding_y + target_indices[0] * qubit_height
                    y2 = padding_y + target_indices[1] * qubit_height
                    svg.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" class="connection-line" />')
                    svg.append(f'<circle cx="{cx}" cy="{y1}" r="5" class="control-dot" />')
                    svg.append(f'<circle cx="{cx}" cy="{y2}" r="5" class="control-dot" />')

                else:
                    ctrl_y = [padding_y + idx * qubit_height for idx in target_indices[:-1]]
                    tgt_y = padding_y + target_indices[-1] * qubit_height

                    all_y = ctrl_y + [tgt_y]
                    min_y, max_y = min(all_y), max(all_y)

                    svg.append(f'<line x1="{cx}" y1="{min_y}" x2="{cx}" y2="{max_y}" class="connection-line" />')

                    for cy in ctrl_y:
                        svg.append(f'<circle cx="{cx}" cy="{cy}" r="5" class="control-dot" />')

                    y = tgt_y
                    gw, gh = 36, 36
                    svg.append(f'<rect x="{cx - gw/2}" y="{y - gh/2}" width="{gw}" height="{gh}" class="gate-rect" />')
                    svg.append(f'<text x="{cx}" y="{y}" class="gate-text">{gate_name}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)
