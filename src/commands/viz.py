import os
import sys
from src.cli import register_command
from src.compiler import compile_to_eqir
from src.visualization.circuit_svg import CircuitSVGGenerator

@register_command("viz")
def viz_command(args, workspace_root):
    if not args.file:
        print("Error: No input file specified.", file=sys.stderr)
        sys.exit(1)

    try:
        graph, ast = compile_to_eqir(args.file, workspace_root)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        generator = CircuitSVGGenerator(graph)
        svg_content = generator.generate()
    except Exception as e:
        print(f"Error generating SVG: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if not output_path:
        output_path = os.path.splitext(args.file)[0] + ".svg"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"Successfully generated circuit visualization: {output_path}")
    except Exception as e:
        print(f"Error writing to output file {output_path}: {e}", file=sys.stderr)
        sys.exit(1)
