from src.cli import register_command
from src.formatter import EigenFormatter

@register_command("fmt")
def fmt_command(args, workspace_root):
    with open(args.file, 'r', encoding='utf-8') as f:
        content = f.read()
    formatter = EigenFormatter()
    formatted = formatter.format_code(content)
    with open(args.file, 'w', encoding='utf-8') as f:
        f.write(formatted)
    print(f"Formatted '{args.file}' successfully.")
