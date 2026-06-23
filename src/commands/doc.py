from src.cli import register_command
from src.doc_generator import EigenDocGenerator

@register_command("doc")
def doc_command(args, workspace_root):
    with open(args.file, 'r', encoding='utf-8') as f:
        content = f.read()
    doc_gen = EigenDocGenerator()
    docs = doc_gen.generate_docs(content, args.file)
    out_path = args.file.rsplit('.', 1)[0] + "_reference.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(docs)
    print(f"Generated API documentation at '{out_path}'")
