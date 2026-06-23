from src.cli import register_command
from src.lsp.lsp_server import LSPServer

@register_command("lsp")
def lsp_command(args, workspace_root):
    server = LSPServer()
    server.run()
