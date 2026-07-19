from src.cli import register_command
from src.packager import EigenPackager

@register_command("init")
def init_command(args, workspace_root):
    packager = EigenPackager(workspace_root)
    packager.init_package(args.name)

@register_command("add")
def add_command(args, workspace_root):
    packager = EigenPackager(workspace_root)
    packager.add_dependency(args.dependency, args.ver)

@register_command("install")
def install_command(args, workspace_root):
    packager = EigenPackager(workspace_root)
    packager.install_dependencies()

@register_command("search")
def search_command(args, workspace_root):
    packager = EigenPackager(workspace_root)
    packager.search_packages(args.query)

@register_command("publish")
def publish_command(args, workspace_root):
    packager = EigenPackager(workspace_root)
    packager.publish_package()
