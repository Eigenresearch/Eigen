import sys
from src.cli import register_command


@register_command("playground")
def playground_command(args, workspace_root):
    print("Not yet implemented")
    return


@register_command("watch")
def watch_command(args, workspace_root):
    print("Not yet implemented")
    return


@register_command("migrate")
def migrate_command(args, workspace_root):
    print("Not yet implemented")
    return


@register_command("completions")
def completions_command(args, workspace_root):
    print("Not yet implemented")
    return


@register_command("repl")
def repl_command(args, workspace_root):
    try:
        from src.cli_enhancements import REPL, REPLState
    except ImportError:
        print("Not yet implemented")
        return

    repl = REPL(evaluator=lambda source: source)
    print("Eigen REPL — type :exit to quit, :help for help")
    try:
        while not repl._exit_requested:
            try:
                line = input(repl.prompt)
            except (EOFError, KeyboardInterrupt):
                break
            result = repl.step(line)
            if result.state == REPLState.ERROR:
                print(f"Error: {result.error}", file=sys.stderr)
            elif result.output:
                print(result.output)
    except Exception as e:
        print(f"REPL error: {e}", file=sys.stderr)
