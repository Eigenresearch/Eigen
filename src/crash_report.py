import datetime
import os
import tempfile
import traceback

def write_crash_report(error, call_stack, ip, opcode, locals_map):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"crash-{now}.log"

    def _write_report(f):
        f.write("=" * 60 + "\n")
        f.write(f"           EIGEN VM CRASH REPORT - {now}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Error Message: {error}\n\n")
        f.write(f"Instruction Pointer (IP): {ip}\n")
        f.write(f"Opcode at crash: {opcode}\n\n")

        f.write("Call Stack:\n")
        for idx, frame in enumerate(reversed(call_stack)):
            line_str = f"line {frame.current_line}" if frame.current_line is not None else "unknown line"
            f.write(f"  Frame {idx}: function '{frame.func_name}' at {line_str}\n")
            f.write(f"    Locals: {frame.locals}\n")

        f.write("\nGlobals:\n")
        f.write(f"  {locals_map}\n\n")

        f.write("Traceback:\n")
        tb_lines = traceback.format_exception(type(error), error, getattr(error, '__traceback__', None))
        f.write("".join(tb_lines))

    try:
        path = os.path.join(os.getcwd(), log_name)
        with open(path, 'w', encoding='utf-8') as f:
            _write_report(f)
    except (PermissionError, OSError):
        path = os.path.join(tempfile.gettempdir(), log_name)
        with open(path, 'w', encoding='utf-8') as f:
            _write_report(f)

    print(f"\n[CRASH DETECTED] Eigen VM crashed! Crash log generated at: {path}")
