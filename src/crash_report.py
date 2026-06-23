import datetime
import traceback

def write_crash_report(error, call_stack, ip, opcode, locals_map):
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    log_name = f"crash-{now}.log"
    
    with open(log_name, 'w', encoding='utf-8') as f:
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
        traceback.print_exc(file=f)
        
    print(f"\n[CRASH DETECTED] Eigen VM crashed! Crash log generated at: {log_name}")
