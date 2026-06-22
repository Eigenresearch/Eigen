import re

class EigenDocGenerator:
    def __init__(self):
        pass

    def generate_docs(self, source: str, filepath: str) -> str:
        lines = source.splitlines()
        filename = filepath.split('/')[-1].split('\\')[-1]
        
        md_lines = [
            f"# API Reference: {filename}",
            "",
            "Generated automatically from Eigen source code comments.",
            ""
        ]
        
        accumulated_comments = []
        
        for line in lines:
            line_strip = line.strip()
            
            # Match comments
            if line_strip.startswith('#') or line_strip.startswith('//'):
                comment_text = re.sub(r'^(#|//)\s*', '', line_strip)
                accumulated_comments.append(comment_text)
                continue
                
            # Match qfunc declarations
            qfunc_match = re.match(r'qfunc\s+(\w+)\s*\((.*?)\)', line_strip)
            if qfunc_match:
                q_name = qfunc_match.group(1)
                q_params = qfunc_match.group(2)
                
                md_lines.append(f"## Quantum Subroutine `{q_name}`")
                md_lines.append("")
                md_lines.append("```eigen")
                md_lines.append(f"qfunc {q_name}({q_params})")
                md_lines.append("```")
                md_lines.append("")
                if accumulated_comments:
                    md_lines.append("\n".join(accumulated_comments))
                    md_lines.append("")
                else:
                    md_lines.append("*No description available.*")
                    md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
                
                accumulated_comments = []
                continue

            # Match standard func declarations
            func_match = re.match(r'func\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(\w+))?', line_strip)
            if func_match:
                f_name = func_match.group(1)
                f_params = func_match.group(2)
                f_ret = func_match.group(3) or "void"
                
                md_lines.append(f"## Classical Function `{f_name}`")
                md_lines.append("")
                md_lines.append("```eigen")
                ret_str = f" -> {f_ret}" if func_match.group(3) else ""
                md_lines.append(f"func {f_name}({f_params}){ret_str}")
                md_lines.append("```")
                md_lines.append("")
                if accumulated_comments:
                    md_lines.append("\n".join(accumulated_comments))
                    md_lines.append("")
                else:
                    md_lines.append("*No description available.*")
                    md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
                
                accumulated_comments = []
                continue

            # Match struct declarations
            struct_match = re.match(r'struct\s+(\w+)', line_strip)
            if struct_match:
                s_name = struct_match.group(1)
                
                md_lines.append(f"## Structure `{s_name}`")
                md_lines.append("")
                md_lines.append("```eigen")
                md_lines.append(f"struct {s_name}")
                md_lines.append("```")
                md_lines.append("")
                if accumulated_comments:
                    md_lines.append("\n".join(accumulated_comments))
                    md_lines.append("")
                else:
                    md_lines.append("*No description available.*")
                    md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
                
                accumulated_comments = []
                continue

            # If it's a blank line, keep comments.
            # If it's code, but not a declaration, clear comments.
            if line_strip:
                accumulated_comments = []
                
        return "\n".join(md_lines)
