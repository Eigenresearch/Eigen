import re

class EigenFormatter:
    def __init__(self):
        pass

    def format_line_content(self, line: str) -> str:
        # Separate line from comments
        comment = ""
        # Check for # or // comment
        # Note: Be careful about comments inside strings.
        # For simplicity, we search for comment symbol from right or left, but we can do a simple character loop.
        
        in_string = False
        comment_start = -1
        i = 0
        while i < len(line):
            char = line[i]
            if in_string:
                if char == '\\' and i + 1 < len(line):
                    i += 2
                    continue
                if char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '#' or (char == '/' and i + 1 < len(line) and line[i+1] == '/'):
                    comment_start = i
                    break
            i += 1
            
        if comment_start != -1:
            comment = line[comment_start:]
            code = line[:comment_start]
        else:
            code = line

        # Format code spaces
        code = code.strip()
        if not code:
            return comment.strip()

        # Tokenize code loosely to adjust spacing
        # Adjust spacing around operators, but not within strings
        parts = []
        in_string = False
        string_buf = ""
        other_buf = ""
        
        i = 0
        while i < len(code):
            char = code[i]
            if in_string:
                string_buf += char
                if char == '\\' and i + 1 < len(code):
                    string_buf += code[i+1]
                    i += 2
                    continue
                if char == '"':
                    parts.append((True, string_buf))
                    string_buf = ""
                    in_string = False
                i += 1
                continue
            if char == '"':
                if other_buf:
                    parts.append((False, other_buf))
                    other_buf = ""
                string_buf = char
                in_string = True
            else:
                other_buf += char
            i += 1
            
        if other_buf:
            parts.append((False, other_buf))
        if string_buf:
            parts.append((True, string_buf))
            
        # Format the non-string parts
        formatted_parts = []
        for is_str, val in parts:
            if is_str:
                formatted_parts.append((True, val))
            else:
                # Add spaces around operators
                # Operators: ->, ==, =, +, -, *, /, :, ,
                v = val
                # Normalize spaces first
                v = re.sub(r'\s+', ' ', v)
                
                # Format arrows and comparators
                v = re.sub(r'\s*->\s*', ' -> ', v)
                v = re.sub(r'\s*==\s*', ' == ', v)
                v = re.sub(r'\s*!=\s*', ' != ', v)
                v = re.sub(r'\s*<=\s*', ' <= ', v)
                v = re.sub(r'\s*>=\s*', ' >= ', v)
                v = re.sub(r'\s*\+=\s*', ' += ', v)
                v = re.sub(r'\s*-=\s*', ' -= ', v)
                v = re.sub(r'\s*\*=\s*', ' *= ', v)
                v = re.sub(r'\s*/=\s*', ' /= ', v)
                v = re.sub(r'(?<![+\-*/!<>=])\s*=\s*(?!=)', ' = ', v)
                v = re.sub(r'\s*\+\s*(?!=)', ' + ', v)
                v = re.sub(r'\s*-\s*(?![>=])', ' - ', v)
                v = re.sub(r'\s*\*\s*(?!=)', ' * ', v)
                v = re.sub(r'\s*/\s*(?!=)', ' / ', v)
                v = re.sub(r'\s*<\s*(?![=])', ' < ', v)
                v = re.sub(r'(?<![-=])\s*>\s*(?![=])', ' > ', v)
                v = re.sub(r'\s*,\s*', ', ', v)
                v = re.sub(r'\s*:\s*', ': ', v)
                
                # Delimiters
                v = re.sub(r'\s*\(\s*', '(', v)
                v = re.sub(r'\s*\)\s*', ') ', v)
                v = re.sub(r'\s*\{\s*', ' {', v)
                
                # Cleanup double spaces
                v = re.sub(r'\s+', ' ', v)
                v = v.strip()
                formatted_parts.append((is_str, v))

        final_code = "".join(v for _, v in formatted_parts).strip()
        # Apply paren/curl cleanup only on non-string segments so string
        # contents (which may contain ')', ',', '{', etc.) are preserved.
        final_code = self._rejoin_formatted_parts(formatted_parts)
        
        if comment:
            if final_code:
                return f"{final_code}  {comment.strip()}"
            else:
                return comment.strip()
        return final_code

    def _rejoin_formatted_parts(self, formatted_parts) -> str:
        """Join formatted string/non-string parts and apply cross-boundary
        paren cleanup only to non-string segments so string contents are
        never modified."""
        if not formatted_parts:
            return ""
        result = ""
        for is_str, val in formatted_parts:
            if is_str:
                if result and not result.endswith((' ', '\t', '(')):
                    result += " "
                result += val
            else:
                seg = val
                if result and not result.endswith((' ', '\t', '(')):
                    if seg.startswith(('(', ')', '{', '}', ',', ':', '+', '-', '*', '/')):
                        seg = seg
                    else:
                        result += " "
                result += seg
        return result.strip()

    def format_code(self, source: str) -> str:
        lines = source.splitlines()
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                formatted_lines.append("")
                continue
                
            # If line starts with a closing brace, decrease indent before formatting
            if line_strip.startswith('}'):
                indent_level = max(0, indent_level - 1)
                
            code_line = self.format_line_content(line_strip)
            
            indent = "    " * indent_level
            if code_line:
                formatted_lines.append(f"{indent}{code_line}")
            else:
                formatted_lines.append("")
                
            # If line ends with an opening brace, increase indent for the next lines
            if line_strip.endswith('{'):
                indent_level += 1
                
        # Remove trailing empty lines and ensure single ending newline
        while formatted_lines and not formatted_lines[-1]:
            formatted_lines.pop()
            
        return "\n".join(formatted_lines) + "\n"
