import os

def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def write_file(filepath, content):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return 1
    except Exception:
        return 0

def print_format(fmt, val):
    print(fmt % val)
    return 0
