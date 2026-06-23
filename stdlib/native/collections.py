def append_int(arr, val):
    arr.append(val)
    return len(arr)

def remove_at(arr, idx):
    if 0 <= idx < len(arr):
        return arr.pop(idx)
    return -1
