def mean(arr):
    # arr is a VMRef to HeapObject of type 'array' or a python list
    # If it is a VMRef, the VM should extract the data list from heap!
    # Let's write the VM to unpack the arguments before passing to stdlib,
    # or handle unpacking here. Unpacking in VM is much cleaner!
    if not arr:
        return 0.0
    return sum(arr) / len(arr)

def variance(arr):
    if not arr:
        return 0.0
    m = mean(arr)
    return sum((x - m) ** 2 for x in arr) / len(arr)
