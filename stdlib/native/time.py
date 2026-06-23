import time

def now():
    return time.time()

def sleep(ms):
    time.sleep(ms / 1000.0)
    return 0
