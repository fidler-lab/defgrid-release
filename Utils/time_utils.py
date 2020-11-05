import time
t0 = time.time()
import torch

def tt(s, batch=1):
    """
    Prints the time

    Args:
        s: (int): write your description
        batch: (todo): write your description
    """
    global t0
    torch.cuda.synchronize()
    print(s, ' ', (time.time() - t0) / batch)
    t0 = time.time()
