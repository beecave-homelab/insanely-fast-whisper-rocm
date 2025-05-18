import torch
import time

while True:
    if torch.cuda.is_available():
        print("GPU is available")
        time.sleep(10)
    else:
        print("GPU is not available")
        time.sleep(10)
