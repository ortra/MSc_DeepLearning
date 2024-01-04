import torch

"""
this script check the device we are using
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
