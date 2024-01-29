import torch
from ray import get_gpu_ids
cuda = torch.cuda.is_available()
device_count = torch.cuda.device_count()
current_device = torch.cuda.current_device()
device = torch.cuda.device(0)
device_model = torch.cuda.get_device_name(0)
print(f"Has cuda: {cuda}")
print(f"Number of devices: {device_count}")
print(f"Current device: {current_device}")
print(f"Device: {device}")
print(f"Device model: {device_model}")
print(f"GPU ids: {get_gpu_ids()}")


import ray
@ray.remote(num_gpus=1)
def f():
    import os
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
ray.get(f.remote())