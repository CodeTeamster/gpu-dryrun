import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
from pynvml import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_available_gpus():
    """Get available GPU information and filter based on memory status"""
    nvmlInit()
    available_gpus = []

    try:
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)

            total_memory = info.total / (1024**3)  # Convert to GB
            free_memory = info.free / (1024**3)    # Available memory
            memory_usage = (total_memory - free_memory) / total_memory * 100  # Memory usage percentage
            gpu_utilization = utilization.gpu  # GPU utilization

            # Only select GPUs with memory usage < 50% and GPU utilization < 30%
            if memory_usage < 70 and gpu_utilization < 30:
                available_gpus.append({
                    'index': i,
                    'total_memory': total_memory,
                    'free_memory': free_memory,
                    'target_memory': free_memory * 0.6,  # Target: use 60% of free memory
                    'utilization': gpu_utilization
                })
    finally:
        nvmlShutdown()

    # Sort by available memory in descending order
    available_gpus.sort(key=lambda x: x['free_memory'], reverse=True)
    return available_gpus

def calculate_model_size(gpu_memory):
    """Calculate model parameters based on available GPU memory"""
    # Reserve 20% memory for intermediate calculations and overhead
    available_memory = gpu_memory * 0.8  # GB
    # Rough estimate: each parameter takes 4 bytes (float32)
    available_params = int((available_memory * 1024**3) / 4)

    # Calculate layer dimensions (simple proportional allocation)
    in_dim = 8192  # Keep input dimension constant
    out_dim = 4096  # Keep output dimension constant
    # Adjust middle layer size based on available memory
    mid_dim = min(8192, int(available_params / (in_dim + out_dim)))

    return in_dim, mid_dim, out_dim

def train(rank, world_size, gpu_indices):
    setup(rank, world_size)

    # Get current GPU memory information
    gpu_info = get_available_gpus()[rank]
    in_dim, mid_dim, out_dim = calculate_model_size(gpu_info['target_memory'])

    print(f"GPU {gpu_indices[rank]}: Using model dimensions - Input: {in_dim}, Hidden: {mid_dim}, Output: {out_dim}")

    model = nn.Sequential(
        nn.Linear(in_dim, mid_dim),
        nn.ReLU(),
        nn.Linear(mid_dim, out_dim)
    ).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Adjust batch_size based on GPU memory
    batch_size = min(64, int(gpu_info['target_memory'] * 1024 / 4))  # Simple estimate, 4KB per sample
    dataset = TensorDataset(torch.randn(1000, in_dim), torch.randint(0, out_dim, (1000,)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    while True:
        sampler.set_epoch(0)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(rank))
            loss = nn.CrossEntropyLoss()(outputs, labels.to(rank))
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f'Loss: {loss.item():.3f}')

    cleanup()

def main():
    available_gpus = get_available_gpus()

    # If no suitable GPUs are available, exit the program
    if not available_gpus:
        print("No suitable GPUs found!")
        return

    # Print available GPU information
    print(f"Found {len(available_gpus)} suitable GPUs:")
    for gpu in available_gpus:
        print(f"GPU {gpu['index']}: {gpu['total_memory']:.2f}GB total, "
              f"{gpu['free_memory']:.2f}GB free, "
              f"Utilization: {gpu['utilization']}%")

    # Set the GPUs to use
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        gpu_indices = cuda_visible_devices.split(",")
        gpu_indices = [int(idx) for idx in gpu_indices]
    else:
        gpu_indices = [gpu['index'] for gpu in available_gpus]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))

    world_size = len(gpu_indices)
    print(f"\nUsing {world_size} GPUs: {gpu_indices}")

    # Start training
    torch.multiprocessing.spawn(train, args=(world_size, gpu_indices), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
