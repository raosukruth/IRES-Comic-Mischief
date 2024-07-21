import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Use PyTorch's CUDA memory management functions
        total_memory = torch.cuda.get_device_properties(i).total_memory
        reserved_memory = torch.cuda.memory_reserved(i)
        allocated_memory = torch.cuda.memory_allocated(i)

        # Use CUDA API to get free memory
        free_memory_cuda, total_memory_cuda = torch.cuda.mem_get_info(i)
        
        print(f"  Total memory (from device properties): {total_memory / (1024 ** 3):.2f} GB")
        print(f"  Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")
        print(f"  Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
        print(f"  Free memory (from CUDA API): {free_memory_cuda / (1024 ** 3):.2f} GB")
        print(f"  Total memory (from CUDA API): {total_memory_cuda / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available. No GPUs detected.")