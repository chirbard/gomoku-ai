import numpy as np
import torch
import time

# Matrix size
matrix_size = 1000

# NumPy on CPU
start_time = time.time()
np_matrix_a = np.random.rand(matrix_size, matrix_size)
np_matrix_b = np.random.rand(matrix_size, matrix_size)
np_result = np.dot(np_matrix_a, np_matrix_b)
numpy_time = time.time() - start_time
print(f"NumPy (CPU) time: {numpy_time:.6f} seconds")

# PyTorch on GPU
if torch.cuda.is_available():
    start_time = time.time()
    torch_matrix_a = torch.rand(matrix_size, matrix_size, device='cuda')
    torch_matrix_b = torch.rand(matrix_size, matrix_size, device='cuda')
    torch_result = torch.mm(torch_matrix_a, torch_matrix_b)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    torch_time = time.time() - start_time
    print(f"PyTorch (GPU) time: {torch_time:.6f} seconds")
else:
    print("CUDA is not available. Cannot perform GPU calculations.")


# matrix size 15000
# NumPy (CPU) time: 49.266316 seconds
# PyTorch (GPU) time: 1.827247 seconds

# matrix size 1000
# NumPy (CPU) time: 0.043604 seconds
# PyTorch (GPU) time: 0.129130 seconds
