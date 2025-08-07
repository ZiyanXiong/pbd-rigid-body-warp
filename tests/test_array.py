import warp as wp
import numpy as np

# Initialize Warp
wp.init()

@wp.kernel
def create_buffer_kernel(
    # An output array to verify the result
    output: wp.array(dtype=wp.float32)
):
    # Get the unique thread ID
    tid = wp.tid()

    # 1. Create a buffer of 8 floats, local to this thread
    local_buffer = wp.array(dtype=wp.float32, length=8)

    # 2. Populate the buffer with values
    #    (e.g., fill with multiples of the thread ID)
    for i in range(8):
        local_buffer[i] = float(tid + 1) * float(i)

    # 3. Use the buffer (e.g., compute the sum)
    sum_val = float(0.0)
    for i in range(8):
        sum_val += local_buffer[i]

    # 4. Store the final result in the global output array
    output[tid] = sum_val

# --- Launch the kernel ---

# Number of threads to launch
num_threads = 4

# Create a Warp array on the GPU to store the output
# This array is in global memory, accessible by all threads
output_array = wp.empty(num_threads, dtype=wp.float32)

# Launch the kernel with 4 threads
wp.launch(
    kernel=create_buffer_kernel,
    dim=num_threads,
    inputs=[output_array]
)

# Copy the result back to the CPU (as a NumPy array) to print it
result_np = output_array.numpy()

print("Results from each thread:")
print(result_np)

# Expected output for thread `i`: (i+1)*0 + (i+1)*1 + ... + (i+1)*7 = (i+1)*28
# For thread 0: 1*28 = 28
# For thread 1: 2*28 = 56
# For thread 2: 3*28 = 84
# For thread 3: 4*28 = 112