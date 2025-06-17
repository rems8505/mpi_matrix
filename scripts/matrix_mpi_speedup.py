from mpi4py import MPI
import numpy as np
import time
import sys

def distribute_data(comm, a, b, size):
    """Distribute matrix data across MPI nodes"""
    rank = comm.Get_rank()
    n = a.shape[0] if rank == 0 else None
    
    # Broadcast matrix size first
    n = comm.bcast(n, root=0)
    
    # Scatter rows of matrix A
    rows_per_node = n // size
    a_local = np.zeros((rows_per_node, n), dtype=np.float32)  # Use float32 to save memory
    
    if rank == 0:
        # Split matrix A into chunks for scattering
        a_chunks = np.split(a[:size*rows_per_node], size)
    else:
        a_chunks = None
    
    comm.Scatter(a_chunks, a_local, root=0)
    
    # Broadcast entire matrix B to all nodes
    b_local = np.zeros((n, n), dtype=np.float32)
    comm.Bcast(b, root=0)
    
    return a_local, b_local

def gather_results(comm, local_result, result_shape):
    """Gather results from all nodes"""
    result = np.zeros(result_shape, dtype=np.float32) if comm.Get_rank() == 0 else None
    comm.Gather(local_result, result, root=0)
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Matrix size (adjustable via command line)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # Default 200x200 for Free Tier
    
    # Only root generates initial matrices
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"Matrix Multiplication: {n}x{n} | Nodes: {size}")
        print(f"{'='*50}")
        
        # Generate random matrices (float32 for memory efficiency)
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        
        # Serial computation for comparison
        serial_start = time.time()
        serial_result = np.dot(a, b)
        serial_end = time.time()
        serial_time = serial_end - serial_start
        
        print(f"\n[Serial] Time: {serial_time:.4f}s")
        mpi_start = time.time()
    else:
        a, b = None, None
        serial_time = None

    # Distribute data
    a_local, b_local = distribute_data(comm, a, b, size)
    
    # Local computation
    local_start = time.time()
    local_result = np.dot(a_local, b_local)
    local_end = time.time()
    
    # Gather results
    result = gather_results(comm, local_result, (n, n))
    
    # Performance analysis (only on root)
    if rank == 0:
        mpi_end = time.time()
        mpi_time = mpi_end - mpi_start
        
        # Verify results match
        is_correct = np.allclose(result, serial_result, atol=1e-4)
        
        # Calculate metrics
        speedup = serial_time / mpi_time
        efficiency = (speedup / size) * 100
        
        print(f"\n[MPI] Time: {mpi_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x (Ideal: {size}x)")
        print(f"Efficiency: {efficiency:.2f}%")
        print(f"Result Correct: {'✅' if is_correct else '❌'}")
        
        # Performance summary table
        print("\nPerformance Summary:")
        print(f"{'Metric':<15} | {'Value':>10}")
        print(f"{'-'*15} | {'-'*10}")
        print(f"{'Serial Time':<15} | {serial_time:>10.4f}s")
        print(f"{'MPI Time':<15} | {mpi_time:>10.4f}s")
        print(f"{'Speedup':<15} | {speedup:>10.2f}x")
        print(f"{'Efficiency':<15} | {efficiency:>9.2f}%")
        
        # Save results to file
        with open("performance.log", "a") as f:
            f.write(f"{n},{size},{serial_time:.4f},{mpi_time:.4f},{speedup:.2f},{efficiency:.2f}\n")

if __name__ == "__main__":
    main()




# Key Features of This Implementation
# Memory Optimization:
# Uses float32 instead of float64 to reduce memory usage (critical for Free Tier)
# Splits matrices efficiently for scattering


# Performance Metrics:
# Speedup: Serial Time / MPI Time
# Efficiency: (Speedup / Number of Nodes) * 100
# Result Validation: Checks if MPI results match serial results

# Command Line Flexibility:
# # Run with default 200x200 matrix
# mpirun -np 2 --host master,worker1 python3 matrix_mpi.py

# # Run with custom size (e.g., 150x150)
# mpirun -np 2 --host master,worker1 python3 matrix_mpi.py 150


# Data Logging:
# Appends results to performance.log in CSV format:
# 200,2,0.3821,0.2176,1.76,87.82
# 150,2,0.2015,0.1253,1.61,80.50