from mpi4py import MPI
import numpy as np
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = 200  # Same as serial version

    if rank == 0:
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)
        start_time = time.time()
    else:
        a, b = None, None

    # Broadcast matrix B to all nodes
    b = comm.bcast(b, root=0)
    
    # Scatter rows of A
    rows_per_node = n // size
    a_local = np.zeros((rows_per_node, n))
    comm.Scatter(a, a_local, root=0)

    # Local computation
    local_result = np.dot(a_local, b)

    # Gather results
    result = None
    if rank == 0:
        result = np.zeros((n, n))
    comm.Gather(local_result, result, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"MPI Time ({size} nodes): {end_time - start_time:.4f}s")
        print(f"Result[0][0] = {result[0][0]}")

if __name__ == "__main__":
    main()