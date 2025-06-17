#!/bin/bash
SIZES=(100 150 200)  # Matrix sizes to test

for size in "${SIZES[@]}"; do
    echo "Testing size: ${size}x${size}"
    mpirun -np 2 --host master,worker1 python3 matrix_mpi.py $size
    echo ""
done

# Execute the benchmarks:
# bash
# chmod +x run_benchmarks.sh
# ./run_benchmarks.sh