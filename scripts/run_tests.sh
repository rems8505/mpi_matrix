#!/bin/bash
SIZES=(100 200)  # Reduced for t2.micro
for size in "${SIZES[@]}"; do
    echo -e "\nTesting Matrix Size: ${size}x${size}"
    python3 matrix_serial.py $size
    mpirun -np 2 --host master,worker1 python3 matrix_mpi.py $size
done