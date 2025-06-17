import numpy as np
import time

def matrix_multiply(a, b):
    return np.dot(a, b)

if __name__ == "__main__":
    n = 200  # Reduced size for t2.micro
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    
    start = time.time()
    result = matrix_multiply(a, b)
    end = time.time()
    
    print(f"Serial Time: {end - start:.4f}s | Result[0][0] = {result[0][0]}")