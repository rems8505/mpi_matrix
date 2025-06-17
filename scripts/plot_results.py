import matplotlib.pyplot as plt

# Example data (replace with actual results.log parsing)
sizes = [100, 200]
serial_times = [0.05, 0.40]
mpi_times = [0.03, 0.22]

plt.plot(sizes, serial_times, 'r-', label='Serial')
plt.plot(sizes, mpi_times, 'b--', label='MPI (2 Nodes)')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.legend()
plt.grid()
plt.savefig('performance.png')