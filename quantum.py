# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Defining the Potential Energy
def potentialV(x):
    return x ** 2


# Input Computational Parameters
lower_limit = float(input("Enter the lower limit of the domain: "))
upper_limit = float(input("Enter the upper limit of the domain: "))
grid_points_N = int(input("Enter the number of grid points: "))

# Defining x and the step size h
x = np.linspace(lower_limit, upper_limit, grid_points_N)
h = x[1] - x[0]

# Creating the kinetic energy matrix
kinetic_energy_T = np.zeros((grid_points_N - 2) ** 2).reshape(grid_points_N - 2, grid_points_N - 2)

for i in range(grid_points_N - 2):
    for j in range(grid_points_N - 2):
        if i == j:
            kinetic_energy_T[i, j] = -2
        elif np.abs(i - j) == 1:
            kinetic_energy_T[i, j] = 1
        else:
            kinetic_energy_T[i, j] = 0

# Creating the potential energy matrix
potential_energy_V = np.zeros((grid_points_N - 2) ** 2).reshape(grid_points_N - 2, grid_points_N - 2)

for i in range(grid_points_N - 2):
    for j in range(grid_points_N - 2):
        if i == j:
            potential_energy_V[i, j] = potentialV(x[i + 1])
        else:
            potential_energy_V[i, j] = 0

# Creating the Hamiltonian Matrix
H = -kinetic_energy_T / (2 * h ** 2) + potential_energy_V

# Find the eigen values and eigen vectors and sort them in ascending order
val, vec = np.linalg.eig(H)
z = np.argsort(val)
z = z[0:5]
energies = (val[z] / val[z][0])
print(energies)

# Plot wave functions for first 5 lowest states
plt.figure(figsize=(12, 10))
for i in range(len(z)):
    y = []
    y = np.append(y, vec[:, z[i]])
    y = np.append(y, 0)
    y = np.insert(y, 0, 0)
    plt.plot(x, y, lw=3, label="{} ".format(i))
plt.legend()
plt.title('Normalized wave functions for a Harmonic Oscillator using Finite Difference Method', size=14)
plt.show()
