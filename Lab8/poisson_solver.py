# !V1

import numpy as np
import matplotlib.pyplot as plt

def main():
    # !Insert problem data ---------------------

    # Define the shape of the area considered and its elements
    area_length = 5 # m
    
    plate_length = 1 # m
    plate_width = 0.1 # m

    plate_distance = 0.5 # m

    plate1_potential = +100 # V
    plate2_potential = -100 # V
    border_potential =    0 # V

    # Define the number of steps (could differ in each direction)
    N = 1000

    # Define the convergence parameter
    epsilon = 1e-5

    # ------------------------------------------

    # Define the indeces of the plates
    plates_start_y = int(N * (area_length - plate_length) / (2 * area_length))
    plates_end_y = int(N * (area_length + plate_length) / (2 * area_length))
    
    plate1_start_x = int(N * (area_length - plate_width - plate_distance) / (2 * area_length))
    plate1_end_x = int(N * (area_length + plate_width - plate_distance) / (2 * area_length))
    plate2_start_x = int(N * (area_length - plate_width + plate_distance) / (2 * area_length))
    plate2_end_x = int(N * (area_length + plate_width + plate_distance) / (2 * area_length))

    # Define the effect of the matrix M on a general vector V
    def M(V):
        # Calculate the mean value of the neighbors using numpy.roll
        V_neighbors = np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) + np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1)
        V_neighbors /= 4.0

        # Enforce zero change where boundary contidions apply (the borders and the plates) and the above formula does not apply
        V_neighbors[0, :] = V_neighbors[-1, :] = V_neighbors[:, 0] = V_neighbors[:, -1] = 0
        V_neighbors[plate1_start_x:plate1_end_x, plates_start_y:plates_end_y] = 0
        V_neighbors[plate2_start_x:plate2_end_x, plates_start_y:plates_end_y] = 0

        return V_neighbors

    # Initialize the initial potential with the given conditions
    X = np.linspace(0, area_length, N)
    Y = np.linspace(0, area_length, N)
    X, Y = np.meshgrid(X, Y)
    
    # Initialize the potential matrix V with boundary conditions
    V_initial = np.full((N, N), border_potential, dtype=float) # V at the borders
    V_initial[plate1_start_x:plate1_end_x, plates_start_y:plates_end_y] = plate1_potential # V at plate 1
    V_initial[plate2_start_x:plate2_end_x, plates_start_y:plates_end_y] = plate2_potential # V at plate 2

    V = V_initial.copy()

    # Iterate with the convergence condition as exit (Jacobi method)
    i = 0 # iteration counter (just for information purposes)
    while(True):
        i += 1 

        # Compute the next guess with the application of M
        V_guess = V_initial + M(V)

        # Check for convergence
        if(np.linalg.norm(V - V_guess) < epsilon): 
            V = V_guess
            break

        V = V_guess

    print(f"Result converged in {i} iterations.")

    # Plot the resulting potential
    plt.figure()
    plt.pcolormesh(X, Y, V.T, cmap="plasma") # Transpose V for correct orientation
    plt.colorbar(label='Potential (V)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Potential V(x, y)')

    # Compute the electric field
    Ex, Ey = np.gradient(-V, area_length / (N - 1))

    # Plot field lines
    plt.figure()
    speed = np.hypot(Ex, Ey)
    strm = plt.streamplot(X, Y, Ex.T, Ey.T, color=speed.T, cmap='plasma', density=1.5, linewidth=1)
    plt.colorbar(strm.lines, label='|E| (V/m)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Electric Field E(x, y)')

    # Apply the laplacian operator to the potential to get the charge density
    epsilon_0 = 8.854e-12 # F/m
    rho = -(np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) + np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (area_length / (N - 1))**2 * epsilon_0
    
    # Plot the charge density
    plt.figure()
    plt.pcolormesh(X, Y, rho.T, cmap="seismic")
    plt.colorbar(label='Charge Density (C/m²)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Charge Density ρ(x, y)')
    plt.show()

    return

if __name__ == "__main__":
    main()
