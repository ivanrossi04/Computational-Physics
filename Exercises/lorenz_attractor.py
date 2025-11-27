# !V1

'''
Lorenz Attractor Simulation and Visualization
This script simulates the Lorenz attractor using the Symplectic Euler method
Two simulations are run with slightly different initial conditions to illustrate sensitivity to initial conditions.
Because the trajectories can get very close, an animation is run to help visualize the difference over time.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # Parameters for the Lorenz attractor
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = 28.0

    # Time parameters
    dt = 0.01
    num_steps = 5000

    # Arrays to hold the trajectories of the two simulations
    x = np.zeros((num_steps, 2))
    y = np.zeros((num_steps, 2))
    z = np.zeros((num_steps, 2))

    # Initial conditions
    # First simulation
    x[0, 0], y[0, 0], z[0, 0] = 1.0, 1.0, 1.0
    # Second simulation with a slight offset
    x[0, 1], y[0, 1], z[0, 1] = x[0, 0] + 0.01, y[0, 0], z[0, 0]
    
    # Time integration using the Symplectic Euler method
    for j in range(2):  # Loop over the two simulations
        for i in range(1, num_steps):
            dx = sigma * (y[i-1, j] - x[i-1, j])
            dy = x[i-1, j] * (rho - z[i-1, j]) - y[i-1, j]
            dz = x[i-1, j] * y[i-1, j] - beta * z[i-1, j]

            x[i, j] = x[i-1, j] + dx * dt
            y[i, j] = y[i-1, j] + dy * dt
            z[i, j] = z[i-1, j] + dz * dt

    # Plotting the Lorenz attractor
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:, 0], y[:, 0], z[:, 0], label="Simulation 1")
    ax.plot(x[:, 1], y[:, 1], z[:, 1], label="Simulation 2")
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()

    def update(i):
        ax.lines[0].set_data(x[:i, 0], y[:i, 0]), ax.lines[0].set_3d_properties(z[:i, 0])
        ax.lines[1].set_data(x[:i, 1], y[:i, 1]), ax.lines[1].set_3d_properties(z[:i, 1])
        return [ax.lines[0], ax.lines[1]]

    ani = animation.FuncAnimation(fig = fig, func = update, frames = num_steps, interval = 10, blit = True)

    plt.show()

if __name__ == "__main__":
    main()
