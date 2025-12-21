# !V1

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from numba import jit

# @jit(nopython=True)
def update_wave(u_prev: np.ndarray, u_curr: np.ndarray, u_next: np.ndarray, room_layout: np.ndarray, size: int, courant_sq: float) -> None:
    '''
    Update the wave state using finite difference methods.
    
    Args:
        u_prev: The wave state at the previous time step
        u_curr: The wave state at the current time step
        u_next: The wave state at the next time step (computation result stored here)
        room_layout: Boolean array representing walls (True) and free space (False)
        size: The size of the 2D arrays (number of grid points per dimension)
        courant_sq: The square of the Courant number (c * DeltaT / DeltaX)^2
    '''

    # Update the interior points using finite difference
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if not room_layout[i, j]:  # Free space
                laplacian = (u_curr[i+1, j] + u_curr[i-1, j] + 
                           u_curr[i, j+1] + u_curr[i, j-1] - 4 * u_curr[i, j])
                u_next[i, j] = (2 * u_curr[i, j] - u_prev[i, j] + 
                              courant_sq * laplacian)
            else:  # Wall - no change
                u_next[i, j] = u_curr[i, j]
    
    # Apply Neumann boundary conditions at the edges
    u_next[0, :] = u_next[1, :]      # Top
    u_next[size-1, :] = u_next[size-2, :]  # Bottom
    u_next[:, 0] = u_next[:, 1]      # Left
    u_next[:, size-1] = u_next[:, size-2]  # Right

def main():
    # Simulation parameters
    N = 100  # Number of grid points per dimension
    N_steps = 2001  # Number of time steps
    L = 10.0  # Room side length in meters
    DeltaT = 1e-4  # Time step (s)
    
    # Physical parameters
    c = 300.0  # Speed of sound (m/s)
    
    # Wave parameters
    A = 1.0  # Amplitude of the wave
    x0 = 2.0  # Initial x position of the wave center (m)
    y0 = 2.0  # Initial y position of the wave center (m)
    
    DeltaX = L / (N - 1)  # Spatial step (m)
    courant_sq = (c * DeltaT / DeltaX) ** 2  # Courant number squared
    print(f"Courant squared: {courant_sq}")
    
    if courant_sq >= 0.5:
        print("Warning: Courant condition not satisfied, the simulation may be unstable! Aborting simulation.")
        return -1
    
    # Create room layout: wall in the middle with a hole
    room_layout = np.zeros((N, N), dtype=bool)
    mid_index = N // 2
    hole_size = N // 14
    hole_start = mid_index - hole_size // 2
    hole_end = hole_start + hole_size
    
    # Place wall, of width w = (10m / N), with a hole in the center
    for i in range(N):
        if not (hole_start <= i < hole_end):
            room_layout[i, mid_index] = True
    
    # Configure the first state of the wave (Gaussian initial condition)
    u1 = np.zeros((N, N))
    for i in range(N):
        x = i * DeltaX
        for j in range(N):
            y = j * DeltaX
            if not room_layout[i, j]:
                u1[i, j] = A * np.exp(-((x - x0)**2 + (y - y0)**2) / 0.5)
    
    # Initialize u2 using Taylor expansion (initial velocity = 0)
    u2 = np.zeros((N, N))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if not room_layout[i, j]:  # Free space only
                laplacian = (u1[i+1, j] + u1[i-1, j] + u1[i, j+1] + u1[i, j-1] - 4 * u1[i, j])
                
                u2[i, j] = u1[i, j] + 0.5 * courant_sq * laplacian
    
    # Apply Neumann boundary conditions to u2
    u2[0, :] = u2[1, :] # Top
    u2[N-1, :] = u2[N-2, :] # Bottom
    u2[:, 0] = u2[:, 1] # Left
    u2[:, N-1] = u2[:, N-2] # Right

    # Set up the plot
    plt.figure()

    display_data = np.flipud(u1.copy())
    im = plt.imshow(display_data, extent=[0, L, 0, L], origin='lower', cmap='seismic', vmin=-1, vmax=1)
    
    cbar = plt.colorbar(im, label='Wave Amplitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('2D Sound Wave Propagation')
    
    # Temporary array for updates
    tmp = np.zeros((N, N))
    
    def animate(frame):
        '''Animation update function called for each frame.'''

        # Perform 3 simulation updates per animation frame to avoid shifting matrices after each computation
        update_wave(u1, u2, tmp, room_layout, N, courant_sq)
        update_wave(u2, tmp, u1, room_layout, N, courant_sq)
        update_wave(tmp, u1, u2, room_layout, N, courant_sq)
        
        # Update display
        display_data = np.flipud(u2.copy())
        im.set_array(display_data)
        
        return [im]
    
    # Create animation (update every 3 simulation steps)
    num_frames = N_steps // 3
    print(f"Running {N_steps} simulation steps with {num_frames} animation frames...")
    
    ani = animation.FuncAnimation(plt.gcf(), animate, frames=num_frames, interval=20, blit=True, repeat=False)
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()