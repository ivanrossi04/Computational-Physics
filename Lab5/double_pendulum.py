# !V1

'''Program that simulates and animates the motion of a double pendulum using the Runge-Kutta 4th order method.'''

import math
import numpy as np

import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import threading

N = 10
trajectories = [None] * N # List to store trajectories of each pendulum
trajectories_lock = threading.Lock() # Lock for synchronizing access to trajectories

def animate_pendulums(all_coords: list, dt: float, trail_length: int = 50) -> None:
    '''
    This function creates an animation of multiple double pendulums simultaneously.

    Parameters:
        all_coords : list of numpy arrays
            A list containing coordinate arrays for each pendulum. Each array has shape (4,N)
            containing the 4 Cartesian coordinates at each time step (x1, y1, x2, y2),
            with the origin of the axes coinciding with the fixed point of the pendulum.
        dt : float
            The time step in seconds.
        trail_length : int, optional
            The number of steps to keep for plotting the trajectory (default is 50).
    '''

    fig, ax = plt.subplots()
    
    # Define colors for each pendulum
    num_pendulums = len(all_coords)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_pendulums))
    
    # Create lists to store plot elements for each pendulum
    lines1 = []
    lines2 = []
    trails2 = []
    trails3 = []
    discs2 = []
    discs3 = []
    
    # Initialize plot elements for each pendulum
    for i, coord in enumerate(all_coords):
        color = colors[i]
        
        # Define the line of the first pendulum
        l1x = np.array([0, coord[0, 0]])
        l1y = np.array([0, coord[1, 0]])
        line1, = ax.plot(l1x, l1y, color=color, alpha=0.6, linewidth=1)
        lines1.append(line1)

        # Define the line of the second pendulum
        l2x = np.array([coord[0, 0], coord[2, 0]])
        l2y = np.array([coord[1, 0], coord[3, 0]])
        line2, = ax.plot(l2x, l2y, color=color, alpha=0.6, linewidth=1)
        lines2.append(line2)

        # Trajectories
        trail2, = ax.plot([], [], color=color, alpha=0.3, linewidth=0.5)
        trails2.append(trail2)
        trail3, = ax.plot([], [], color=color, alpha=0.5, linewidth=1)
        trails3.append(trail3)
        
        # Discs
        disc2, = ax.plot([], [], 'o', color=color, markersize=6, alpha=0.7)
        discs2.append(disc2)
        disc3, = ax.plot([], [], 'o', color=color, markersize=8, alpha=0.9)
        discs3.append(disc3)

    # Fixed origin disc
    origin_disc, = ax.plot([0], [0], 'ko', markersize=8)

    ax.set(xlim=[-4, 4], ylim=[-4, 4], xlabel='X [m]', ylabel='Y [m]')
    ax.set_aspect('equal')
    ax.set_title(f'Double Pendulum Simulation - {num_pendulums} Pendulums')
    ax.grid(True, alpha=0.3)

    def update(frame):
        artists = []
        
        for i, coord in enumerate(all_coords):
            # Update pendulum arms
            l1x = np.array([0, coord[0, frame]])
            l1y = np.array([0, coord[1, frame]])
            lines1[i].set_data(l1x, l1y)
            
            l2x = np.array([coord[0, frame], coord[2, frame]])
            l2y = np.array([coord[1, frame], coord[3, frame]])
            lines2[i].set_data(l2x, l2y)

            # Update discs
            discs2[i].set_data([coord[0, frame]], [coord[1, frame]])
            discs3[i].set_data([coord[2, frame]], [coord[3, frame]])
            
            # Update trajectories
            frame_start = max(frame - trail_length, 0)
            history2x = coord[0, frame_start:frame+1]
            history2y = coord[1, frame_start:frame+1]
            history3x = coord[2, frame_start:frame+1]
            history3y = coord[3, frame_start:frame+1]
            
            trails2[i].set_data(history2x, history2y)
            trails3[i].set_data(history3x, history3y)
            
            artists.extend([lines1[i], lines2[i], discs2[i], discs3[i], trails2[i], trails3[i]])
        
        artists.append(origin_disc)
        return artists

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(all_coords[0][0]), interval=int(dt*1000), blit=True)
    plt.show()

def propagate(state, dt, max_time, number):

    # Physical constants
    g = 9.81 # m/s^2
    
    # Pendulum variables
    # The initial positions are required from the user
    mass_1 = 1.0 # kg
    length_1 = 1.0 # m
    def f_1(theta_1, theta_2, omega_1, omega_2):
        # Angular acceleration of the first pendulum
        delta = theta_2 - theta_1
        denominator = (mass_1 + mass_2) * length_1 - mass_2 * length_1 * math.cos(delta)**2
        
        numerator = (mass_2 * length_1 * omega_1**2 * math.sin(delta) * math.cos(delta) +
                    mass_2 * g * math.sin(theta_2) * math.cos(delta) +
                    mass_2 * length_2 * omega_2**2 * math.sin(delta) -
                    (mass_1 + mass_2) * g * math.sin(theta_1))
        
        return numerator / denominator
    
    mass_2 = 1.0 # kg
    length_2 = 1.0 # m
    def f_2(theta_1, theta_2, omega_1, omega_2):
        # Angular acceleration of the second pendulum
        delta = theta_2 - theta_1
        denominator = (mass_1 + mass_2) * length_1 - mass_2 * length_1 * math.cos(delta)**2
        
        numerator = (-mass_2 * length_2 * omega_2**2 * math.sin(delta) * math.cos(delta) +
                     (mass_1 + mass_2) * g * math.sin(theta_1) * math.cos(delta) -
                     (mass_1 + mass_2) * length_1 * omega_1**2 * math.sin(delta) -
                     (mass_1 + mass_2) * g * math.sin(theta_2))
        
        return numerator / denominator

    # Time variables
    time = 0.0 # s
    
    trajectory = [[
        length_1 * math.sin(state[0]),
        -length_1 * math.cos(state[0]),
        length_1 * math.sin(state[0]) + length_2 * math.sin(state[2]),
        -length_1 * math.cos(state[0]) - length_2 * math.cos(state[2])
    ]]

    def derivatives(state):
        theta_1, omega_1, theta_2, omega_2 = state

        dtheta_1_dt = omega_1
        domega_1_dt = f_1(theta_1, theta_2, omega_1, omega_2)
        dtheta_2_dt = omega_2
        domega_2_dt = f_2(theta_1, theta_2, omega_1, omega_2)

        return np.array([dtheta_1_dt, domega_1_dt, dtheta_2_dt, domega_2_dt])

    while(time < max_time):

        k1 = dt * derivatives(state)
        k2 = dt * derivatives(state + k1/2)
        k3 = dt * derivatives(state + k2/2)
        k4 = dt * derivatives(state + k3)
        state += (k1 + 2*k2 + 2*k3 + k4) / 6
        time += dt

        if int(time / dt) % 20 == 0: 
            trajectory.append([
                length_1 * math.sin(state[0]),
                -length_1 * math.cos(state[0]),
                length_1 * math.sin(state[0]) + length_2 * math.sin(state[2]),
                -length_1 * math.cos(state[0]) - length_2 * math.cos(state[2])
            ])

    trajectory = np.array(trajectory).T

    # A semaphore here should be necessary to avoid race conditions
    with trajectories_lock:
        trajectories[number] = trajectory

def main():
    dt = 0.001 # s
    max_time = 10.0 # s

    angle_1 = float(input("Insert the FIRST pendulum initial angle (rad):")) # rad
    angvel_1 = 0.0

    angle_2 = float(input("Insert the SECOND pendulum initial angle (rad):")) # rad
    angvel_2 = 0.0

    initial_state = [angle_1, angvel_1, angle_2, angvel_2]

    threads = []
    for i in range(N):
        initial_state_copy = initial_state.copy()
        initial_state_copy[0] += i * 0.001 # Slightly vary the initial angle for each pendulum
        t = threading.Thread(target=propagate, args=(initial_state_copy, dt, max_time, i))
        threads.append(t)

    start_time = time.perf_counter()

    # Start each thread
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, "seconds")

    # Animate all pendulums simultaneously in a single plot
    animate_pendulums(trajectories, dt*20)

if __name__ == "__main__":
    main()
