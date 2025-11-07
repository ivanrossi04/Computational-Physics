'''Program that simulates and animates the motion of a double pendulum using the Runge-Kutta 4th order method.'''

import math
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

# TODO: review the forces acting on the pendulum
# TODO: add parallelization to speed up multiple simulation
# TODO: animate multiple simulations together to show sensitivity to initial conditions

def animazione(coord: np.ndarray, dt: float, t: np.ndarray, trail_length: int = 50) -> None:
    '''
    This function creates an animation of the double pendulum trajectory.

    Parameters:
        coord : numpy array (4,N)
            An array containing the 4 Cartesian coordinates at each time step (x1, y1, x2, y2),
            with the origin of the axes coinciding with the fixed point of the pendulum.
        dt : float
            The time step in seconds.
        t : numpy array (N)
            An array containing the number of steps, used only to derive N.
        trail_length : int, optional
            The number of steps to keep for plotting the trajectory (default is 50).
    '''

    fig, ax = plt.subplots()
    
    history2x = np.zeros(( trail_length))
    history2y = np.zeros(( trail_length))
    history3x = np.zeros(( trail_length))
    history3y = np.zeros(( trail_length))
    
    # Define the line of the first pendulum
    l1x=np.zeros((2),dtype=np.float64)
    l1y=np.zeros((2),dtype=np.float64)
    l1x[0]=0
    l1y[0]=0
    l1x[1]=coord[0,0]
    l1y[1]=coord[1,0]
    line1 = ax.plot(l1x,l1y)[0]

    # Define the line of the second pendulum
    l2x=np.zeros((2),dtype=np.float64)
    l2y=np.zeros((2),dtype=np.float64)
    l2x[0]=coord[0,0]
    l2y[0]=coord[1,0]
    l2x[1]=coord[2,0]
    l2y[1]=coord[3,0]
    line2 = ax.plot(l2x,l2y)[0]

    # Trajectories
    trail2 = ax.plot(history2x, history2y, 'b-', alpha=0.5)[0]
    trail3 = ax.plot(history3x, history3y, 'g-', alpha=0.5)[0]
    # Discs
    disc1, = ax.plot([], [], 'ro', markersize=10)  # Disco rosso per il primo punto
    disc2, = ax.plot([], [], 'bo', markersize=10)  # Disco blu per il secondo punto
    disc3, = ax.plot([], [], 'go', markersize=10)  # Disco verde per il terzo punto

    ax.set(xlim=[-4, 4], ylim=[-4, 4], xlabel='X [m]', ylabel='Y [m]')
    ax.legend()


    def update(frame):
        l1x[0]=0
        l1y[0]=0
        l1x[1]=coord[0,frame]
        l1y[1]=coord[1,frame]
        l2x[0]=coord[0,frame]
        l2y[0]=coord[1,frame]
        l2x[1]=coord[2,frame]
        l2y[1]=coord[3,frame]
        line1.set_xdata(l1x)
        line1.set_ydata(l1y)
        line2.set_xdata(l2x)
        line2.set_ydata(l2y)

         # Aggiorna i dischi
        disc1.set_data([0], [0])  # Origine (0,0)
        disc2.set_data([coord[0,frame]], [coord[1,frame]])  # Secondo punto
        disc3.set_data([coord[2,frame]], [coord[3,frame]])  # Terzo punto
        
        # Aggiorna le traiettorie
        history2x = np.zeros(( trail_length))
        history2y = np.zeros(( trail_length))
        history3x = np.zeros(( trail_length))
        history3y = np.zeros(( trail_length))
        frame_in=max(frame-trail_length,0)
        for ifr in range(frame-frame_in):
            history2x[ifr]=coord[0,ifr+frame_in]
            history2y[ifr]=coord[1,ifr+frame_in]
            history3x[ifr]=coord[2,ifr+frame_in]
            history3y[ifr]=coord[3,ifr+frame_in]
        
        
        trail2.set_data(history2x, history2y)
        trail3.set_data(history3x, history3y)

        
        #history3 = np.roll(history3, -1, axis=1)
        #history3[:, -1] = [coord[2,frame], coord[3,frame]]
        #trail3.set_data(history3[0, :], history3[1, :])
        
        return line1, line2, disc1, disc2, disc3, trail2, trail3

    # make the window size a square, to avoid distortions
    ax.set_aspect('equal')

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=int(dt*1000),blit=True)
    plt.show()

def main():

    g = 9.81 # m/s^2
    # Pendulum variables
    # The initial positions are required from the user
    mass_1 = 1.0 # kg
    length_1 = 1.0 # m
    angle_1 = float(input("Insert the FIRST pendulum initial angle (in radians, the maximum values are +-pi rad = +-180 deg: ")) # rad
    angvel_1 = 0.0
    def f_1(theta_1, theta_2, omega_1, omega_2):
        a = -mass_2 * length_2 * math.sin(theta_1 - theta_2) * omega_2 ** 2
        b =  g * (mass_1 + mass_2) * math.sin(theta_1)
        c = -math.cos(theta_1 - theta_2) * (-mass_2 * length_1 * math.sin(theta_1 - theta_2) * omega_1 ** 2 + mass_2 * g * math.sin(theta_2))
        return -(a + b + c) / (length_1 * (mass_1 + mass_2 * (1 - math.cos(theta_1 - theta_2) ** 2)))

    mass_2 = 1.0 # kg
    length_2 = 1.0 # m
    angle_2 = float(input("Insert the SECOND pendulum initial angle (in radians, the maximum values are +-pi rad = +-180 deg: ")) # rad
    angvel_2 = 0.0
    def f_2(theta_1, theta_2, omega_1, omega_2):
        a = -mass_2 / (mass_1 + mass_2) * math.cos(theta_1 - theta_2) * (mass_2 * length_2 * math.sin(theta_1 - theta_2) * omega_2 ** 2 + g * (mass_1 + mass_2) * math.sin(theta_1))
        b = -mass_2 * length_1 * math.sin(theta_1 - theta_2) * omega_1 ** 2
        c =  mass_2 * g * math.sin(theta_2)

        return -(a + b + c) / (length_2 * (mass_1 + mass_2 * (1 - math.cos(theta_1 - theta_2) ** 2)))

    trajectories = []
    for i in range(2):
        angle_1 += i * 0.01

        # Time variables
        time = 0.0 # s
        max_time = 10.0 # s
        dt = 0.001 # s
        
        trajectory = [[
            length_1 * math.sin(angle_1),
            -length_1 * math.cos(angle_1),
            length_1 * math.sin(angle_1) + length_2 * math.sin(angle_2),
            -length_1 * math.cos(angle_1) - length_2 * math.cos(angle_2)
        ]]

        state = np.array([angle_1, angvel_1, angle_2, angvel_2])

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
        trajectories.append(trajectory)

    t = np.arange(0, max_time, dt*20)
    for traj in trajectories:
        animazione(traj, dt*20, t)


if __name__ == "__main__":
    main()
