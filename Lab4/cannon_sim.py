# !V1

import numpy as np
import math
import matplotlib.pyplot as plot

from numba import jit

# !Insert the problems data ------------------------------
g = 9.81 # m/s

# projectile data
r = 5 # cm
projectile_density = 2.7 # g/cm^3

# friction parameters
air_density = 1.22 # kg/m^3
air_resistance = 0.47

# initial values
theta = math.pi / 4 # rad
time_intervals = np.power(10.0, np.arange(-6, 0))
v_0 = 100 # m/s

x_0 = 0.0 # m

# sperical projectile computation
projectile_volume = 4 / 3 * math.pi * r ** 3 / 1e6 # m^3
projectile_area = math.pi * r ** 2 / 10000 # m^2
projectile_mass = projectile_volume * projectile_density * 1000 # kg

# pre-computed constants forces
g_force = projectile_mass * g
drag_coefficient = 0.5 * air_density * air_resistance * projectile_area
# --------------------------------------------------------


@jit(nopython=True)
def f_ideal(x: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        return np.array([0, -g_force])

@jit(nopython=True)
def f_drag(x: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        return np.array([0, -g_force]) - drag_coefficient * np.linalg.norm(v) * v

@jit(nopython=True)
def explicit_euler(position, velocity, time, mass, f, dt):
    time += dt
    force = f(position, velocity, time)
    position += velocity * dt
    velocity += force / mass * dt

    return position, velocity, time

@jit(nopython=True)
def runge_kutta(position, velocity, time, mass, f, dt):
    kx = np.zeros((4, 2))
    kv = np.zeros((4, 2))

    kx[0] = velocity
    kv[0] = f(position, kx[0], time) / mass

    kx[1] = velocity + kv[0] * (dt / 2)
    kv[1] = f(position + kx[0] * (dt/2), kx[1], time + dt / 2) / mass

    kx[2] = velocity + kv[1] * (dt / 2)
    kv[2] = f(position + kx[1] * (dt / 2), kx[2], time + dt / 2) / mass

    kx[3] = velocity + kv[2] * dt
    kv[3] = f(position + kx[2] * dt, kx[3], time + dt) / mass

    time += dt
    position += dt / 6 * (kx[0] + 2 * kx[1] + 2 * kx[2] + kx[3])
    velocity += dt / 6 * (kv[0] + 2 * kv[1] + 2 * kv[2] + kv[3])

    return position, velocity, time

@jit(nopython=True)
def propagate(position, velocity, time, delta_t, mass, force, method):
    while(True):
            prev_x, prev_y = position[0], position[1]
            position, velocity, time = method(position, velocity, time, mass, force, delta_t)

            if position[1] <= 0: 
                # interpolate the function to zero
                position[0] = (position[1] * prev_x - prev_y * position[0]) / (position[1] - prev_y) - x_0
                position[1] = 0.0

                print(position)
                return position

def main():
    # Estimate maximum iterations needed for pre-allocation:
    # - Theoretical flight time (ideal case): 2*v*sin(θ)/g (It should be less with air resistance)
    # - Add buffer for margin
    estimated_flight_time = 2 * v_0 * math.sin(theta) / g

    # Ideal study (no friction)
    # Compute range with different time steps
    computed_range_euler = []
    computed_range_rk4 = []

    for delta_t in time_intervals:
        max_iterations = int(estimated_flight_time / delta_t) + 1000

        # Explicit Euler method
        position = np.array([x_0, 0.0])
        velocity = np.array([v_0 * math.cos(theta), v_0 * math.sin(theta)])
        time = 0.0
        final_position = propagate(position, velocity, time, delta_t, projectile_mass, f_ideal, explicit_euler)
        computed_range_euler.append(final_position[0])

        # RK4 method
        position = np.array([x_0, 0.0])
        velocity = np.array([v_0 * math.cos(theta), v_0 * math.sin(theta)])
        time = 0.0
        final_position = propagate(position, velocity, time, delta_t, projectile_mass, f_ideal, runge_kutta)
        computed_range_rk4.append(final_position[0])

    ideal_range = 2 * v_0 ** 2 / g * math.sin(theta) * math.cos(theta)
    print('Ideal range: ', ideal_range, 'm')
    print('Euler ranges:', computed_range_euler)
    print('RK4 ranges:', computed_range_rk4)

    plot.figure()
    plot.plot(time_intervals, np.abs(ideal_range - np.array(computed_range_euler)), ls='-', marker='o', label='Explicit Euler Error')
    plot.plot(time_intervals, np.abs(ideal_range - np.array(computed_range_rk4)), ls='-', marker='s', label='RK4 Error')
    plot.xscale('log')
    plot.yscale('log')
    plot.xlabel('Time Step (s)')
    plot.ylabel('Absolute Error (m)')
    plot.title('Comparison of Numerical Methods Error (Ideal Case)')
    plot.legend()
    plot.grid(True, alpha=0.3)
    plot.show(block=True)

    # --------------------------------------------------------
    # Realistic study (with air resistance)
    dt = 1e-4  # time step for simulation
    max_iterations = int(estimated_flight_time / dt) + 1000

    # Pre-allocate trajectory array
    position = np.zeros((max_iterations, 2))
    position[0] = [x_0, 0.0]

    velocity = np.array([v_0 * math.cos(theta), v_0 * math.sin(theta)])
    time = 0.0
    idx = 0
    
    while True:
        idx += 1
        
        # RK4 
        rkposition, velocity, time = runge_kutta(position[idx-1], velocity, time, projectile_mass, f_drag, dt)
        position[idx] = rkposition

        if rkposition[1] <= 0:
            # interpolate the function to zero
            if rkposition[0] == position[idx-1][0]: 
                position[idx] = np.array([rkposition[0], 0.0])
            else: 
                position[idx] = np.array([(rkposition[1] * position[idx][0] - position[idx][1] * rkposition[0]) / (rkposition[1] - position[idx-1][1]) - x_0, 0.0])
            break
    
    # Trim the array to actual size used
    position = position[:idx+1]
    
    plot.figure()
    plot.plot(position[:, 0], position[:, 1], ls='-', label='Projectile Path with Air Resistance')
    plot.xlim(0, 1e3)
    plot.ylim(0, 7e2)
    plot.xlabel('Distance (m)')
    plot.ylabel('Height (m)')
    plot.title('Projectile Motion with Air Resistance')
    plot.legend()
    plot.grid(True, alpha=0.3)
    plot.show(block=True)

    return

if __name__ == '__main__':
    main()