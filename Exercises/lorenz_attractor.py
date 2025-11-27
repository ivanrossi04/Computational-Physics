# !V1

'''
Lorenz Attractor simulation with comparison of numerical integration methods.

This module implements the Lorenz attractor system and demonstrates different
numerical integration methods, including the Symplectic (Semi-Implicit) Euler method
for better energy conservation in Hamiltonian-like systems.

The Lorenz system is defined by:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

Standard parameters: σ=10, ρ=28, β=8/3 (chaotic regime)
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz_derivatives(state: np.ndarray, sigma: float = 10.0, 
                       rho: float = 28.0, beta: float = 8.0/3.0) -> np.ndarray:
    """
    Compute the derivatives of the Lorenz system.
    
    Parameters:
        state : np.ndarray
            Current state [x, y, z]
        sigma, rho, beta : float
            Lorenz system parameters
    
    Returns:
        np.ndarray: Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


def explicit_euler(state: np.ndarray, dt: float, 
                   derivatives_func, **kwargs) -> np.ndarray:
    """
    Explicit (Forward) Euler method.
    
    This is the simplest numerical integration method but has poor
    energy conservation properties for Hamiltonian systems.
    
    Parameters:
        state : np.ndarray
            Current state
        dt : float
            Time step
        derivatives_func : callable
            Function that computes derivatives
        **kwargs : additional parameters for derivatives_func
    
    Returns:
        np.ndarray: Updated state after one time step
    """
    return state + dt * derivatives_func(state, **kwargs)


def symplectic_euler(position: np.ndarray, velocity: np.ndarray, 
                     dt: float, acceleration_func) -> tuple:
    """
    Symplectic (Semi-Implicit) Euler method for Hamiltonian systems.
    
    This method updates velocity first, then uses the new velocity to update
    position. This preserves the symplectic structure of Hamiltonian systems,
    leading to better long-term energy conservation.
    
    For a system with q (position) and p (momentum/velocity):
        p_{n+1} = p_n + dt * F(q_n)
        q_{n+1} = q_n + dt * p_{n+1}  (uses NEW velocity!)
    
    Parameters:
        position : np.ndarray
            Current position
        velocity : np.ndarray
            Current velocity
        dt : float
            Time step
        acceleration_func : callable
            Function that computes acceleration from position
    
    Returns:
        tuple: (new_position, new_velocity)
    """
    # Update velocity first using current position
    acceleration = acceleration_func(position)
    new_velocity = velocity + dt * acceleration
    
    # Update position using NEW velocity (this is what makes it symplectic)
    new_position = position + dt * new_velocity
    
    return new_position, new_velocity


def runge_kutta_4(state: np.ndarray, dt: float, 
                  derivatives_func, **kwargs) -> np.ndarray:
    """
    Fourth-order Runge-Kutta method.
    
    More accurate than Euler methods for general systems.
    
    Parameters:
        state : np.ndarray
            Current state
        dt : float
            Time step
        derivatives_func : callable
            Function that computes derivatives
        **kwargs : additional parameters for derivatives_func
    
    Returns:
        np.ndarray: Updated state after one time step
    """
    k1 = derivatives_func(state, **kwargs)
    k2 = derivatives_func(state + dt/2 * k1, **kwargs)
    k3 = derivatives_func(state + dt/2 * k2, **kwargs)
    k4 = derivatives_func(state + dt * k3, **kwargs)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def simulate_lorenz(initial_state: np.ndarray, dt: float, num_steps: int,
                    method: str = 'rk4', **params) -> np.ndarray:
    """
    Simulate the Lorenz system using the specified numerical method.
    
    Parameters:
        initial_state : np.ndarray
            Initial [x, y, z] coordinates
        dt : float
            Time step
        num_steps : int
            Number of simulation steps
        method : str
            'euler' for explicit Euler, 'rk4' for Runge-Kutta 4
        **params : Lorenz system parameters (sigma, rho, beta)
    
    Returns:
        np.ndarray: Trajectory with shape (num_steps + 1, 3)
    """
    trajectory = np.zeros((num_steps + 1, 3))
    trajectory[0] = initial_state
    state = initial_state.copy()
    
    for i in range(num_steps):
        if method == 'euler':
            state = explicit_euler(state, dt, lorenz_derivatives, **params)
        elif method == 'rk4':
            state = runge_kutta_4(state, dt, lorenz_derivatives, **params)
        else:
            raise ValueError(f"Unknown method: {method}")
        trajectory[i + 1] = state
    
    return trajectory


# --- Harmonic Oscillator for demonstrating Symplectic Euler ---
# The symplectic Euler method shines for Hamiltonian systems like the harmonic oscillator

def harmonic_acceleration(position: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """
    Acceleration for a harmonic oscillator: a = -ω²x
    
    Parameters:
        position : np.ndarray
            Current position
        omega : float
            Angular frequency
    
    Returns:
        np.ndarray: Acceleration
    """
    return -omega**2 * position


def compute_harmonic_energy(position: np.ndarray, velocity: np.ndarray, 
                            omega: float = 1.0) -> float:
    """
    Compute total energy of harmonic oscillator: E = 0.5*(v² + ω²x²)
    
    Parameters:
        position, velocity : np.ndarray
            Current state
        omega : float
            Angular frequency
    
    Returns:
        float: Total energy (kinetic + potential)
    """
    kinetic = 0.5 * np.sum(velocity**2)
    potential = 0.5 * omega**2 * np.sum(position**2)
    return kinetic + potential


def simulate_harmonic_explicit_euler(x0: float, v0: float, dt: float, 
                                     num_steps: int, omega: float = 1.0) -> tuple:
    """
    Simulate harmonic oscillator using explicit Euler method.
    
    Parameters:
        x0, v0 : float
            Initial position and velocity
        dt : float
            Time step
        num_steps : int
            Number of steps
        omega : float
            Angular frequency
    
    Returns:
        tuple: (positions, velocities, energies) arrays
    """
    positions = np.zeros(num_steps + 1)
    velocities = np.zeros(num_steps + 1)
    energies = np.zeros(num_steps + 1)
    
    positions[0] = x0
    velocities[0] = v0
    energies[0] = compute_harmonic_energy(np.array([x0]), np.array([v0]), omega)
    
    x, v = x0, v0
    for i in range(num_steps):
        # Explicit Euler: both updated using OLD values
        x_new = x + dt * v
        v_new = v + dt * (-omega**2 * x)
        x, v = x_new, v_new
        
        positions[i + 1] = x
        velocities[i + 1] = v
        energies[i + 1] = compute_harmonic_energy(np.array([x]), np.array([v]), omega)
    
    return positions, velocities, energies


def simulate_harmonic_symplectic_euler(x0: float, v0: float, dt: float, 
                                       num_steps: int, omega: float = 1.0) -> tuple:
    """
    Simulate harmonic oscillator using Symplectic Euler method.
    
    The symplectic method updates velocity first, then position using the
    NEW velocity. This preserves the phase space volume and leads to
    better long-term energy conservation.
    
    Parameters:
        x0, v0 : float
            Initial position and velocity
        dt : float
            Time step
        num_steps : int
            Number of steps
        omega : float
            Angular frequency
    
    Returns:
        tuple: (positions, velocities, energies) arrays
    """
    positions = np.zeros(num_steps + 1)
    velocities = np.zeros(num_steps + 1)
    energies = np.zeros(num_steps + 1)
    
    positions[0] = x0
    velocities[0] = v0
    energies[0] = compute_harmonic_energy(np.array([x0]), np.array([v0]), omega)
    
    x, v = x0, v0
    for i in range(num_steps):
        # Symplectic Euler: update velocity first, then position with NEW velocity
        v_new = v + dt * (-omega**2 * x)  # Update velocity using current position
        x_new = x + dt * v_new            # Update position using NEW velocity
        x, v = x_new, v_new
        
        positions[i + 1] = x
        velocities[i + 1] = v
        energies[i + 1] = compute_harmonic_energy(np.array([x]), np.array([v]), omega)
    
    return positions, velocities, energies


def plot_lorenz_attractor():
    """Generate and plot the Lorenz attractor trajectory."""
    # Simulation parameters
    initial_state = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    num_steps = 10000
    
    # Run simulation with RK4 (more accurate for chaotic systems)
    trajectory = simulate_lorenz(initial_state, dt, num_steps, method='rk4')
    
    # 3D plot
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             lw=0.5, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz Attractor (RK4)')
    
    # Time series
    ax2 = fig.add_subplot(122)
    time = np.arange(num_steps + 1) * dt
    ax2.plot(time, trajectory[:, 0], label='x', alpha=0.7)
    ax2.plot(time, trajectory[:, 1], label='y', alpha=0.7)
    ax2.plot(time, trajectory[:, 2], label='z', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title('Lorenz System Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/lorenz_attractor.png', dpi=150)
    plt.show()


def compare_energy_conservation():
    """
    Compare energy conservation between Explicit Euler and Symplectic Euler
    methods for a harmonic oscillator.
    
    This demonstrates why symplectic methods are preferred for long-term
    simulations of Hamiltonian systems.
    """
    # Simulation parameters
    x0 = 1.0  # Initial position
    v0 = 0.0  # Initial velocity
    omega = 1.0  # Angular frequency
    dt = 0.1  # Time step (relatively large to show energy drift)
    num_steps = 1000
    
    # Simulate with both methods
    x_euler, v_euler, e_euler = simulate_harmonic_explicit_euler(
        x0, v0, dt, num_steps, omega)
    x_sympl, v_sympl, e_sympl = simulate_harmonic_symplectic_euler(
        x0, v0, dt, num_steps, omega)
    
    time = np.arange(num_steps + 1) * dt
    initial_energy = e_euler[0]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Phase space plots
    axes[0, 0].plot(x_euler, v_euler, 'b-', alpha=0.7, label='Explicit Euler')
    axes[0, 0].plot(x_sympl, v_sympl, 'r-', alpha=0.7, label='Symplectic Euler')
    # Add theoretical circle
    theta = np.linspace(0, 2*np.pi, 100)
    axes[0, 0].plot(x0*np.cos(theta), -x0*omega*np.sin(theta), 
                    'g--', alpha=0.5, label='Theoretical')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Velocity')
    axes[0, 0].set_title('Phase Space Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Position vs time
    axes[0, 1].plot(time, x_euler, 'b-', alpha=0.7, label='Explicit Euler')
    axes[0, 1].plot(time, x_sympl, 'r-', alpha=0.7, label='Symplectic Euler')
    axes[0, 1].plot(time, x0*np.cos(omega*time), 'g--', alpha=0.5, label='Theoretical')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].set_title('Position vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy vs time
    axes[1, 0].plot(time, e_euler, 'b-', alpha=0.7, label='Explicit Euler')
    axes[1, 0].plot(time, e_sympl, 'r-', alpha=0.7, label='Symplectic Euler')
    axes[1, 0].axhline(y=initial_energy, color='g', linestyle='--', 
                       alpha=0.5, label='Initial Energy')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Total Energy')
    axes[1, 0].set_title('Energy Conservation Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy error
    axes[1, 1].plot(time, (e_euler - initial_energy) / initial_energy * 100, 
                    'b-', alpha=0.7, label='Explicit Euler')
    axes[1, 1].plot(time, (e_sympl - initial_energy) / initial_energy * 100, 
                    'r-', alpha=0.7, label='Symplectic Euler')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Energy Error (%)')
    axes[1, 1].set_title('Relative Energy Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/energy_conservation_comparison.png', dpi=150)
    plt.show()
    
    # Print summary
    print("\n=== Energy Conservation Summary ===")
    print(f"Initial Energy: {initial_energy:.6f}")
    print(f"\nExplicit Euler:")
    print(f"  Final Energy: {e_euler[-1]:.6f}")
    print(f"  Energy Drift: {(e_euler[-1] - initial_energy) / initial_energy * 100:.2f}%")
    print(f"  (Energy grows unboundedly - unstable for long simulations)")
    print(f"\nSymplectic Euler:")
    print(f"  Final Energy: {e_sympl[-1]:.6f}")
    print(f"  Energy Error: {(e_sympl[-1] - initial_energy) / initial_energy * 100:.2f}%")
    print(f"  (Energy oscillates but stays bounded - stable for long simulations)")


def main():
    """Main function to demonstrate the numerical methods."""
    print("Lorenz Attractor and Symplectic Euler Method Demonstration")
    print("=" * 60)
    
    print("\n1. Generating Lorenz Attractor visualization...")
    plot_lorenz_attractor()
    
    print("\n2. Comparing Euler methods for energy conservation...")
    print("   (Using harmonic oscillator - a true Hamiltonian system)")
    compare_energy_conservation()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("\nKey takeaway: Symplectic Euler preserves the phase space structure")
    print("and provides bounded energy errors, making it ideal for long-term")
    print("simulations of Hamiltonian systems (oscillators, planetary motion, etc.)")


if __name__ == "__main__":
    main()
