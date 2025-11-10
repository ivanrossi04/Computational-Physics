## Double Pendulum

Date: **November 7, 2025**

We want to simulate the motion of the double pendulum. By studying the system considering its Lagrangian, we find the motion equations expressed as a function of the angles, with $\Delta = \theta_1 - \theta_2$:

\(
\ddot\theta_1 \;=\; \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g \sin(\theta_1-2\theta_2) - 2\sin\Delta\; m_2\big(\dot\theta_2^{2}l_2 + \dot\theta_1^{2}l_1\cos\Delta\big)}
{\,l_1\big(2m_1+m_2 - m_2\cos(2\Delta)\big)\,}
\)

\(
\ddot\theta_2 \;=\; \frac{2\sin\Delta\big(\dot\theta_1^{2}l_1(m_1+m_2) + g(m_1+m_2)\cos\theta_1 + \dot\theta_2^{2}l_2 m_2\cos\Delta\big)}
{\,l_2\big(2m_1+m_2 - m_2\cos(2\Delta)\big)\,}
\)

1. Implement a code that propagates the equations via the Runge-Kutta 4 method and verify that the total energy is conserved.
2. Animate the pendulum using the `matplotlib.animation` module, drawing the trajectories of the pendulums, which will be stored every 10 steps and will have a certain predetermined length.

**Optional:**
3. Introduce the possibility to animate different simulations. This way it is possible to verify the chaotic nature of the system by varying the initial conditions by a small amount, expecting big changes in return.
4. Implement a procedure that creates a plot of the coordinates ($\theta_1$ and $\theta_2$) trajectories in time.