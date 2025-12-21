## Simulation of Sound Waves in a 2D Plane

**Date:** December 12, 2025

Consider the pressure wave equation in a two-dimensional space:

$$\frac{\partial^2}{\partial t^2} p(\mathbf{r},t) = c^2\nabla^2_{\mathbf{r}}p(\mathbf{r},t) $$

where $p$ is the pressure variation, $c = 300 \, \text{m/s}$ is the speed of sound. Consider a 2-dimensional simulation cell that reproduces two rooms with an opening between them.

On the walls, the Neumann boundary condition applies:
 
$$\nabla_{\mathbf{r}} p(\mathbf{r},t) \cdot \mathbf{n} = 0$$
 
 where $\mathbf{n}$ is the vector normal to the surface. At the initial time $t_0$, we will consider a Gaussian pulse centered at $(x_0,y_0)$ with amplitude $A$ and width $\sigma$:

$$p((x,y),t) = \frac{A}{2\pi\sigma}e^{-\frac{(x-x_0)^2+(y-y_0)^2}{2\sigma^2}}$$

Furthermore, we consider the temporal derivative of $p$ at time $t_0$ to be zero:

$$\frac{\partial}{\partial t}p(\mathbf{r},t_0) = 0$$

Create a program that calculates the function $p(\mathbf{r},t)$ using the finite difference method and produces an animation with matplotlib.

### Simulation parameters
- **Length of the simulation cell:** $L = 10$ m
- **Length of the first room:** $D_p = 5$ m
- **Width of the barrier:** $L_b = 0.5$ m
- **Length of the hole:** $S = 0.5$ m
- **Initial pulse position:** $x_0 = 2$ m, $y_0 = 2$ m
- **Pulse amplitude:** $A = 1$
- **Pulse width:** $\sigma = 0.5$ m
- **Spatial grid:** $N \times N$ points with $N = 100$
- **Time step:** $\Delta t = 10^{-4}$ s