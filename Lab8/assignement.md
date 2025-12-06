## Study of a Capacitor

**Date:** November 28, 2025

### Problem Description

Consider a parallel-plate capacitor with flat rectangular plates. Assume that the height of the plates is much greater than their length, so we can take a transverse cross-section.
Create a program that calculates the electrostatic potential using the Jacobi algorithm iteratively on the Poisson (Laplace) equation:

\[\nabla^2\phi(x,y) = \frac{\rho(x,y)}{\epsilon_0}\]

Important notes:
- Storing the matrix M̃ in memory is not needed
- Use numpy methods and optionally numba for optimization
- No charge distribution needs to be set
- Use SI units throughout

### Tasks

a) Create a plot of the electrostatic potential
b) Calculate the charge density distribution and create a plot
c) Calculate the linear capacitance density and compare it with that of an ideal capacitor

### Simulation Parameters

- **Simulation cell side:** 1 m
- **Grid points:** 100×100 and 200×200
- **Plate length:** 0.7 m
- **Plate width:** 0.1 m
- **Plate separation:** 0.5 m
- **Plate 1 potential:** +100 V
- **Plate 2 potential:** -100 V
- **Boundary potential:** 0 V