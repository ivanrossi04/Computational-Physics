## Range of a Medieval Cannon

Date: **October 31, 2025**

We want to calculate the range of a typical medieval cannon, also considering air resistance. We will evaluate the difference with respect to the ideal case without friction as the launch angle $\theta$ (elevation) and the initial velocity $v_0$ vary.

### Problem Setup

We consider typical spherical projectiles with:
- radius: $r = 5$ cm
- density: $2.7$ g/cm³ (granite)
- initial velocity: between $100$ m/s and $200$ m/s

The friction force due to air has the opposite direction to the velocity and its magnitude is:

$$
F_{\text{att}} = \frac{1}{2} \rho_{\text{air}} C_d A v^2
$$

where:
- $\rho_{\text{air}} = 1.22$ kg/m³ (typical value)
- $C_d = 0.47$ (drag coefficient for a sphere)
- $A = \pi r^2$ (cross-sectional area of the projectile)
- $v$ (projectile velocity)

We recall that in the ideal case without friction, the range is:

$$
d_{\text{ideal}} = \frac{2 v_0^2}{g} \sin(\theta) \cos(\theta)
$$

### Implementation

We will create a program that propagates the motion using the explicit Euler method and the 4th-order Runge-Kutta method. The program will have the ability to cancel the friction force in order to verify correct operation by reproducing the ideal result.

To calculate the range, we will evaluate the pair of time-steps in which the vertical coordinate of the projectile changes sign. Then we will perform a linear interpolation.

**Suggested parameters:** $\theta = 45°$, $v_0 = 100$ m/s, and $\Delta t = 0.001$ s.

It is useful to use numpy arrays of size 4 for the generalized coordinates. The `while` instruction may be useful:

```python
found = False
while (not found):
    y1 = y0 + generalized_force(y0, m, g, Catt) * dt
    if (y0[1] * y1[1] < 0):
        gitt = y0[0] - (y1[0] - y0[0]) / (y1[1] - y0[1]) * y0[1]
        found = True
    y0 = y1
```