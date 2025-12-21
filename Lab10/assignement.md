## Monte Carlo Integration

**Date:** December 19, 2025


We want to write a program to calculate the volume $V_M$ of the unit sphere in M-dimensional space. This value can be written as:

$$V_M = \int dr \, f(\mathbf{r}) \quad \mathbf{r} \in \mathbb{R}^M$$

where:

$$f(\mathbf{r}) = \begin{cases} 1 & \mathbf{r} \cdot \mathbf{r} < 1 \\ 0 & \text{otherwise} \end{cases}$$

We will verify that:


$$V_M = \frac{\pi^{M/2}}{\Gamma(M/2 + 1)}$$

The program must also calculate the expected uncertainty on the result.