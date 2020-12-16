# Project 2: Solving partial differential equations with neural networks 

<ADD LINK TO FINAL REPORT HERE>


## Overview of project structure
- **/figs** Contains all of the figures used in the report + some additional once that went unused
- **/notebooks** Contains the jupyter notebooks that generate all of the results used in the report. This is where most of the code in this project resides
- **/report** Contains the source code used to generate the report pdf
- **/src** Contains a single script with some reused utility functions

## Abstract
We investigate the application of neural networks to the solution of both partial differential equations, exemplified by the 1-D heat equation; as well as systems of nonlinear ordinary differential equations, exemplified by a system of equations describing the dynamics of a class of Recurrent Neural Networks that can be used to find eigenvalues and eigenvectors of a real, symmetric matrix. For the heat equation we compare the solution from a neural network with both the analytic solution and a numerical solution from a simple Finite-Difference scheme for different resolutions of the computational grid and different amounts of training points for the network. We perform a similar analysis for the nonlinear system; this time comparing the networks solution with a numerical integration using the Forward-Euler method. However, as the nonlinear system does not readily admit an analytical solution, only the equilibrium state of the system can be independently computed (through standard numerical eigendecomposition of the mentioned matrix) as a "ground truth" of the solution. In general we find that the neural network-based solution can reach a reasonable level of accuracy with fewer grid-points/training points than the standard numerical integration schemes, and with far less stringent requirements on the distance between grid-points. Notably, the neural network-solution does not need its training grid-points to satisfy the famously restrictive stability criterion for the Finite-Difference method applied to the heat equation. We do find, however, that it is more difficult to scale up the accuracy of the network-based solutions, as compared to more traditional schemes.
