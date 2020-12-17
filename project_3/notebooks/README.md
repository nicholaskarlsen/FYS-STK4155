## Explenation of notebooks

**Heat_Equation_Pytorch.ipynb** - Contains our full analysis pertaining to the solutions of the Heat equation using Neural networks. In addition to the results presented in the report, this notebook also contains identical analysis on the Sigmoid and Tanh activation function as well as experimentation with using random sapling rather than sampling from equispaced grid. Further, plots of the costfunction evaluated at each epoch during the training of the neural networks is included for each model.

**Eigen_torch_minibatches_30pts.ipynb** - Notebook which was used to solve the Eigenvalue problem using a Neural network trained on 30 data points. The notebook is rendered, providing all the results which was included in the final report as well as some additional supplementary material. In particular, the minimal eigenpairs as well as the complete training history of the networks in the form of a graph containing the costfunction evaluated during each epoch during training.

**Eigen_torch_minibatches_30pts.ipynb** - This notebook is an exactc mirror of the one above, but trained on 300 points rather than 30.

**PDE_NN_TEMTEST.ipynb** - Early developmental notebook for solving the heat equation with neural networks. Heavily inspired by code from the week 43 lecture notes, the network is implemented from scratch and the automatic differentiation is done by pure autograd. Not used in actual production, but might be capable of more clearly illustrating what's happening under the hood than the pytorch-based final notebooks. NO RESULTS FROM THIS NOTEBOOK APPEAR IN THE REPORT!

**Eigen_NN_TEMTEST.ipynb** - Early developmental notebook for solving the differential equation from Yi et al. Heavily inspired by code from the week 43 lecture notes, the network is implemented from scratch and the automatic differentiation is done by pure autograd. Not used in actual production, but might be capable of more clearly illustrating what's happening under the hood than the pytorch-based final notebooks. NO RESULTS FROM THIS NOTEBOOK APPEAR IN THE REPORT! HAS NOT BEEN TESTED THOROUGHLY!

**Explicit_heat_eq.ipynb** - Contains the Finite-Differences implementation for the 1-D heat equation. Very straightforward, rather simplistic, and not too exciting.
