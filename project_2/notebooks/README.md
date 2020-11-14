## Explenation of notebooks

**LeakyRELU_production.ipynb** - The notebook which was used to produce the best hand-tuned model for the terrain data using a neural network with the Leaky ReLU activation function.
**Sigmoid_production.ipynb** - Same as above, but with Sigmoid as the activation function.
**RELU_production_bestFFNN.ipynb ** - Same as above, but with ReLU as the activation function. This was the best overall model that we included in the report, the rest being ommited (and instead presented in the above two notebooks)

**Logreg.ipynb** - Notebook used to produce a model for classifying the MNIST dataset using logistic regression with SGD.

**MNIST.ipynb** - Contains the code which produces the main results in our report regarding the Neural network classification of the MNIST dataset. Also contains code used to benchmark our results against Sci-kit learns MLPClassifier at the very bottom.

**SGD_test.ipynb** - Notebook containing the code used to produce the results included in the report pertaining to stochastic gradient descent

**Terrain_grid_search.ipynb ** - Notebook which performs a cross-validated grid search of parameters for a the Neural network modeling terrain data using Sigmoid, ReLU and LeakyReLU. However, nothing of interest was found due to the rather computationally expensive nature of this approach limiting the runtime.
