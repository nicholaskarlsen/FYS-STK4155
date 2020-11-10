import numpy as np

def logreg(X,Y,theta):
# This is the skeleton for the computations
# X.shape = [examples,predictors]
# Y.shape = [examples, classes]
# theta.shape = [predictors,classes]

    prob = np.exp(X @ theta)/np.sum(np.exp(X @ theta)),axis=1)
    # prob.shape = [examples,classes], probabilites
    cost_function = -np.sum(Y * np.ln(prob) + (1-Y)*np.ln(prob))
    # scalar, from the double-sum over examples and classes, to only punish
    # missing correct labels, remove everything after the last plus sign.
    weight = prob * (Y - (1-Y)/(1-prob))
    # helper matrix for vectorizing the computation of the cost_gradients
    # shape = [examples,classes]. Punishes probailities for mislabeling,
    # scary wrt numerics. In case numerical errors arise, remove the stuff
    # after the first minus sign, then only missing correct labels is punished.
    cost_gradients = X.T @ (prob * np.sum(weigths,axis=1)) - X.T @ weights
    # Derivatives of cost wrt thetas, shape = [predictors,classes]

return 
