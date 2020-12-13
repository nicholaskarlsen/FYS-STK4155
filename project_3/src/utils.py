def minibatch(N, M):
    """Splits data set x into M roughly equally minibatches. If not evenly divisible, the excess
    is evenly spread throughout some of the batches.

    Args:
        N (Int): Number of datapoints
        M (Int): Number of minibatches

    Returns:
        Array: [M,.]-dim array containing the minibatch indices
    """
    indices = np.random.permutation(N)  # random permutation of [0, ..., len(x)-1]
    indices = np.array_split(indices, M)  # Split permutation into M sub-arrays
    return indices