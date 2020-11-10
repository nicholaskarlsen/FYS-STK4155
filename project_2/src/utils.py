def CrossValidation(X, y, score_func,k):
    X_cv, y_cv = split_dataset(X, y)

    score = 0
    for i in range(k):
        X_training = ...
        y_training = ...
        X_test = ...
        y_test =...
        score += score_func(X_training, y_training, X_test, y_test)

    score /= k

    return score
