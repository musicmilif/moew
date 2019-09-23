import numpy as np
from sklearn.model_selection import train_test_split


def generate_y(X, dist, threshold=.08):
    if len(X) == 0 or len(X[0]) == 0:
        return None
    
    size, ndim = len(X), len(X[0])
    if dist == 'beta':
        prob1 = np.sum(X, axis=1) > ndim/2
        prob2 = np.sum(X, axis=1) / ndim
        prob2 = (0.5 - np.abs(0.5 - prob2))*threshold*4
        prob2 = np.random.binomial(1, prob2, size)
    elif dist == 'norm':
        tot_mean, tot_std = X.mean(), X.std()
        prob1 = np.sqrt(np.linalg.norm((X - tot_mean), axis=1)**2/X.shape[1]) > tot_std
        prob2 = np.random.rand(size) < threshold
    y = np.logical_xor(prob1, prob2)

    return np.array(y, dtype=np.int8)


def generate_data(ndata, ratio=(.1, .3), dist='beta', ndim=2):
    shift_time = 10

    X = np.array([]).reshape(0, ndim)
    if dist == 'beta':
        alpha, beta = 5, 1
        diff = alpha - beta
        for _ in range(shift_time-1):
            tmp_X = np.random.beta(alpha, diff, size=ndata//shift_time*ndim).reshape((-1, ndim))
            X = np.concatenate([X, tmp_X])
            alpha -= diff/(shift_time-1)
            beta += diff/(shift_time-1)
    elif dist == 'norm':
        mean, sigma = 0, 5
        for _ in range(shift_time):
            tmp_X = np.random.normal(mean, sigma, size=ndata//shift_time*ndim).reshape((-1, ndim))
            X = np.concatenate([X, tmp_X])
            mean += 1
            sigma += .1
    else:
        raise NotImplementedError()
        
    y = generate_y(X, dist)
    train_X, test_X, train_y, test_y =  train_test_split(X, y, test_size=ratio[1], shuffle=False)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=ratio[0], shuffle=False)

    return (train_X, valid_X, test_X), (train_y, valid_y, test_y)

