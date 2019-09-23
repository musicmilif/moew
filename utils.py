import numpy as np
import torch
import lightgbm as lgb


def sample_from_ball(cnt=1, dim=1, radius=2):
    points = np.random.normal(size=(cnt, dim))
    points /= np.expand_dims(np.linalg.norm(points, axis=1), axis=1)
    scales = np.power(np.random.uniform(size=(cnt, 1)), 1 / dim)
    points *= scales * radius
    
    return points


def tensor_to_numpy(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.device.type == 'cuda':
        x = x.to(torch.device('cpu'))
    return x.numpy()


def tensor_to_dmatrix(x, y, w=None):    
    return lgb.Dataset(tensor_to_numpy(x), tensor_to_numpy(y), weight=tensor_to_numpy(w))


def imp_sampling(train_y, valid_y):
    train_prob = np.bincount(train_y.astype(np.int16))
    valid_prob = np.bincount(valid_y.astype(np.int16))
    
    return valid_prob / train_prob


def get_weight(emb, alpha, y, pi, c=1):
    """
    Args:
        emb: np.array. num_train*dim_alpha
        alpha: np.array. dim_alpha*num_parallel_alpha
        y: np.array. num_train
        pi: np.array. num_class
    """
    
    weight = c * np.array([pi[int(y_)] for y_ in y])
    weight = weight[:, np.newaxis] * 1/(1+np.exp(-emb.dot(alpha.T)))
    
    return weight
