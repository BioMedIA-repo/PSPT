import torch

from future.builtins import range
from topk.logarithm import LogTensor


def recursion(S, X, j):


    _P_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 1, -1, -2))
    _N_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 2, -1, -2))

    return _N_, _P_


def approximation(S, X, j, p):


    _P_ = sum(S[j + i] / X ** (i + 1) for i in range(0, p, 2))
    _N_ = sum(S[j + i] / X ** (i + 1) for i in range(1, p, 2))

    return _N_, _P_


def d_logS_d_expX(S, X, j, p, grad, thresh, eps=1e-5):


    _X_ = LogTensor(X)
    _S_ = [LogTensor(S[i]) for i in range(S.size(0))]


    _N_, _P_ = recursion(_S_, _X_, j)


    if not isinstance(_N_, LogTensor):
        _N_ = LogTensor(-1.0 / eps * torch.ones_like(X))
    if not isinstance(_P_, LogTensor):
        _P_ = LogTensor(-1.0 / eps * torch.ones_like(X))

    P, N = _P_.torch(), _N_.torch()


    diff = (P - N) / (N.abs() + eps)


    u_indices = torch.lt(diff, thresh)
    s_indices = u_indices.eq(0)


    grad = grad.type_as(X).resize_as_(X)


    if s_indices.sum():

        _N_ = LogTensor(_N_.torch()[s_indices])
        _P_ = LogTensor(_P_.torch()[s_indices])
        _X_ = LogTensor(X[s_indices])
        _S_ = [LogTensor(S[i][s_indices]) for i in range(S.size(0))]


        _SG_ = (_P_ - _N_) * _X_ / _S_[j]
        grad.masked_scatter_(s_indices, _SG_.torch().exp())


    if u_indices.sum():
        _X_ = LogTensor(X[u_indices])
        _S_ = [LogTensor(S[i][u_indices]) for i in range(S.size(0))]


        _N_, _P_ = approximation(_S_, _X_, j, p)


        _UG_ = (_P_ - _N_) * _X_ / _S_[j]
        grad.masked_scatter_(u_indices, _UG_.torch().exp())

    return grad
