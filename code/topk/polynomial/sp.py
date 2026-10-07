import torch
import torch.nn as nn
import torch.autograd as ag

from topk.polynomial.divide_conquer import divide_and_conquer
from topk.polynomial.multiplication import Multiplication
from topk.polynomial.grad import d_logS_d_expX


class LogSumExp(nn.Module):
    def __init__(self, k, p=None, thresh=1e-5):
        super(LogSumExp, self).__init__()
        self.k = k
        self.p = int(1 + 0.2 * k) if p is None else p
        self.mul = Multiplication(self.k + self.p - 1)
        self.thresh = thresh

        self.register_buffer('grad_k', torch.Tensor(0))
        self.register_buffer('grad_km1', torch.Tensor(0))

        self.buffers = (self.grad_km1, self.grad_k)

    def forward(self, x):
        f = LogSumExp_F()
        return f.apply(x, self.k, self.p, self.thresh, self.mul, self.buffers)


class LogSumExp_F(ag.Function):

    @staticmethod
    def forward(self, x, k, p, thresh, mul, buffers):


        self.save_for_backward(x)

        self.k, self.p, self.thresh = k, p, thresh


        self.grad_km1, self.grad_k = buffers


        n_s = x.size(0)
        kp = self.k + self.p - 1

        assert kp <= x.size(1)


        x = x.clone()


        x_summed = x.sum(1)


        x.t_().mul_(-1)


        x = [x, x.clone().fill_(0)]


        log_res = divide_and_conquer(x, kp, mul=mul)


        coeff = log_res + x_summed[None, :]


        coeff = coeff.view(kp + 1, n_s)


        self.saved_coeff = coeff

        return coeff[self.k - 1: self.k + 1]

    @staticmethod
    def backward(self, grad_sk):


        X, = self.saved_tensors
        S = self.saved_coeff


        S = S.unsqueeze(2).expand(S.size(0), X.size(0), X.size(1))


        self.grad_km1 = d_logS_d_expX(S, X, self.k - 1, self.p, self.grad_km1, self.thresh)
        self.grad_k = d_logS_d_expX(S, X, self.k, self.p, self.grad_k, self.thresh)


        grad_x = grad_sk[0, :, None] * self.grad_km1 + grad_sk[1, :, None] * self.grad_k

        return grad_x, None, None, None, None, None


def log_sum_exp(x):


    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))


def log_sum_exp_k_autograd(x, k):

    n_s = x.size(0)

    assert k <= x.size(1)


    x = x.clone()


    x_summed = x.sum(1)


    x.t_().mul_(-1)


    x = [x, x.clone().fill_(0)]


    log_res = divide_and_conquer(x, k, mul=Multiplication(k))


    coeff = log_res + x_summed[None, :]


    coeff = coeff.view(k + 1, n_s)

    return coeff[k - 1: k + 1]
