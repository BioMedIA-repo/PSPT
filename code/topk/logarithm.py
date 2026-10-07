import torch
import torch.autograd as ag

from numbers import Number


def log(x, like):


    if isinstance(x, LogTensor):
        return x.torch()

    if not isinstance(x, Number):
        raise TypeError('Not supported type: received {}, '
                        'was expected LogTensor or Number'
                        .format(type(x)))


    like_is_var = isinstance(like, ag.Variable)
    data = like.data if like_is_var else like
    new = data.new(1).fill_(x).log_().expand_as(data)
    new = ag.Variable(new) if like_is_var else new
    return new


def _imul_inplace(x1, x2):
    return x1.add_(x2)


def _imul_outofplace(x1, x2):
    return x1 + x2


def _add_inplace(x1, x2):
    M = torch.max(x1, x2)
    M.add_(((x1 - M).exp_().add_((x2 - M).exp_())).log_())
    return M


def _add_outofplace(x1, x2):
    M = torch.max(x1, x2)
    return M + torch.log(torch.exp(x1 - M) + torch.exp(x2 - M))


class LogTensor(object):


    def __init__(self, x):
        super(LogTensor, self).__init__()

        self.var = isinstance(x, ag.Variable)
        self._x = x
        self.add = _add_outofplace if self.var else _add_inplace
        self.imul = _imul_outofplace if self.var else _imul_inplace

    def __add__(self, other):
        other_x = log(other, like=self._x)
        return LogTensor(self.add(self._x, other_x))

    def __imul__(self, other):
        other_x = log(other, like=self._x)
        self._x = self.imul(self._x, other_x)
        return self

    def __iadd__(self, other):
        other_x = log(other, like=self._x)
        self._x = self.add(self._x, other_x)
        return self

    def __radd__(self, other):


        return self.__add__(other)

    def __sub__(self, other):


        other_x = log(other, like=self._x)
        diff = other_x - self._x
        x = self._x + log1mexp(diff)
        return LogTensor(x)

    def __pow__(self, power):
        return LogTensor(self._x * power)

    def __mul__(self, other):
        other_x = log(other, like=self._x)
        x = self._x + other_x
        return LogTensor(x)

    def __rmul__(self, other):


        return self.__mul__(other)

    def __div__(self, other):


        other_x = log(other, like=self._x)
        x = self._x - other_x
        return LogTensor(x)

    def __truediv__(self, other):


        return self.__div__(other)

    def torch(self):


        return self._x

    def __repr__(self):
        tensor = self._x.data if self.var else self._x
        s = 'Log Tensor with value:\n{}'.format(tensor)
        return s


def log1mexp(U, eps=1e-3):


    res = torch.log1p(-torch.exp(U))


    small = torch.lt(U.abs(), eps)
    res[small] = torch.log(-U[small])

    return res
