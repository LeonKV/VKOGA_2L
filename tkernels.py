#!/usr/bin/env python3

# Torch implementation of the kernels!!
# import sys
# sys.path.append('/usr/local/lib/python3.7/dist-packages/')
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


# Abstract kernel
class Kernel(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def diagonal(self, X):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


# Abstract RBF
class RBF(Kernel):
    @abstractmethod
    def __init__(self):
        super(RBF, self).__init__()

    def eval(self, x, y):
        return self.rbf(self.ep,
                        torch.cdist(x, y))

    def diagonal(self, X):
        return torch.ones(X.shape[0], 1) * self.rbf(self.ep, torch.tensor(0.0))

    def __str__(self):
        return self.name + ' [gamma = %2.2e]' % self.ep

    def set_params(self, par):
        self.ep = par


# Implementation of concrete RBFs
class Gaussian(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'gauss'
        self.rbf = lambda ep, r: torch.exp(-(ep * r) ** 2)


class GaussianTanh(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'gauss_tanh'
        self.rbf = lambda ep, r: torch.exp(-(ep * torch.tanh(r)) ** 2)


class IMQ(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'imq'
        self.rbf = lambda ep, r: 1. / torch.sqrt(1 + (ep * r) ** 2)


class Matern(RBF):
    def __init__(self, ep=1, k=0):
        self.ep = ep
        if k == 0:
            self.name = 'mat0'
            self.rbf = lambda ep, r: torch.exp(-ep * r)
        elif k == 1:
            self.name = 'mat1'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (1 + ep * r)
        elif k == 2:
            self.name = 'mat2'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (3 + 3 * ep * r + (ep * r) ** 2)
        elif k == 3:
            self.name = 'mat3'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (15 + 15 * ep * r + 6 * (ep * r) ** 2 + 1 * (ep * r) ** 3)
        elif k == 4:
            self.name = 'mat4'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        105 + 105 * ep * r + 45 * (ep * r) ** 2 + 10 * (ep * r) ** 3 + 1 * (ep * r) ** 4)
        elif k == 5:
            self.name = 'mat5'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        945 + 945 * ep * r + 420 * (ep * r) ** 2 + 105 * (ep * r) ** 3 + 15 * (ep * r) ** 4 + 1 * (
                            ep * r) ** 5)
        elif k == 6:
            self.name = 'mat6'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        10395 + 10395 * ep * r + 4725 * (ep * r) ** 2 + 1260 * (ep * r) ** 3 + 210 * (
                            ep * r) ** 4 + 21 * (ep * r) ** 5 + 1 * (ep * r) ** 6)
        elif k == 7:
            self.name = 'mat7'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        135135 + 135135 * ep * r + 62370 * (ep * r) ** 2 + 17325 * (ep * r) ** 3 + 3150 * (
                            ep * r) ** 4 + 378 * (ep * r) ** 5 + 28 * (ep * r) ** 6 + 1 * (ep * r) ** 7)
        else:
            self.name = None
            self.rbf = None
            raise Exception('This Matern kernel is not implemented')


class HatFunction(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.rbf = lambda ep, r: torch.clamp(1 - ep*r, min=0)
        self.name = 'Hat function'


class PowerKernels(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.rbf = lambda ep, r: torch.abs(r)
        self.name = 'Power kernel'


class Wendland(RBF):
    def __init__(self, ep=1, k=0, d=1):
        self.ep = ep
        self.name = 'wen_' + str(d) + '_' + str(k)
        l = np.floor(d / 2) + k + 1
        if k == 0:
            p = lambda r: 1
        elif k == 1:
            p = lambda r: (l + 1) * r + 1
        elif k == 2:
            p = lambda r: (l + 3) * (l + 1) * r ** 2 + 3 * (l + 2) * r + 3
        elif k == 3:
            p = lambda r: (l + 5) * (l + 3) * (l + 1) * r ** 3 + (45 + 6 * l * (l + 6)) * r ** 2 + (
                        15 * (l + 3)) * r + 15
        elif k == 4:
            p = lambda r: (l + 7) * (l + 5) * (l + 3) * (l + 1) * r ** 4 + (
                        5 * (l + 4) * (21 + 2 * l * (8 + l))) * r ** 3 + (45 * (14 + l * (l + 8))) * r ** 2 + (
                                      105 * (l + 4)) * r + 105
        else:
            raise Exception('This Wendland kernel is not implemented')
        c = np.math.factorial(l + 2 * k) / np.math.factorial(l)
        e = l + k
        # self.rbf = lambda ep, r: np.maximum(1 - ep * r, 0) ** e * p(ep * r) / c
        self.rbf = lambda ep, r: torch.clamp(1 - ep * r, min=0) ** e * p(ep * r) / c


#  # Polynomial kernels
# class Polynomial(Kernel):
#     def __init__(self, a=0, p=1):
#         self.a = a
#         self.p = p
#
#     def eval(self, x, y):
#         return (np.atleast_2d(x) @ np.atleast_2d(y).transpose() + self.a) ** self.p
#
#     def diagonal(self, X):
#         return ((np.linalg.norm(X, axis=1) + self.a) ** self.p)[:, None]
#
#     def __str__(self):
#      return 'polynomial' + ' [a = %2.2e, p = %2.2e]' % (self.a, self.p)
#
#     def set_params(self, par):
#         self.a = par[0]
#         self.p = par[1]

# A demo usage
def main():
    ker = Matern(k=2)

    x = torch.linspace(-5, 5, 10001)[:, None]
    y = torch.zeros(1, 1)
    A = ker.eval(x, y)
    B = torch.exp(-x ** 2)

    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()
    ax.plot(x, A)
    ax.plot(x, B)
    ax.set_title('A kernel: ' + str(ker))
    plt.show()


if __name__ == '__main__':
    main()
