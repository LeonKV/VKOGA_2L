import math
import torch
import numpy as np

from torch import nn
from tkernels import Kernel


class OptimizedKernel(torch.nn.Module):

    def __init__(self, kernel, dim,
                 reg_para=1e-5, learning_rate=1e-3, n_epochs=100, batch_size=32, n_folds=None,
                 flag_initialize_diagonal=False, flag_symmetric_A=False):
        super().__init__()

        # Some settings, mostly optimization related
        self.kernel = kernel

        self.dim = dim
        self.reg_para = reg_para
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.flag_symmetric_A = flag_symmetric_A

        # Define linear maps - hardcoded
        if torch.is_tensor(flag_initialize_diagonal):
            self.B = nn.Parameter(flag_initialize_diagonal, requires_grad=True)
        elif flag_initialize_diagonal:
            self.B = nn.Parameter(torch.eye(self.dim, self.dim), requires_grad=True)
        else:
            self.B = nn.Parameter(torch.rand(self.dim, self.dim), requires_grad=True)

        if self.flag_symmetric_A:
            self.A = (self.B + self.B.t()) / 2
        else:
            self.A = self.B


        if n_folds is None:
            self.n_folds = self.batch_size
        else:
            self.n_folds = n_folds


        # Set optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=.7)

        # Initliaze lists from tracking
        self.list_obj = []
        self.list_parameters = []

    def kernel_eval(self, X, Z):
        self.rbf = lambda ep, r: torch.exp(-ep * r) * (3 + 3 * ep * r + 1 * (ep * r) ** 2) # Matern(k=2) kernel
        return self.rbf(1,torch.cdist(X, Z))

    def optimize(self, X, y, flag_optim_verbose=True):

        assert X.shape[0] == y.shape[0], 'Data sizes do not match'
        n_batches = X.shape[0] // self.batch_size

        # Append initial parameters
        if self.flag_symmetric_A:
            self.list_parameters.append(torch.clone((self.B + self.B.t()) / 2).detach().numpy())
        else:
            self.list_parameters.append(torch.clone(self.A).detach().numpy())


        for idx_epoch in range(self.n_epochs):
            shuffle = np.random.permutation(X.shape[0])  # reshuffle the data set every epoch

            list_obj_loc = []

            for idx_batch in range(n_batches):

                # Select minibatch from the data
                ind = shuffle[idx_batch * self.batch_size : (idx_batch + 1) * self.batch_size]
                if len(y.shape) == 1:
                    Xb, yb = X[ind, :], y[ind]
                else:
                    Xb, yb = X[ind, :], y[ind, :]

                # Compute kernel matrix for minibatch
                kernel_matrix = self.kernel.eval(Xb @ self.A, Xb @ self.A)
                # kernel_matrix = self.kernel.eval(Xb @ torch.clone(self.A).detach().numpy(), Xb @ torch.clone(self.A).detach().numpy())
                # kernel_matrix = self.kernel_eval(Xb @ self.A, Xb @ self.A) # To avoid detach on self.A

                # use cross validation loss via rippa to assess the error
                optimization_objective, _ = compute_cv_loss_via_rippa_ext_2(
                    kernel_matrix, yb, self.n_folds, self.reg_para)

                # Keep track of optimization quantity within epoch
                list_obj_loc.append(optimization_objective.detach().item())
                if idx_epoch == 0 and flag_optim_verbose:
                    print('First epoch: Iteration {:5d}: Training objective: {:.3e}'.format(
                        idx_batch, optimization_objective.detach().item()))

                # Do optimization stuff
                optimization_objective.backward()
                self.optimizer.step()  # do one optimization step
                self.optimizer.zero_grad()  # set gradients to zero

                if self.flag_symmetric_A:
                    self.A = (self.B + self.B.t()) / 2
                else:
                    self.A = self.B

            # Keep track of some quantities and print something
            mean_obj = np.mean(list_obj_loc)

            if flag_optim_verbose:
                print('Epoch {:5d} finished, mean training objective: {:.3e}.'.format(
                    idx_epoch + 1, mean_obj))
                if idx_epoch % 100 == 0:
                    print("Learned A:\n", self.A)

            self.list_obj.append(mean_obj)

            self.list_parameters.append(torch.clone(self.A).detach().numpy())


def compute_cv_loss_via_rippa_ext_2(kernel_matrix, y, n_folds, reg_for_matrix_inversion):
    """
    Implementation without the need to provide a kernel and points: Simply provide the kernel matrix
    """

    # Some precomputations
    kernel_matrix_reg = kernel_matrix + reg_for_matrix_inversion * torch.eye(kernel_matrix.shape[0])
    inv_kernel_matrix = torch.inverse(kernel_matrix_reg)
    coeffs = torch.linalg.solve(kernel_matrix_reg, y) #[0]

    # Some initializations and preparations: It is required that n_folds divides y.shape[0] without remainder
    array_error = torch.zeros(y.shape[0], 1)
    n_per_fold = int(y.shape[0] / n_folds)
    indices = torch.arange(0, y.shape[0]).view(n_per_fold, n_folds)

    # Standard Rippa's scheme
    if n_folds == y.shape[0]:
        array_error = coeffs / torch.diag(inv_kernel_matrix).view(-1,1)

    # Extended Rippa's scheme
    else:
        for j in range(n_folds):
            inv_kernel_matrix_loc1 = inv_kernel_matrix[indices[:, j], :]
            inv_kernel_matrix_loc = inv_kernel_matrix_loc1[:, indices[:, j]]

            array_error[j * n_per_fold: (j+1) * n_per_fold, 0] = \
                (torch.linalg.solve(inv_kernel_matrix_loc, coeffs[indices[:, j]])).view(-1)

    cv_error_sq = torch.sum(array_error ** 2) / array_error.numel()

    return cv_error_sq, array_error

