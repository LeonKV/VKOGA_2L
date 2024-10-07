import torch
import torch.nn as nn
import torch.optim as optim

class OptimizedKernel:
    def __init__(self, kernel, dim, reg_para=1e-3, learning_rate=5e-3, n_epochs=10, batch_size=32, 
                 n_folds=None, flag_initialize_diagonal=True, flag_symmetric_A=False):
        self.kernel = kernel
        self.dim = dim
        self.reg_para = reg_para
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.flag_initialize_diagonal = flag_initialize_diagonal
        self.flag_symmetric_A = flag_symmetric_A

        # Initialize the matrix A (dim, dim), requires_grad=True to track and learn the params in A
        if self.flag_initialize_diagonal:
            self.A = torch.eye(self.dim, requires_grad=True)  # Diagonal initialization
        else:
            self.A = torch.randn(self.dim, self.dim, requires_grad=True)  # Random initialization

        if self.flag_symmetric_A:
            self.A = (self.A + self.A.T) / 2  # symetric A?

    def eval(self, X, Z):
        # k(Ax, Az)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Z_tensor = torch.tensor(Z, dtype=torch.float32)
        X_transformed = X_tensor @ self.A
        Z_transformed = Z_tensor @ self.A

        return self.kernel_eval(X_transformed, Z_transformed)
    
    def kernel_eval(self, X, Z):
        self.rbf = lambda ep, r: torch.exp(-ep * r) * (3 + 3 * ep * r + 1 * (ep * r) ** 2) # Matern(k=2) kernel
        return self.rbf(1,torch.cdist(X, Z))

    def optimize(self, X, y, flag_optim_verbose=True):
        # Convert data to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Adam Optimizer (Paper page 123)
        optimizer = optim.Adam([self.A], lr=self.learning_rate)

        # Loop through every Batch in every Epoch
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            if epoch % 20 == 0:
                print(self.A)
            for i in range(0, X.shape[0], self.batch_size):
                # One Batch
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                optimizer.zero_grad()

                K = self.eval(X_batch, X_batch) # Distance betweeen every point dim(K)=(batch_size, batch_size)
                if i==0 and epoch == 0:
                    print(K.shape)
                    print(K)
                    pass
                # K_tensor = torch.tensor(K, dtype=torch.float32)
                y_pred = K @ y_batch

                # high reg_para leads to low values in A
                loss = nn.MSELoss()(y_pred, y_batch) + self.reg_para * torch.norm(self.A)
                
                loss.backward()

                # Update parameters (matrix A)
                optimizer.step()

                # total loss over every epoch
                epoch_loss += loss.item()

            if flag_optim_verbose:
                if epoch % 20 == 0:
                    print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.6f}')

        # detach A so its no longer a tensor but a Numpy array again
        #self.A = self.A.detach()

        return self
