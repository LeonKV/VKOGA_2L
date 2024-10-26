import torch
import numpy as np
from utilities import OptimizedKernel
from tkernels import Matern, Gaussian

# Define the desired transformation matrix (self.A)
desired_A = torch.tensor([[0.5, 0.0], [0.0, 2.0]])

# Generate random data
X = torch.randn(1000, 2)

# Transform X with A
X_transformed = X @ desired_A

# Generate y based on X_transformed and some noise
y = X_transformed[:, 0] + X_transformed[:, 1] # + torch.randn(1000)

# Init the model
model = OptimizedKernel(kernel=Gaussian(), dim=X.shape[1], reg_para=1e-3, learning_rate=1e-3, n_epochs=1000, 
                                                flag_initialize_diagonal=True,
                                                flag_symmetric_A=False)

# Optimize the model (learn A)
model.optimize(X, y)

# Compare the learned A with the desired A
print("Desired A:\n", desired_A.numpy())
print("Learned A:\n", model.A.detach().numpy())