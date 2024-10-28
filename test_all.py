import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from vkoga_2L import VKOGA_2L
import tkernels
import kernels

def generate_data(n_samples=500, n_features=2, n_targets=1):
    desired_A = torch.tensor([[0.5, 0.0], [0.0, 2.0]])  # True transformation matrix to calc X and compare later to learned A
    X = torch.randn(n_samples, n_features)
    X_transformed = X @ desired_A
    y = X_transformed[:, 0] + X_transformed[:, 1] # + torch.randn(1000)
    return X, y, desired_A

# Initialize VKOGA_2L model (some Parameters are described in Page 123)
model = VKOGA_2L(
    kernel=[kernels.Gaussian(), tkernels.Gaussian()], # quadratic Matern kernel used
    flag_2L_optimization=True,
    verbose=True,
    greedy_type='f_greedy',
    reg_par=0,
    restr_par=1e-2,
    tol_f=1e-10,
    tol_p=1e-10,
    reg_para_optim=1e-5,
    learning_rate=1e-3,
    n_epochs_optim=1000,
    batch_size=32,
)

# Generate data
X, Y, desired_A = generate_data(n_samples=500, n_features=2, n_targets=1)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the model on training data
print("Training the VKOGA_2L model with OptimizedKernel...")
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Print some predictions and ground truth
predictions = predictions.flatten()
print("Predictions:", predictions[:5])
print("Ground truth:", y_test[:5])

# Print Error
mse = np.mean((predictions - y_test.numpy()) ** 2)
print(f"MSE on Test Data: {mse:.6f}")

print("Learned A:")
print(model.A)
print("Desired A:")
print(desired_A)