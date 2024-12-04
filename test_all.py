import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from vkoga_2L import VKOGA_2L
import tkernels
import kernels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# Fetch dataset
parkinsons_telemonitoring = fetch_ucirepo(id=189)
X = parkinsons_telemonitoring.data.features
y = parkinsons_telemonitoring.data.targets

# Preprocess Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=None)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

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

# Fit the model on training data
print("Training the VKOGA_2L model with OptimizedKernel...")
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Print some predictions and ground truth
print("Predictions:", predictions[:5])
print("Ground truth:", y_test.numpy()[:5])
print(predictions.shape)
print(y_test.numpy().shape)

# predictions = predictions.flatten()
# Print Error
mse = np.mean((predictions - y_test.numpy()) ** 2)
print(f"MSE on Test Data: {mse:.6f}")
