import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from vkoga_2L import VKOGA_2L
from kernels import Matern, Gaussian

def generate_data(n_samples=500, n_features=3, n_targets=2):
    A_true = A_true = np.array([[2, -1, 0.5], [0.5, 1, -0.5], [1, 1, 2]])  # True transformation matrix to calc X and compare later to learned A
    X = np.random.randn(n_samples, n_features)
    X_transformed = X @ A_true
    kernel = Matern(k=2)
    K = kernel.eval(X_transformed, X_transformed)
    # Y = K @ np.random.randn(n_samples, 2)
    Y = np.dot(X, A_true) + 0.0 * np.random.randn(n_samples, n_features)  # Calculating A with some noise
    return X, Y, A_true

# Initialize VKOGA_2L model (some Parameters are described in Page 123)
model = VKOGA_2L(
    kernel=[Matern(k=2), Matern(k=2)], # quadratic Matern kernel used
    flag_2L_optimization=True,
    verbose=True,
    greedy_type='f_greedy',
    reg_par=1e-2,
    restr_par=1e-2,
    tol_f=1e-10,
    tol_p=1e-10,
    reg_para_optim=0,
    learning_rate=5e-3,
    n_epochs_optim=100,
    batch_size=32
)

# Generate data
X, Y, A_true = generate_data(n_samples=500, n_features=3, n_targets=2)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the model on training data
print("Training the VKOGA_2L model with OptimizedKernel...")
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Print the first few predictions and ground truth
print("Predictions:", predictions[:5])
print("Ground truth:", y_test[:5])

# Evaluate performance
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.6f}")

print("learned A:")
print(model.A)
print("true A:")
print(A_true)