import torch
import numpy as np
from utilities import OptimizedKernel
from tkernels import Matern, Gaussian
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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Init the model
model = OptimizedKernel(kernel=Gaussian(), dim=X.shape[1], reg_para=1e-6, learning_rate=1e-3, n_epochs=1000, 
                                                flag_initialize_diagonal=True,
                                                flag_symmetric_A=False)

# Optimize the model (learn A)
model.optimize(X_train, y_train)

print("Learned A:\n", model.A.detach().numpy())