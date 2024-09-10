import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Generate AR(3) time series data
n = 100
e = np.random.normal(0, 1, n)  # White noise
y = np.zeros(n)

# AR(3) coefficients
phi_1, phi_2, phi_3 = 0.6, -0.3, 0.2

# Create AR(3) process
for t in range(3, n):
    y[t] = phi_1 * y[t-1] + phi_2 * y[t-2] + phi_3 * y[t-3] + e[t]

# Lagged variables for AR(3) model
X_full = np.column_stack([y[2:-1], y[1:-2], y[0:-3]])
y_full = y[3:]

# Create a pandas DataFrame for easy viewing
df = pd.DataFrame({'Lag1': X_full[:, 0], 'Lag2': X_full[:, 1], 'Lag3': X_full[:, 2], 'y': y_full})

# Display the first few rows of the DataFrame
print(df.head())

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(y, label='Generated AR(3) Time Series')
plt.title('AR(3) Generated Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Sherman-Morrison update function
def sherman_morrison_update(A_inv, u, v):
    u = u.reshape(-1, 1)  # Ensure column vector
    v = v.reshape(1, -1)  # Ensure row vector
    A_inv_u = np.dot(A_inv, u)
    v_T_A_inv = np.dot(v, A_inv)
    return A_inv - np.outer(A_inv_u, v_T_A_inv) / (1 + np.dot(v, A_inv_u))

# Split data into training and validation
X_train = X_full[-35:-5]  # From 35th to 5th most recent data points for training
y_train = y_full[-35:-5]
X_valid = X_full[-5:]  # Most recent 5 data points for validation
y_valid = y_full[-5:]

# Fit the initial model on the training data
model = LinearRegression().fit(X_train, y_train)

# Calculate validation error on the 5 most recent data points
y_pred_valid = model.predict(X_valid)
validation_error = np.mean((y_valid - y_pred_valid)**2)
print(f"Initial validation error with 35th to 5th points: {validation_error}")

# Get the inverse of (X_train.T * X_train)
A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

# Start dynamic updating by adding one older data point at a time
best_error = validation_error
best_step = 0

for step in range(10, 0, -1):  # Adding older data points one at a time
    X_new = X_full[-35 - step].reshape(1, -1)  # Add older data point as row
    y_new = np.array([y_full[-35 - step]])

    u = X_new.T
    v = X_new

    A_inv_updated = sherman_morrison_update(A_inv, u, v)

    # Update model coefficients without affecting the intercept
    new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
    model.coef_ = new_coef

    # Update the intercept separately
    model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

    # Predict on the validation set
    y_pred_valid_new = model.predict(X_valid)
    new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
    print(f"Step {step}: Validation error after adding older data: {new_validation_error}")

    # Check if the new error is better
    if new_validation_error < best_error:
        best_error = new_validation_error
        best_step = step
        A_inv = A_inv_updated
    else:
        print(f"Stopping early as validation error increased at step {step}.")
        break

print(f"Best model found at step {best_step} with validation error: {best_error}")

#########seasonal#############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Generate AR(3) time series data with a seasonal effect
n = 100
e = np.random.normal(0, 1, n)  # White noise
y = np.zeros(n)

# AR(3) coefficients
phi_1, phi_2, phi_3 = 0.6, -0.3, 0.2

# Generate a seasonal effect (e.g., monthly cycle with 12 periods)
months = np.arange(n) % 12  # 12-month cycle
seasonal_effect = 2 * np.sin(2 * np.pi * months / 12)  # Sinusoidal seasonal effect

# Create AR(3) process with seasonal effect
for t in range(3, n):
    y[t] = phi_1 * y[t-1] + phi_2 * y[t-2] + phi_3 * y[t-3] + seasonal_effect[t] + e[t]

# Lagged variables for AR(3) model (including seasonal effect)
X_full = np.column_stack([y[2:-1], y[1:-2], y[0:-3], seasonal_effect[2:-1]])
y_full = y[3:]

# Create a pandas DataFrame for easy viewing
df = pd.DataFrame({'Lag1': X_full[:, 0], 'Lag2': X_full[:, 1], 'Lag3': X_full[:, 2], 'Seasonal': X_full[:, 3], 'y': y_full})

# Display the first few rows of the DataFrame
print(df.head())

# Plot the time series with seasonal effect
plt.figure(figsize=(10, 6))
plt.plot(y, label='Generated AR(3) Time Series with Seasonal Effect')
plt.title('AR(3) Generated Time Series with Seasonal Effect')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Sherman-Morrison update function
def sherman_morrison_update(A_inv, u, v):
    u = u.reshape(-1, 1)  # Ensure column vector
    v = v.reshape(1, -1)  # Ensure row vector
    A_inv_u = np.dot(A_inv, u)
    v_T_A_inv = np.dot(v, A_inv)
    return A_inv - np.outer(A_inv_u, v_T_A_inv) / (1 + np.dot(v, A_inv_u))

# Split data into training and validation
X_train = X_full[-35:-5]  # From 35th to 5th most recent data points for training
y_train = y_full[-35:-5]
X_valid = X_full[-5:]  # Most recent 5 data points for validation
y_valid = y_full[-5:]

# Fit the initial model on the training data (now includes seasonal effect)
model = LinearRegression().fit(X_train, y_train)

# Calculate validation error on the 5 most recent data points
y_pred_valid = model.predict(X_valid)
validation_error = np.mean((y_valid - y_pred_valid)**2)
print(f"Initial validation error with 35th to 5th points: {validation_error}")

# Get the inverse of (X_train.T * X_train)
A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

# Start dynamic updating by adding one older data point at a time
best_error = validation_error
best_step = 0

for step in range(10, 0, -1):  # Adding older data points one at a time
    X_new = X_full[-35 - step].reshape(1, -1)  # Add older data point as row (includes seasonal effect)
    y_new = np.array([y_full[-35 - step]])

    u = X_new.T
    v = X_new

    A_inv_updated = sherman_morrison_update(A_inv, u, v)

    # Update model coefficients without affecting the intercept
    new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
    model.coef_ = new_coef

    # Update the intercept separately
    model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

    # Predict on the validation set
    y_pred_valid_new = model.predict(X_valid)
    new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
    print(f"Step {step}: Validation error after adding older data: {new_validation_error}")

    # Check if the new error is better
    if new_validation_error < best_error:
        best_error = new_validation_error
        best_step = step
        A_inv = A_inv_updated
    else:
        print(f"Stopping early as validation error increased at step {step}.")
        break

print(f"Best model found at step {best_step} with validation error: {best_error}")


###############
# Preprocessing

# Let's load the uploaded CSV file and inspect the contents.
import pandas as pd

# Load the CSV file
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)


# Display the first few rows of the data
data.head()

# Display the first few rows of the data

         DATE  IPG2211A2N
0  1985-01-31     72.5052
1  1985-02-28     70.6720
2  1985-03-31     62.4502
3  1985-04-30     57.4714
4  1985-05-31     55.3151


data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as index

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(data['IPG2211A2N'], label='Electric Production')
plt.title('Electric Production Time Series')
plt.xlabel('Date')
plt.ylabel('Production')
plt.legend()
plt.show()

# Extract the target variable (IPG2211A2N)
y = data['IPG2211A2N'].values
n = len(y)

# Define the seasonal effect (monthly seasonality)
months = data.index.month
seasonal_effect = 2 * np.sin(2 * np.pi * months / 12)

# Lagged variables for the TM forecasting model (including seasonal effect)
X_full = np.column_stack([y[2:-1], y[1:-2], y[0:-3], seasonal_effect[2:-1]])
y_full = y[3:]

# Split data into training and validation
X_train = X_full[-35:-5]  # From 35th to 5th most recent data points for training
y_train = y_full[-35:-5]
X_valid = X_full[-5:]  # Most recent 5 data points for validation
y_valid = y_full[-5:]

# Fit the initial model on the training data (now includes seasonal effect)
model = LinearRegression().fit(X_train, y_train)

# Calculate validation error on the 5 most recent data points
y_pred_valid = model.predict(X_valid)
validation_error = np.mean((y_valid - y_pred_valid)**2)
print(f"Initial validation error with 35th to 5th points: {validation_error}")

# Get the inverse of (X_train.T * X_train)
A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

# Start dynamic updating by adding one older data point at a time
best_error = validation_error
best_step = 0

for step in range(10, 0, -1):  # Adding older data points one at a time
    X_new = X_full[-35 - step].reshape(1, -1)  # Add older data point as row (includes seasonal effect)
    y_new = np.array([y_full[-35 - step]])

    u = X_new.T
    v = X_new

    A_inv_updated = sherman_morrison_update(A_inv, u, v)

    # Update model coefficients without affecting the intercept
    new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
    model.coef_ = new_coef

    # Update the intercept separately
    model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

    # Predict on the validation set
    y_pred_valid_new = model.predict(X_valid)
    new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
    print(f"Step {step}: Validation error after adding older data: {new_validation_error}")

    # Check if the new error is better
    if new_validation_error < best_error:
        best_error = new_validation_error
        best_step = step
        A_inv = A_inv_updated
    else:
        print(f"Stopping early as validation error increased at step {step}.")
        break

print(f"Best model found at step {best_step} with validation error: {best_error}")

###early stop#############
# Define early stopping rounds (the number of steps with increasing error before stopping)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the CSV file
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as index

# Extract the target variable (IPG2211A2N)
y = data['IPG2211A2N'].values
n = len(y)

# Define a more refined seasonal effect (monthly seasonality with sine and cosine)
months = data.index.month
seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
seasonal_effect_cos = np.cos(2 * np.pi * months / 12)

# Lagged variables for the TM forecasting model (including seasonal effect)
# Expand to use 5 lags for better long-term memory
X_full = np.column_stack([y[4:-1], y[3:-2], y[2:-3], y[1:-4], y[0:-5], seasonal_effect_sin[4:-1], seasonal_effect_cos[4:-1]])
y_full = y[5:]

# Split data into training and validation (expand training window)
X_train = X_full[-50:-5]  # Expand training window to last 50 data points excluding the last 5 for validation
y_train = y_full[-50:-5]
X_valid = X_full[-5:]  # Most recent 5 data points for validation
y_valid = y_full[-5:]

# Fit the initial model on the training data (now includes seasonal effect)
model = LinearRegression().fit(X_train, y_train)

# Calculate validation error on the 5 most recent data points
y_pred_valid = model.predict(X_valid)
validation_error = np.mean((y_valid - y_pred_valid)**2)
print(f"Initial validation error with 50th to 5th points: {validation_error}")

# Get the inverse of (X_train.T * X_train)
A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

# Sherman-Morrison update function
def sherman_morrison_update(A_inv, u, v):
    u = u.reshape(-1, 1)  # Ensure column vector
    v = v.reshape(1, -1)  # Ensure row vector
    A_inv_u = np.dot(A_inv, u)
    v_T_A_inv = np.dot(v, A_inv)
    return A_inv - np.outer(A_inv_u, v_T_A_inv) / (1 + np.dot(v, A_inv_u))

# Early stopping setup
early_stopping_rounds = 3
worse_count = 0

# Start dynamic updating by adding one older data point at a time
best_error = validation_error
best_step = 0

for step in range(10, 0, -1):  # Adding older data points one at a time
    X_new = X_full[-50 - step].reshape(1, -1)  # Add older data point as row (includes seasonal effect)
    y_new = np.array([y_full[-50 - step]])

    u = X_new.T
    v = X_new

    A_inv_updated = sherman_morrison_update(A_inv, u, v)

    # Update model coefficients without affecting the intercept
    new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
    model.coef_ = new_coef

    # Update the intercept separately
    model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

    # Predict on the validation set
    y_pred_valid_new = model.predict(X_valid)
    new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
    print(f"Step {step}: Validation error after adding older data: {new_validation_error}")

    # Check if the new error is better
    if new_validation_error < best_error:
        best_error = new_validation_error
        best_step = step
        A_inv = A_inv_updated
        worse_count = 0  # Reset the worse count since error has improved
    else:
        worse_count += 1  # Increase the count for worse error
        if worse_count >= early_stopping_rounds:
            print(f"Stopping early due to consecutive worse validation errors at step {step}.")
            break

print(f"Best model found at step {best_step} with validation error: {best_error}")

# Visualization
plt.plot(y_valid, label="Actual")
plt.plot(y_pred_valid, label="Predicted")
plt.title("Validation Results")
plt.xlabel("Time")
plt.ylabel("Electricity Production")
plt.legend()
plt.show()

##########yule with quadratic terms############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the CSV file
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as index

# Extract the target variable (IPG2211A2N)
y = data['IPG2211A2N'].values
n = len(y)

# Define a more refined seasonal effect (monthly seasonality with sine and cosine)
months = data.index.month
seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
seasonal_effect_cos = np.cos(2 * np.pi * months / 12)

# Lagged variables for the TM forecasting model (including seasonal effect)
# Use lagged values and their squared values (quadratic terms)
lag_1 = y[4:-1]
lag_2 = y[3:-2]
lag_3 = y[2:-3]
lag_4 = y[1:-4]
lag_5 = y[0:-5]

# Quadratic terms (squared lagged variables)
lag_1_sq = lag_1 ** 2
lag_2_sq = lag_2 ** 2
lag_3_sq = lag_3 ** 2

# Include seasonal effects, lagged variables, and their quadratic terms
X_full = np.column_stack([lag_1, lag_2, lag_3, lag_4, lag_5, lag_1_sq, lag_2_sq, lag_3_sq, seasonal_effect_sin[4:-1], seasonal_effect_cos[4:-1]])
y_full = y[5:]

# Split data into training and validation (expand training window)
X_train = X_full[-50:-5]  # Use 50 training points
y_train = y_full[-50:-5]
X_valid = X_full[-5:]  # Most recent 5 data points for validation
y_valid = y_full[-5:]

# Fit the initial model on the training data (now includes quadratic terms and seasonal effect)
model = LinearRegression().fit(X_train, y_train)

# Calculate validation error on the 5 most recent data points
y_pred_valid = model.predict(X_valid)
validation_error = np.mean((y_valid - y_pred_valid)**2)
print(f"Initial validation error with 50th to 5th points: {validation_error}")

# Get the inverse of (X_train.T * X_train)
A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

# Sherman-Morrison update function
def sherman_morrison_update(A_inv, u, v):
    u = u.reshape(-1, 1)  # Ensure column vector
    v = v.reshape(1, -1)  # Ensure row vector
    A_inv_u = np.dot(A_inv, u)
    v_T_A_inv = np.dot(v, A_inv)
    return A_inv - np.outer(A_inv_u, v_T_A_inv) / (1 + np.dot(v, A_inv_u))

# Early stopping setup
early_stopping_rounds = 3
worse_count = 0

# Start dynamic updating by adding one older data point at a time
best_error = validation_error
best_step = 0

for step in range(10, 0, -1):  # Adding older data points one at a time
    X_new = X_full[-50 - step].reshape(1, -1)  # Add older data point as row (includes quadratic terms and seasonal effect)
    y_new = np.array([y_full[-50 - step]])

    u = X_new.T
    v = X_new

    A_inv_updated = sherman_morrison_update(A_inv, u, v)

    # Update model coefficients without affecting the intercept
    new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
    model.coef_ = new_coef

    # Update the intercept separately
    model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

    # Predict on the validation set
    y_pred_valid_new = model.predict(X_valid)
    new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
    print(f"Step {step}: Validation error after adding older data: {new_validation_error}")

    # Check if the new error is better
    if new_validation_error < best_error:
        best_error = new_validation_error
        best_step = step
        A_inv = A_inv_updated
        worse_count = 0  # Reset the worse count since error has improved
    else:
        worse_count += 1  # Increase the count for worse error
        if worse_count >= early_stopping_rounds:
            print(f"Stopping early due to consecutive worse validation errors at step {step}.")
            break

print(f"Best model found at step {best_step} with validation error: {best_error}")

# Visualization
plt.plot(y_valid, label="Actual")
plt.plot(y_pred_valid, label="Predicted")
plt.title("Validation Results with Quadratic Terms")
plt.xlabel("Time")
plt.ylabel("Electricity Production")
plt.legend()
plt.show()

######raw xgboost electrcity data#######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV file (Electricity data)
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')

# Extract the target variable
y = data['IPG2211A2N'].values
n = len(y)

# Define seasonal effects (use both sine and cosine components for seasonality)
months = data.index.month
seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
seasonal_effect_cos = np.cos(2 * np.pi * months / 12)

# Lagged variables for TM forecasting model (including expanded seasonal effect)
X_full = np.column_stack([y[4:-1], y[3:-2], y[2:-3], y[1:-4], y[0:-5], seasonal_effect_sin[4:-1], seasonal_effect_cos[4:-1]])
y_full = y[5:]

# Split data into training and validation sets
X_train = X_full[-50:-5]  # Expanding the training window
y_train = y_full[-50:-5]
X_valid = X_full[-5:]  # Most recent 5 data points for validation
y_valid = y_full[-5:]

# Number of boosting iterations
n_iterations = 300
# Lower learning rate for more gradual updates
learning_rate = 0.01
# Regularization parameters
lambda_reg = 1.0  # L2 regularization term

# Initial prediction (mean of the target)
initial_prediction = np.mean(y_train)
predictions_train = np.full(y_train.shape, initial_prediction)

# Define the loss function, gradient, and hessian
def loss(y, y_pred):
    return (y - y_pred) ** 2

def gradient(y, y_pred):
    return -2 * (y - y_pred)

def hessian(y, y_pred):
    return 2 * np.ones_like(y_pred)

# Function for dynamic updating with decision tree
def dynamic_update(X, y, n_iterations, learning_rate, lambda_reg):
    predictions = np.full(y.shape, initial_prediction)
    
    for i in range(n_iterations):
        # Compute gradient and hessian
        gradients = gradient(y, predictions)
        hessians = hessian(y, predictions)
        
        # Fit a decision tree to the gradients (increase depth for better learning)
        tree = DecisionTreeRegressor(max_depth=4)  # Adjusting tree depth
        tree.fit(X, gradients)
        prediction_update = tree.predict(X)
        
        # Update predictions using gradient update rule
        predictions -= learning_rate * prediction_update / (hessians + lambda_reg)
        
    return predictions

# Initial boosting iterations on training data
predictions_train = dynamic_update(X_train, y_train, n_iterations, learning_rate, lambda_reg)

# Adding new data points dynamically and updating the model
best_error = float('inf')  # Track the best validation error
early_stopping_rounds = 3  # Add early stopping
worse_count = 0

for step in range(10):
    # Add one older data point at a time
    X_new = X_full[-50 - step].reshape(1, -1)  # Add older data point
    y_new = np.array([y_full[-50 - step]])

    # Combine new data with training set
    X_combined = np.vstack([X_train, X_new])
    y_combined = np.hstack([y_train, y_new])

    # Refit the model using combined data
    predictions_combined = dynamic_update(X_combined, y_combined, n_iterations, learning_rate, lambda_reg)

    # Calculate validation error
    y_pred_valid = predictions_combined[-len(y_valid):]  # Use combined model to predict validation data
    validation_error = mean_squared_error(y_valid, y_pred_valid)
    print(f"Step {step + 1}: Validation error after adding new data: {validation_error}")

    # Track the best error
    if validation_error < best_error:
        best_error = validation_error
        worse_count = 0  # Reset counter if error improves
    else:
        worse_count += 1
        if worse_count >= early_stopping_rounds:
            print(f"Stopping early as validation error increased at step {step + 1}.")
            break

# Final predictions after dynamic updating
print(f"Best validation error: {best_error}")

# Visualization of predictions
plt.figure(figsize=(10, 6))
plt.plot(y_valid, label="Actual Validation Data", marker='o')
plt.plot(predictions_combined[-len(y_valid):], label="Predicted Validation Data", marker='x')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Validation Results After Dynamic Model Updating with XGBoost-style Algorithm')
plt.legend()
plt.show()

#######NN for electricty data########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ReLU function
def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU function
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Mean Squared Error Loss
def loss(y, y_pred):
    return 0.5 * np.mean((y - y_pred)**2)

# Forward pass function
def forward_pass(x, W1, b1, W2, b2):
    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)  # Use ReLU
    z2 = np.dot(W2, a1) + b2
    y_pred = z2  # Output layer for regression tasks
    return y_pred, z1, a1

# Backward pass function
def backward_pass(x, y, y_pred, z1, a1, W2):
    dz2 = y_pred - y  # Gradient of loss w.r.t. output
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(W2.T, dz2) * relu_derivative(z1)
    dW1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Dynamic update function for weights and biases
def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Load the CSV file (Electricity data)
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')

# Extract the target variable
y_data = data['IPG2211A2N'].values
n = len(y_data)

# Define seasonal effects
months = data.index.month
seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
seasonal_effect_cos = np.cos(2 * np.pi * months / 12)

# Feature engineering (lagged variables and seasonal effects)
X_full = np.column_stack([y_data[4:-1], y_data[3:-2], y_data[2:-3], y_data[1:-4], y_data[0:-5],
                          seasonal_effect_sin[4:-1], seasonal_effect_cos[4:-1]])
y_full = y_data[5:]

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_full_scaled = scaler_X.fit_transform(X_full)
y_full_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).flatten()

# Split data into training and validation
X_train = X_full_scaled[-110:-10].T  # Training data
y_train = y_full_scaled[-110:-10].reshape(1, -1)
X_valid = X_full_scaled[-10:].T  # Validation data
y_valid = y_full_scaled[-10:].reshape(1, -1)

# Neural network parameters
input_size = X_train.shape[0]
hidden_size = 20
output_size = 1
learning_rate = 0.001

# Initialize weights and biases
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

# Training the NN with initial data
epochs = 2000
for epoch in range(epochs):
    y_pred_train, z1_train, a1_train = forward_pass(X_train, W1, b1, W2, b2)
    train_loss = loss(y_train, y_pred_train)

    dW1, db1, dW2, db2 = backward_pass(X_train, y_train, y_pred_train, z1_train, a1_train, W2)
    W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

# Initial validation with the initial model
y_pred_valid, _, _ = forward_pass(X_valid, W1, b1, W2, b2)
y_pred_valid_raw = scaler_y.inverse_transform(y_pred_valid.flatten().reshape(-1, 1))  # Reverse back to raw data scale
y_valid_raw = scaler_y.inverse_transform(y_valid.flatten().reshape(-1, 1))  # Reverse validation data back to raw data scale

initial_validation_error = loss(y_valid_raw, y_pred_valid_raw)
print(f'Initial Validation Error (in raw data): {initial_validation_error}')

# Dynamic updating by adding one data point at a time
best_error = initial_validation_error
early_stopping_rounds = 3
worse_count = 0

for step in range(10):
    # Add one older data point at a time
    X_new = X_full_scaled[-110 - step].reshape(-1, 1)  # Single column for new data
    y_new = np.array([[y_full_scaled[-110 - step]]])  # New true label

    # Forward pass with new data
    y_pred_new, z1_new, a1_new = forward_pass(X_new, W1, b1, W2, b2)
    
    # Backward pass with new data and update weights
    dW1_new, db1_new, dW2_new, db2_new = backward_pass(X_new, y_new, y_pred_new, z1_new, a1_new, W2)
    W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1_new, db1_new, dW2_new, db2_new, learning_rate)
    
    # Validation with updated model
    y_pred_valid, _, _ = forward_pass(X_valid, W1, b1, W2, b2)
    y_pred_valid_raw = scaler_y.inverse_transform(y_pred_valid.flatten().reshape(-1, 1))  # Reverse back to raw data scale
    
    new_validation_error = loss(y_valid_raw, y_pred_valid_raw)
    
    print(f"Step {step + 1}: Validation error after adding new data: {new_validation_error}")
    
    # Early stopping check
    if new_validation_error < best_error:
        best_error = new_validation_error
        worse_count = 0
    else:
        worse_count += 1
        if worse_count >= early_stopping_rounds:
            print(f"Stopping early as validation error increased at step {step + 1}.")
            break

print(f"Best validation error (in raw data): {best_error}")

# Visualization of the validation predictions
plt.plot(y_valid_raw, label='Actual Validation Data')
plt.plot(y_pred_valid_raw, label='Predicted Validation Data')
plt.xlabel('Time')
plt.ylabel('Electricity Production')
plt.title('Validation Results After Dynamic NN Model Updating')
plt.legend()
plt.show()


############enhance version for yule model#############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the CSV file
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as index

def dynamic_yule_model_block_update(data, 
                                    target_variable='IPG2211A2N', 
                                    holdout_points=5,
                                    step_size=1,
                                    early_stop_rounds=3,
                                    initial_points=50,  # Use 50 points for initial training
                                    seasonality_terms=True,
                                    add_quadratic_terms=True,
                                    regressor_type=LinearRegression,  # Make this parameter active
                                    verbose=True,
                                    plot_results=True):
    
    # Extract the target variable (IPG2211A2N)
    y = data[target_variable].values
    n = len(y)

    # Initialize the list for the features (lagged variables)
    X_full = []

    # Lagged variables for the TM forecasting model
    lag_1 = y[4:-1]
    lag_2 = y[3:-2]
    lag_3 = y[2:-3]
    lag_4 = y[1:-4]
    lag_5 = y[0:-5]
    
    X_full.append(lag_1)
    X_full.append(lag_2)
    X_full.append(lag_3)
    X_full.append(lag_4)
    X_full.append(lag_5)

    # Add quadratic terms if the parameter is set to True
    if add_quadratic_terms:
        X_full.append(lag_1 ** 2)
        X_full.append(lag_2 ** 2)
        X_full.append(lag_3 ** 2)

    # Add seasonal effects if the parameter is set to True
    if seasonality_terms:
        months = data.index.month
        seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
        seasonal_effect_cos = np.cos(2 * np.pi * months / 12)
        X_full.append(seasonal_effect_sin[4:-1])
        X_full.append(seasonal_effect_cos[4:-1])

    # Convert the list of features to a NumPy array
    X_full = np.column_stack(X_full)
    y_full = y[5:]

    # Split data into training and validation (expand training window)
    holdp = -1 * holdout_points
    initial_p = -1 * initial_points
    X_train = X_full[initial_p:holdp]  # Use initial points for training
    y_train = y_full[initial_p:holdp]
    X_valid = X_full[holdp:]  # Most recent data points for validation
    y_valid = y_full[holdp:]

    # Fit the initial model using the specified regression model type
    model = regressor_type().fit(X_train, y_train)

    # Calculate validation error on the validation set
    y_pred_valid = model.predict(X_valid)
    validation_error = np.mean((y_valid - y_pred_valid)**2)
    if verbose:
        print(f"Initial validation error with {initial_points} training points: {validation_error}")

    # Get the inverse of (X_train.T * X_train)
    A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

    # Block Sherman-Morrison update function
    def block_sherman_morrison_update(A_inv, U, V):
        U_T_A_inv = np.dot(U.T, A_inv)  # Shape (k, N)
        V_A_inv = np.dot(A_inv, V.T)  # Shape (N, k)

        # Use the block Sherman-Morrison formula for rank-k updates
        block_matrix = np.eye(U_T_A_inv.shape[0]) + np.dot(U_T_A_inv, V.T)  # Shape (k, k)
        block_matrix_inv = np.linalg.inv(block_matrix)

        # Update A_inv using the block Sherman-Morrison formula
        A_inv_update = A_inv - np.dot(V_A_inv, np.dot(block_matrix_inv, U_T_A_inv))

        return A_inv_update

    # Early stopping setup
    worse_count = 0
    best_error = validation_error
    best_step = 0

    # Start dynamic updating by adding blocks of older data points at a time
    for step in range(10, 0, -step_size):  # Adding older data points in blocks
        # Add block of older data points (rows)
        X_new = X_full[initial_p - step: initial_p - step + step_size]
        y_new = np.array(y_full[initial_p - step: initial_p - step + step_size])

        u = X_new.T
        v = X_new

        # Update A_inv using the block Sherman-Morrison update (for multiple data points)
        A_inv_updated = block_sherman_morrison_update(A_inv, u, v)

        # Update model coefficients without affecting the intercept
        new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
        model.coef_ = new_coef

        # Update the intercept separately
        model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

        # Predict on the validation set
        y_pred_valid_new = model.predict(X_valid)
        new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
        if verbose:
            print(f"Step {step}: Validation error after adding {step_size} older data points: {new_validation_error}")

        # Check if the new error is better
        if new_validation_error < best_error:
            best_error = new_validation_error
            best_step = step
            A_inv = A_inv_updated
            worse_count = 0  # Reset the worse count since error has improved
        else:
            worse_count += 1  # Increase the count for worse error
            if worse_count >= early_stopping_rounds:
                if verbose:
                    print(f"Stopping early due to consecutive worse validation errors at step {step}.")
                break
            
    print ("y_pred_valid_new")
    print (y_pred_valid_new)
    if verbose:
        print(f"Best model found at step {best_step} with validation error: {best_error}")

    if plot_results:
        plt.plot(y_valid, label="Actual")
        plt.plot(y_pred_valid, label="Predicted")
        plt.title("Validation Results with Block Sherman-Morrison Updating")
        plt.xlabel("Time")
        plt.ylabel("Electricity Production")
        plt.legend()
        plt.show()

# Run the function
dynamic_yule_model_block_update(data, 
                                step_size=1,
                                seasonality_terms=True,
                                add_quadratic_terms=True, 
                                regressor_type=LinearRegression)


####xgboost update#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Load the CSV file (Electricity data)
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')

# Extract the target variable
y = data['IPG2211A2N'].values
n = len(y)

# Define seasonal effects (use both sine and cosine components for seasonality)
months = data.index.month
seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
seasonal_effect_cos = np.cos(2 * np.pi * months / 12)

# Lagged variables for TM forecasting model (including expanded seasonal effect)
X_full = np.column_stack([y[4:-1], y[3:-2], y[2:-3], y[1:-4], y[0:-5], 
                          seasonal_effect_sin[4:-1], seasonal_effect_cos[4:-1]])
y_full = y[5:]

# Split data into training and test sets (hold out last H points for final evaluation)
H = -5
X_train = X_full[:H]  # Training data excluding last 5 points
y_train = y_full[:H]

X_test = X_full[H:]  # Test set (last 5 points, held out)
y_test = y_full[H:]

# Define initial simplified boosting parameters with regularization
learning_rate = 0.05
max_depth = 2
lambda_reg = 7
min_child_weight = 2
n_iterations = 100  # Manageable iteration count

# Initial prediction (mean of the target)
initial_prediction = np.mean(y_train)
predictions_train = np.full(y_train.shape, initial_prediction)

# Define the gradient and Hessian functions
def gradient(y, y_pred):
    return -2 * (y - y_pred)

# Simplified boosting process with regularization
def simplified_boosting(X, y, n_iterations, learning_rate, 
                        max_depth, lambda_reg, min_child_weight):
    
    predictions = np.full(y.shape, initial_prediction)
    
    for i in range(n_iterations):
        # Compute gradients
        gradients = gradient(y, predictions)
        
        # Fit a decision tree to the gradients with regularization
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_child_weight)
        tree.fit(X, gradients)
        
        # Update predictions using the tree's predictions
        prediction_update = tree.predict(X)
        predictions -= learning_rate * prediction_update
    
    return predictions

# Train the simplified model on the training set
predictions_train = simplified_boosting(X_train, y_train, n_iterations, learning_rate,
                                        max_depth, lambda_reg, min_child_weight)

# Sequential one-step-ahead prediction on the test set
y_pred_test = []
X_test_updated = X_test.copy()

for i in range(len(X_test)):
    # Predict one time point (Y_(N+1))
    single_point_pred = simplified_boosting(X_test_updated[:i+1], y_test[:i+1], n_iterations,
                                            learning_rate, max_depth, lambda_reg, min_child_weight)[-1]
    y_pred_test.append(single_point_pred)
    
    # Update the lagged features for the next prediction by using the actual y_test
    if i < len(X_test) - 1:
        X_test_updated[i+1, 0] = y_test[i]  # Update lag_1 with true y_test value
        for j in range(1, 5):  # Update lag_2, lag_3, etc.
            X_test_updated[i+1, j] = X_test_updated[i, j-1]

y_pred_test = np.array(y_pred_test)

# Calculate test error on sequential predictions
initial_test_error = mean_squared_error(y_test, y_pred_test)
print(f"Initial Test error (on held-out last 5 points, sequential prediction): {initial_test_error}")

# Dynamic updating - Weighted Updates (giving more weight to new data)
best_error = initial_test_error
early_stopping_rounds = 3
worse_count = 0
weight_new_data = 0.5  # Adjust this weight to emphasize new data more or less

for step in range(5):
    print(f"\nStep {step + 1}: Updating with new data...")
    
    # Add the next point from test set to training set
    X_train = np.vstack([X_train, X_test_updated[step].reshape(1, -1)])
    y_train = np.append(y_train, y_test[step])

    # Predict with the updated model
    predictions_combined = simplified_boosting(X_train, y_train, n_iterations, learning_rate,
                                               max_depth, lambda_reg, min_child_weight)

    # Predict on validation (remaining test points)
    if step + 1 < len(X_test):
        y_pred_test_updated = simplified_boosting(X_test_updated[step+1:], y_test[step+1:], 
                                                  n_iterations, learning_rate, max_depth, 
                                                  lambda_reg, min_child_weight)
    
    # Compute validation error after adding new data
    new_validation_error = mean_squared_error(y_test[step+1:], y_pred_test_updated)
    print(f"Validation error after adding new data (step {step + 1}): {new_validation_error}")
    
    # Early stopping condition
    if new_validation_error < best_error:
        best_error = new_validation_error
        worse_count = 0
    else:
        worse_count += 1
        if worse_count >= early_stopping_rounds:
            print(f"Stopping early at step {step + 1} as validation error increased.")
            break

print(f"\nBest validation error: {best_error}")

# Visualization of predictions on test data only
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Test Data", marker='o')
plt.plot(y_pred_test, label="Predicted Test Data", marker='x')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Test Results After Simplified Manual XGBoost Model (Held-Out Last 5 Points)')
plt.legend()
plt.show()


#####adding old yule######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the CSV file
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as index

def dynamic_yule_model_block_update(data, 
                                    target_variable='IPG2211A2N', 
                                    holdout_points=5,  # Most recent 5 points for validation
                                    step_size=1,
                                    early_stop_rounds=3,
                                    initial_points=50,  # Start with 50 points for initial training
                                    total_steps=13,  # Add this parameter to control the range
                                    seasonality_terms=True,
                                    add_quadratic_terms=True,
                                    regressor_type=LinearRegression,  # Using Linear Regression model
                                    verbose=True,
                                    plot_results=True):
    
    # Extract the target variable (IPG2211A2N)
    y = data[target_variable].values
    n = len(y)

    # Initialize the list for the features (lagged variables)
    X_full = []

    # Lagged variables for the TM forecasting model
    lag_1 = y[4:-1]
    lag_2 = y[3:-2]
    lag_3 = y[2:-3]
    lag_4 = y[1:-4]
    lag_5 = y[0:-5]
    
    X_full.append(lag_1)
    X_full.append(lag_2)
    X_full.append(lag_3)
    X_full.append(lag_4)
    X_full.append(lag_5)

    # Add quadratic terms if the parameter is set to True
    if add_quadratic_terms:
        X_full.append(lag_1 ** 2)
        X_full.append(lag_2 ** 2)
        X_full.append(lag_3 ** 2)

    # Add seasonal effects if the parameter is set to True
    if seasonality_terms:
        months = data.index.month
        seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
        seasonal_effect_cos = np.cos(2 * np.pi * months / 12)
        X_full.append(seasonal_effect_sin[4:-1])
        X_full.append(seasonal_effect_cos[4:-1])

    # Convert the list of features to a NumPy array
    X_full = np.column_stack(X_full)
    y_full = y[5:]

    # Split data into training and validation
    holdp = -1 * holdout_points
    initial_p = -1 * initial_points
    X_train = X_full[initial_p:holdp]  # Initial training data (t6 to t55)
    y_train = y_full[initial_p:holdp]
    X_valid = X_full[holdp:]  # Validation set (most recent 5 points)
    y_valid = y_full[holdp:]

    # Fit the initial model
    model = regressor_type().fit(X_train, y_train)

    # Function to predict on validation set (held-out points t5 to t1)
    def one_time_point_prediction(model, X_valid, y_valid):
        y_pred_valid = []
        for i in range(holdout_points):
            X_current = X_valid[i:i+1]
            y_pred_valid.append(model.predict(X_current)[0])
        return np.array(y_pred_valid)

    # Initial prediction on validation set (t5, t4, ..., t1)
    y_pred_valid = one_time_point_prediction(model, X_valid, y_valid)
    validation_error = mean_squared_error(y_valid, y_pred_valid)
    
    if verbose:
        print(f"Initial validation error with {initial_points} training points: {validation_error}")

    # Block Sherman-Morrison update function for linear regression coefficient updates
    def block_sherman_morrison_update(A_inv, U, V):
        U_T_A_inv = np.dot(U.T, A_inv)  # Shape (k, N)
        V_A_inv = np.dot(A_inv, V.T)  # Shape (N, k)

        # Use the block Sherman-Morrison formula for rank-k updates
        block_matrix = np.eye(U_T_A_inv.shape[0]) + np.dot(U_T_A_inv, V.T)  # Shape (k, k)
        block_matrix_inv = np.linalg.inv(block_matrix)

        # Update A_inv using the block Sherman-Morrison formula
        A_inv_update = A_inv - np.dot(V_A_inv, np.dot(block_matrix_inv, U_T_A_inv))

        return A_inv_update

    # Early stopping setup
    worse_count = 0
    best_error = validation_error
    best_step = 0

    # Get the inverse of (X_train.T * X_train)
    A_inv = np.linalg.inv(np.dot(X_train.T, X_train))

    # Start dynamic updating by adding older data points incrementally from t56 onward
    for step in range(1, total_steps + 1):  # Use the 'total_steps' parameter to control range
        X_new = X_full[initial_p + step].reshape(1, -1)
        y_new = np.array([y_full[initial_p + step]])

        u = X_new.T
        v = X_new

        # Update A_inv using Sherman-Morrison update
        A_inv_updated = block_sherman_morrison_update(A_inv, u, v)

        # Update model coefficients without affecting the intercept
        new_coef = np.dot(A_inv_updated, np.dot(X_train.T, y_train))
        model.coef_ = new_coef

        # Update the intercept separately
        model.intercept_ = np.mean(y_train - np.dot(X_train, model.coef_))

        # Predict on the validation set (t5, t4, ..., t1)
        y_pred_valid_new = one_time_point_prediction(model, X_valid, y_valid)
        new_validation_error = np.mean((y_valid - y_pred_valid_new)**2)
        
        if verbose:
            print(f"Step {step}: Validation error after adding t{56 + step - 1}: {new_validation_error}")

        # Early stopping based on validation error improvement
        if new_validation_error < best_error:
            best_error = new_validation_error
            best_step = step
            A_inv = A_inv_updated
            worse_count = 0  # Reset worse count if error improves
        else:
            worse_count += 1  # Increase worse count if error worsens
            if worse_count >= early_stop_rounds:
                if verbose:
                    print(f"Stopping early due to consecutive worse validation errors at step {step}.")
                break

    if verbose:
        print(f"Best model found at step {best_step} with validation error: {best_error}")

    if plot_results:
        plt.plot(y_valid, label="Actual")
        plt.plot(y_pred_valid_new, label="Predicted")
        plt.title("Validation Results with Dynamic Updating")
        plt.xlabel("Time")
        plt.ylabel("Electricity Production")
        plt.legend()
        plt.show()

# Run the function
dynamic_yule_model_block_update(data, 
                                step_size=1,
                                initial_points=60,
                                seasonality_terms=True,
                                add_quadratic_terms=True, 
                                total_steps=15,  # Now the number of steps can be set here
                                regressor_type=LinearRegression)

###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the CSV file
file_path = r'C:\backupcgi\final_bak\Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as index

def dynamic_yule_model_arrive_new_data(data, 
                                       target_variable='IPG2211A2N', 
                                       holdout_points=5,  # Number of points for updating and validation
                                       step_size=1,
                                       initial_points=50,  # Start with 50 points for initial training
                                       seasonality_terms=True,
                                       add_quadratic_terms=True,
                                       regressor_type=LinearRegression,  # Using Linear Regression model
                                       verbose=True,
                                       plot_results=True):
    
    # Extract the target variable (IPG2211A2N)
    y = data[target_variable].values
    n = len(y)

    # Initialize the list for the features (lagged variables)
    X_full = []

    # Lagged variables for the TM forecasting model
    lag_1 = y[4:-1]
    lag_2 = y[3:-2]
    lag_3 = y[2:-3]
    lag_4 = y[1:-4]
    lag_5 = y[0:-5]
    
    X_full.append(lag_1)
    X_full.append(lag_2)
    X_full.append(lag_3)
    X_full.append(lag_4)
    X_full.append(lag_5)

    # Add quadratic terms if the parameter is set to True
    if add_quadratic_terms:
        X_full.append(lag_1 ** 2)
        X_full.append(lag_2 ** 2)
        X_full.append(lag_3 ** 2)

    # Add seasonal effects if the parameter is set to True
    if seasonality_terms:
        months = data.index.month
        seasonal_effect_sin = np.sin(2 * np.pi * months / 12)
        seasonal_effect_cos = np.cos(2 * np.pi * months / 12)
        X_full.append(seasonal_effect_sin[4:-1])
        X_full.append(seasonal_effect_cos[4:-1])

    # Convert the list of features to a NumPy array
    X_full = np.column_stack(X_full)
    y_full = y[5:]

    # Initial training and validation data split
    holdp = -1 * holdout_points  # Most recent `holdout_points` are for validation
    initial_p = -1 * initial_points
    X_train = X_full[initial_p:holdp]  # Initial training data (t6 to t50)
    y_train = y_full[initial_p:holdp]
    X_valid = X_full[holdp:]  # Validation set (t5, t4, t3, t2, t1)
    y_valid = y_full[holdp:]

    # Fit the initial model
    model = regressor_type().fit(X_train, y_train)

    # Function to predict on the remaining validation points (one-time point validation)
    def one_time_point_validation(model, X_valid, y_valid):
        y_pred_valid = []
        for i in range(1, len(X_valid)):  # Skip the first point as it's used for update
            X_current = X_valid[i:i+1]
            y_pred_valid.append(model.predict(X_current)[0])
        return np.array(y_pred_valid)

    # Start updating with "new data" (t5, t4, t3, etc.) and validate on the remaining points
    for step in range(holdout_points):  # For 5 iterations, we add data points t5, t4, t3, etc.
        # Add one new data point (start from t5)
        X_new = X_valid[step:step + step_size]
        y_new = np.array([y_valid[step]])

        # Add the new data to the training set
        X_train = np.vstack([X_train, X_new])
        y_train = np.append(y_train, y_new)

        # Fit the model with the updated training set
        model = regressor_type().fit(X_train, y_train)

        # Validate on the remaining validation points (one-time point validation)
        y_pred_valid_new = one_time_point_validation(model, X_valid, y_valid)
        validation_error = mean_squared_error(y_valid[step + 1:], y_pred_valid_new)
        
        if verbose:
            print(f"After adding t{5 - step}: Validation error: {validation_error}")

    if plot_results:
        plt.plot(y_valid[step:], label="Actual")
        plt.plot(y_pred_valid_new, label="Predicted")
        plt.title("Validation Results with Dynamic Updating (New Data Arriving)")
        plt.xlabel("Time")
        plt.ylabel("Electricity Production")
        plt.legend()
        plt.show()

# Run the function
dynamic_yule_model_arrive_new_data(data, 
                                   holdout_points=5,  # This will control how many updates are performed
                                   seasonality_terms=True,
                                   add_quadratic_terms=True, 
                                   regressor_type=LinearRegression)
