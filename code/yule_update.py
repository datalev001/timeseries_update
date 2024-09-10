import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the CSV file
file_path = 'Electric_Production_tm.csv'
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
