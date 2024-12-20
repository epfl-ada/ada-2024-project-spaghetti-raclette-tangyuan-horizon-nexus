import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def analyze_factors_influence_ols(model, X_train_scaled, y_train, feature_names):
    """
    Analyze the influence of factors using OLS regression.

    Args:
        model (statsmodels.OLS): Trained OLS regression model.
        X_train_scaled (array): Scaled training features with a constant term.
        y_train (DataFrame): Training target values.
        feature_names (list): Names of features including 'const'.
    """
    # Convert y_train to a NumPy array for numerical operations
    y_train_np = y_train.values.flatten()  # Ensure it's a 1D array
    residuals = model.resid
    residual_std = residuals.std()

    # Compute variance reduction for each feature
    stds_without_factors = []
    for idx in range(1, len(feature_names)):  # Skip the constant term
        temp_X = X_train_scaled.copy()
        temp_X[:, idx] = 0  # Remove the effect of the current feature
        temp_model = sm.OLS(y_train_np, temp_X).fit()
        std_without_factor = temp_model.resid.std()
        stds_without_factors.append(std_without_factor)

    # Calculate variance explained by the model
    explained_variance_percent = ((y_train_np.std()**2 - residual_std**2) / y_train_np.std()**2) * 100
    print("Percentage of variance explained by the model:")
    print(f"{explained_variance_percent:.2f}%")

    # Pie chart data
    contributions = [np.sqrt(abs((std**2) - residual_std**2)) for std in stds_without_factors]
    contributions.append(residual_std)

    # Correct labels to match contributions
    labels = feature_names[1:] + ['Residual Noise']  # Exclude the constant term from labels

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(contributions, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Influence of Factors on Success")
    plt.show()

    # Print R^2 score
    r2 = model.rsquared
    print(f"\nR^2 Score: {r2:.4f}")
