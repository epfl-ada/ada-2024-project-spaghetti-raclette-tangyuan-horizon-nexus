from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd

def train_ols_model(movie_master_dataset, df_factors):
    """
    Train an OLS model using specified factors and print the summary and coefficients.

    Args:
        movie_master_dataset (DataFrame): Main dataset containing movie metadata.
        df_factors (DataFrame): DataFrame with selected feature columns for prediction.

    Returns:
        model (statsmodels.OLS): Trained OLS regression model.
    """
    df_output = movie_master_dataset[['success']]

    # Print all features in the dataset
    print("Features in movie_master_dataset:")
    print(movie_master_dataset.columns.tolist())

    # Check for missing values
    missing_values = movie_master_dataset.isnull().sum()
    print("\nMissing Values Check:")
    print(missing_values[missing_values > 0])

    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_factors, df_output, test_size=0.2, random_state=42)

    # Print dataset sizes
    print(f"\nNumber of samples in the training set: {X_train.shape[0]}")
    print(f"Number of samples in the testing set: {X_test.shape[0]}")

    # Step 2: Standardize the training and test features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Add a constant column to the scaled data
    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled)

    # Step 4: Train the OLS model using statsmodels
    model = sm.OLS(y_train, X_train_scaled).fit()

    # Print the summary of the model with feature names
    model_summary = model.summary()
    print(model_summary)

    # Step 5: Create a feature-coefficient table
    feature_names = ['const'] + list(X_train.columns)
    coef_table = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.params
    })

    # Print the coefficients table
    print("\nModel Coefficients Table:")
    print(coef_table)

    influenceFactors_with_OLS(model, X_train_scaled, y_train, feature_names)

    # Compute the covariance matrix and normalize it
    B = np.cov(X_train_scaled[:, 1:], rowvar=False)  # Exclude the constant column
    D = np.sqrt(np.diag(B))
    D_inv = np.diag(1.0 / D)
    C = D_inv @ B @ D_inv  # Correlation matrix

    # Create a heatmap for the correlation matrix
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        C, cmap="coolwarm", cbar_kws={'label': 'Scale'}, annot=True,
        xticklabels=X_train.columns, yticklabels=X_train.columns
    )
    plt.title("Correlation Matrix Heatmap")
    plt.show()


    return model

import matplotlib.pyplot as plt
import numpy as np

def influenceFactors_with_OLS(model, X_train_scaled, y_train, feature_names):
    # Convert y_train to a NumPy array for numerical operations
    y_train_np = y_train.values.flatten()  # Flatten to ensure it's a 1D array
    residuals = model.resid
    residual_std = residuals.std()

    # Compute variance reduction for each feature
    stds_without_factors = []
    for idx in range(1, len(feature_names)):  # Skip the constant term
        temp_X = X_train_scaled.copy()
        temp_X[:, idx] = 0  # Set the factor to 0
        temp_model = sm.OLS(y_train_np, temp_X).fit()
        std_without_factor = temp_model.resid.std()
        stds_without_factors.append(std_without_factor)

    # Calculate variance explained
    explained_variance_percent = ((y_train_np.std()**2 - residual_std**2) / y_train_np.std()**2) * 100
    print("Percentage of variance explained by the model:")
    print(f"{explained_variance_percent:.2f}%")

    # Pie chart data
    contributions = [np.sqrt(abs((std**2) - residual_std**2)) for std in stds_without_factors]
    contributions.append(residual_std)

    # Correct labels to match contributions
    labels = feature_names[1:] + ['Residual Noise']  # Exclude the constant term from feature names

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(contributions, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Influence of Factors on Success")
    plt.show()

    # Print R^2 score
    r2 = model.rsquared
    print(f"\nR^2 Score: {r2:.4f}")

