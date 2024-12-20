from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Create HTML saving directory
os.makedirs('html_plots', exist_ok=True)

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

    print("\nModel Coefficients Table:")
    print(coef_table)

    # Compute the covariance matrix and normalize it
    B = np.cov(X_train_scaled[:, 1:], rowvar=False)  # Exclude the constant column
    D = np.sqrt(np.diag(B))
    D_inv = np.diag(1.0 / D)
    C = D_inv @ B @ D_inv  # Correlation matrix

    # Plot the correlation matrix heatmap using Plotly
    fig_corr = go.Figure(data=go.Heatmap(
        z=C,
        x=X_train.columns,
        y=X_train.columns,
        colorscale=['#FF7E1D', '#DE0099', '#B300F2', '#5C00F2', '#0000FF'],
        texttemplate="%{z:.2f}",  # Display correlation values
        textfont=dict(color='white'),  # Make text color white
        colorbar=dict(title='Correlation', tickfont=dict(color='white')),
    ))

    # Adjust layout for square shape
    fig_corr.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis=dict(title='Features', tickfont=dict(color='white')),
        yaxis=dict(title='Features', tickfont=dict(color='white')),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        autosize=False,
        width=800,  # Set a square width
        height=600  # Set a square height
    )

    fig_corr.show()
    fig_corr.write_html('html_plots/correlation_matrix_heatmap.html')

    return model



def influenceFactors_with_OLS(model, X_train_scaled, y_train, feature_names):
    """
    Visualize the influence of factors on movie success.

    Args:
        model (statsmodels.OLS): Trained OLS regression model.
        X_train_scaled (array): Scaled training data.
        y_train (DataFrame): Training target values.
        feature_names (list): List of feature names.

    Returns:
        None
    """
    y_train_np = y_train.values.flatten()
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

    # Calculate explained variance
    explained_variance_percent = ((y_train_np.std()**2 - residual_std**2) / y_train_np.std()**2) * 100
    print("Percentage of variance explained by the model:")
    print(f"{explained_variance_percent:.2f}%")

    # Calculate contributions
    contributions = [np.sqrt(abs((std**2) - residual_std**2)) for std in stds_without_factors]
    contributions.append(residual_std)

    labels = feature_names[1:] + ['Residual Noise']

    # Create interactive pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=contributions,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(colors=['#FF7E1D', '#DE0099', '#B300F2', '#5C00F2', '#0000FF', '#8A2BE2'])
    )])

    fig_pie.update_layout(
        title="Influence of Factors on Success",
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    fig_pie.show()
    fig_pie.write_html('html_plots/influence_of_factors.html')

    # Print R^2 score
    r2 = model.rsquared
    print(f"\nR^2 Score: {r2:.4f}")
