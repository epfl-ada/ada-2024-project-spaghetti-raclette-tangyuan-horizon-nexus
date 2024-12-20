from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import numpy as np
import os

# Create HTML saving directory
os.makedirs('html_plots', exist_ok=True)

def decision_tree_analysis(df_factors, movie_master_dataset, max_depth_range=30):
    """
    Perform decision tree analysis including:
    - Finding the best depth using R^2 score
    - Visualizing the best decision tree
    - Generating learning and validation curves
    - Checking for overfitting

    Args:
        df_factors (DataFrame): Feature matrix (input features).
        movie_master_dataset (DataFrame): Main dataset containing 'success'.
        max_depth_range (int): Maximum depth to consider when searching for the best decision tree.
        
    Returns:
        best_tree (DecisionTreeRegressor): Trained decision tree with the best depth.
    """
    # Define target variable
    y = movie_master_dataset['success']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_factors, y, test_size=0.2, random_state=42)

    # List of possible depths to test
    depths = range(1, max_depth_range + 1)

    # Initialize variables to track the best model
    best_depth = None
    best_r2 = -float('inf')
    best_tree = None

    # --- Find Best Depth ---
    for depth in depths:
        tree_reg = DecisionTreeRegressor(random_state=42, max_depth=depth)
        tree_reg.fit(X_train, y_train)

        y_pred = tree_reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_depth = depth
            best_tree = tree_reg

    # Print the best depth and corresponding R^2 score
    print(f"Best Depth: {best_depth}")
    print(f"R^2 Score at Best Depth: {best_r2 * 100:.2f}%")

    # Evaluate MSE for the best model
    y_pred_best = best_tree.predict(X_test)
    mse_best = mean_squared_error(y_test, y_pred_best)
    print(f"Mean Squared Error at Best Depth: {mse_best}")

    # --- Learning Curve ---
    train_sizes, train_scores, val_scores = learning_curve(
        DecisionTreeRegressor(max_depth=best_depth, random_state=42), 
        X_train, y_train, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Create Learning Curve Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines', name='Training Score',
        line=dict(color='#FF7E1D', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines', name='Validation Score',
        line=dict(color='#B300F2', width=3)
    ))

    fig.update_layout(
        title="Learning Curve: Decision Tree Regressor",
        xaxis_title="Training Set Size",
        yaxis_title="R^2 Score",
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )

    fig.show()
    fig.write_html('html_plots/learning_curve_decision_tree.html')

    # --- Validation Curve: Detect Overfitting ---
    train_scores, val_scores = validation_curve(
        DecisionTreeRegressor(random_state=42), X_train, y_train, 
        param_name="max_depth", param_range=depths, cv=5, scoring="r2"
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Create Validation Curve Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(depths), y=train_mean,
        mode='lines', name='Training Score',
        line=dict(color='#FF7E1D', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=list(depths), y=val_mean,
        mode='lines', name='Validation Score',
        line=dict(color='#B300F2', width=3)
    ))

    fig.update_layout(
        title="Validation Curve: Decision Tree Regressor",
        xaxis_title="Tree Depth",
        yaxis_title="R^2 Score",
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )

    fig.show()
    fig.write_html('html_plots/validation_curve_decision_tree.html')

    return best_tree
