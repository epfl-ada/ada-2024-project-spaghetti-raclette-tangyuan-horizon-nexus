from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


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
        # Train decision tree regressor for each depth
        tree_reg = DecisionTreeRegressor(random_state=42, max_depth=depth)
        tree_reg.fit(X_train, y_train)

        # Predict on the test set
        y_pred = tree_reg.predict(X_test)

        # Evaluate the model
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

    # --- Visualize the Best Decision Tree ---
    plt.figure(figsize=(16, 10))
    plot_tree(
        best_tree, feature_names=df_factors.columns, filled=True, rounded=True, fontsize=10
    )
    plt.title(f"Decision Tree Visualization (Best Depth = {best_depth})")
    plt.show()

    # --- Learning Curve ---
    train_sizes, train_scores, val_scores = learning_curve(
        DecisionTreeRegressor(max_depth=best_depth, random_state=42), 
        X_train, y_train, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Compute averages and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")

    plt.plot(train_sizes, val_mean, label="Validation Score", color="orange")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="orange")

    plt.title("Learning Curve: Decision Tree Regressor")
    plt.xlabel("Training Set Size")
    plt.ylabel("R^2 Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # --- Validation Curve: Detect Overfitting ---
    print("Generating validation curve for overfitting detection...")

    # Generate validation curve
    train_scores, val_scores = validation_curve(
        DecisionTreeRegressor(random_state=42), X_train, y_train, 
        param_name="max_depth", param_range=depths, cv=5, scoring="r2"
    )

    # Calculate averages and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot validation curve
    plt.figure(figsize=(12, 6))
    plt.plot(depths, train_mean, label="Training Score", color="blue")
    plt.fill_between(depths, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")

    plt.plot(depths, val_mean, label="Validation Score", color="orange")
    plt.fill_between(depths, val_mean - val_std, val_mean + val_std, alpha=0.2, color="orange")

    plt.title("Validation Curve: Decision Tree Regressor")
    plt.xlabel("Tree Depth")
    plt.ylabel("R^2 Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    return best_tree
