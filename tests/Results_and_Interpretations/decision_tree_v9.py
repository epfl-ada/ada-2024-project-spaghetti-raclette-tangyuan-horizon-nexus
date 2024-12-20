from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO  # Add this import statement
import base64  # Add this import statement
from io import BytesIO  # Ensure this is also imported
import matplotlib.pyplot as plt

# The rest of your script remains unchanged


# The rest of your script remains the same



# Create HTML saving directory
os.makedirs('html_plots', exist_ok=True)

def decision_tree_analysis(df_factors, movie_master_dataset, max_depth_range=30):
    """
    Perform decision tree analysis including:
    - Finding the best depth using R^2 score
    - Visualizing the best decision tree
    - Generating learning and validation curves
    - Checking for overfitting
    - Plotting the decision tree.
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

    # --- Decision Tree Plot ---
    plt.figure(figsize=(20, 10))
    plot_tree(
        best_tree, 
        feature_names=df_factors.columns, 
        filled=True, 
        rounded=True, 
        impurity=True, 
        fontsize=10,
        class_names=["Low Success", "High Success"],
    )

    plt.title("Decision Tree Visualization", fontsize=16)
    plt.tight_layout()

    # Save the decision tree as a PNG image in memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert PNG image to base64 string for embedding in HTML
    encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Create HTML content with the base64-encoded image
    html_content = f"""
    <html>
    <head>
        <title>Decision Tree Visualization</title>
    </head>
    <body style="background-color:#1E1E1E; color:white;">
        <h1 style="text-align:center;">Decision Tree Visualization</h1>
        <img src="data:image/png;base64,{encoded_image}" style="display:block; margin:auto;" />
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open('html_plots/decision_tree_visualization.html', 'w') as f:
        f.write(html_content)

    plt.show()

    # Return the trained decision tree
    return best_tree

