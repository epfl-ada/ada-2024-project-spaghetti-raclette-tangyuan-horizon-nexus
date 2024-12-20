import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def plot_revenue_vs_success(movie_master_dataset, background_color='#1e1e1e'):
    """
    Plot Box Office Revenue vs. Success with Ratings as Color Encoding.
    
    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata.
    - background_color (str): Background color for the plot (default is dark).
    """
    # Define custom colormap from purple to red
    colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('custom_thermal', colors)

    # Set dark mode style
    plt.style.use('dark_background')

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=500)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Normalize ratings for the color mapping
    norm = mcolors.Normalize(vmin=0, vmax=10)  # Ratings range from 0 to 10

    # Scatter plot with color based on rating and transparency
    sc = ax.scatter(
        movie_master_dataset['success'],
        movie_master_dataset['revenue'],
        c=movie_master_dataset['rating'],
        cmap=cmap,
        norm=norm,
        alpha=0.3,
        s=3
    )

    # Fit a least squares regression line to the log-transformed revenue
    log_revenue = np.log(movie_master_dataset['revenue'].replace(0, np.nan))
    coefficients = np.polyfit(movie_master_dataset['success'], log_revenue, 1)
    poly = np.poly1d(coefficients)

    # Calculate mean and standard deviation of success
    mean_success = movie_master_dataset['success'].mean()
    std_success = movie_master_dataset['success'].std()

    # Define x values for the regression line within 1 standard deviation
    x = np.linspace(mean_success - std_success, mean_success + std_success, 100)
    y = np.exp(poly(x))

    # Plot the regression line
    ax.plot(x, y, color='magenta', linewidth=2, label='Least Squares Regression Line (1 std)')

    # Add a color bar
    cbar = plt.colorbar(sc, ax=ax, aspect=30, pad=0.02)
    cbar.set_label('Movie Rating', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Plot settings
    ax.set_title('Box Office Revenue vs. Success (log)', fontsize=16, color='white')
    ax.set_xlabel('Success = rating * log(vote_count)', fontsize=11, color='white')
    ax.set_ylabel('Box Office Revenue [$] (log)', fontsize=11, color='white')
    ax.set_yscale('log')  # Use a log scale for revenue
    ax.grid(True, color='gray')
    ax.legend()

    plt.show()
