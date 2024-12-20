import pandas as pd
import matplotlib.pyplot as plt

def plot_success_statistics(movie_master_dataset, calculate_success, background_color='#1e1e1e'):
    """
    Plot movie success statistics, including yearly success statistics and individual success per movie.

    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata including 'release_date' and 'success'.
    - calculate_success (function): Function to calculate the success metric.
    - background_color (str): Background color for the plot (default is dark).
    """
    # Ensure 'release_date' is in datetime format
    movie_master_dataset['release_date'] = pd.to_datetime(
        movie_master_dataset['release_date'], errors='coerce'
    )

    # Calculate success for the dataset and filter out NaN values
    movie_master_dataset['success'] = movie_master_dataset.apply(calculate_success, axis=1)
    movie_master_dataset = movie_master_dataset.dropna(subset=['success'])

    # Group by year and calculate success statistics
    success_stats = movie_master_dataset.groupby(
        movie_master_dataset['release_date'].dt.year
    )['success'].agg(['mean', 'median', 'std', 'min', 'max'])

    # Filter out years before 1920
    success_stats = success_stats[success_stats.index >= 1920]

    # Create the plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7), dpi=300)
    fig.patch.set_facecolor(background_color)
    for ax in axes:
        ax.set_facecolor(background_color)

    # Main title
    fig.suptitle('Success Analysis', fontsize=24, color='white')

    # Plot 1: Success Statistics
    axes[0].fill_between(
        success_stats.index, 
        success_stats['mean'] - success_stats['std'],
        success_stats['mean'] + success_stats['std'], 
        color='white', alpha=0.2, label='1 Std Dev'
    )
    axes[0].plot(
        success_stats.index, 
        success_stats['median'], 
        label='Median Success', linestyle='--', color='white', linewidth=3
    )
    axes[0].plot(
        success_stats.index, 
        success_stats['mean'], 
        label='Mean Success', color='coral', linewidth=3
    )
    axes[0].set_title('Success Statistics', fontsize=18, color='white')
    axes[0].set_xlabel('Year', fontsize=18, color='white')
    axes[0].set_ylabel('Success', fontsize=18, color='white')
    axes[0].legend(fontsize=14, loc='upper left')
    axes[0].grid(True, color='gray')
    axes[0].set_ylim(0, 90)

    # Plot 2: Individual Success per Movie
    axes[1].plot(
        success_stats.index, 
        success_stats['mean'], 
        label='Mean Success', color='coral', linewidth=3
    )
    axes[1].scatter(
        movie_master_dataset['release_date'].dt.year, 
        movie_master_dataset['success'], 
        color='magenta', alpha=0.1, label='Individual Success', s=30
    )
    axes[1].set_xlim(1920)
    axes[1].set_title('Success per Movie', fontsize=18, color='white')
    axes[1].set_xlabel('Year', fontsize=18, color='white')
    axes[1].set_ylabel('Success', fontsize=18, color='white')
    axes[1].legend(fontsize=14, loc='upper left')
    axes[1].grid(True, color='gray')
    axes[1].set_ylim(0, 90)

    # Set tick parameters for both axes
    for ax in axes:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    plt.show()
