import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_rating_statistics(movie_master_dataset):
    """
    Plot movie ratings analysis, including yearly statistics and individual ratings.

    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata including 'release_date' and 'rating'.
    """
    # Define background color
    background_color = '#1e1e1e'

    # Ensure the release_date column is in datetime format
    movie_master_dataset['release_date'] = pd.to_datetime(
        movie_master_dataset['release_date'], errors='coerce'
    )

    # Filter out movies with release dates before 1920
    filtered_movie_master_dataset = movie_master_dataset[
        movie_master_dataset['release_date'].dt.year >= 1920
    ]

    # Group by year and calculate the statistics for ratings
    rating_stats = filtered_movie_master_dataset.groupby(
        filtered_movie_master_dataset['release_date'].dt.year
    )['rating'].agg(['mean', 'median', 'std', 'min', 'max'])

    # Create the plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7), dpi=300)
    fig.patch.set_facecolor(background_color)
    for ax in axes:
        ax.set_facecolor(background_color)

    # Main title
    fig.suptitle('Ratings Analysis', fontsize=24, color='white')

    # Plot 1: Yearly Rating Statistics
    axes[0].fill_between(
        rating_stats.index, 
        rating_stats['mean'] - rating_stats['std'],
        rating_stats['mean'] + rating_stats['std'], 
        color='white', alpha=0.2, label='1 Std Dev'
    )
    axes[0].plot(
        rating_stats.index, 
        rating_stats['median'], 
        label='Median Rating', 
        linestyle='--', color='white', linewidth=3
    )
    axes[0].plot(
        rating_stats.index, 
        rating_stats['mean'], 
        label='Mean Rating', 
        color='orange', linewidth=3
    )
    axes[0].set_title('Yearly Rating Statistics', fontsize=18, color='white')
    axes[0].set_xlabel('Year', fontsize=18, color='white')
    axes[0].set_ylabel('Rating', fontsize=18, color='white')
    axes[0].legend(fontsize=14, loc='upper left')
    axes[0].grid(True, color='gray')
    axes[0].set_ylim(0, 10)  # Set y-axis limits

    # Plot 2: Ratings per Movie
    axes[1].plot(
        rating_stats.index, 
        rating_stats['mean'], 
        label='Mean Rating', 
        color='orange', linewidth=3
    )
    axes[1].scatter(
        filtered_movie_master_dataset['release_year'], 
        filtered_movie_master_dataset['rating'], 
        color='magenta', alpha=0.1, label='Individual Ratings', s=30
    )
    axes[1].set_title('Ratings per Movie', fontsize=18, color='white')
    axes[1].set_xlabel('Year', fontsize=18, color='white')
    axes[1].set_ylabel('Rating', fontsize=18, color='white')
    axes[1].legend(fontsize=14, loc='upper left')
    axes[1].grid(True, color='gray')
    axes[1].set_ylim(0, 10)  # Set y-axis limits

    # Set tick parameters for both axes
    for ax in axes:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    plt.show()
