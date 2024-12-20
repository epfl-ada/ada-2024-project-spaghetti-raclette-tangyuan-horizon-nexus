import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_vote_count_statistics(movie_master_dataset):
    """
    Plot movie vote count analysis, including yearly statistics and individual vote counts.

    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata including 'release_date' and 'vote_count'.
    """
    # Define background color
    background_color = '#1e1e1e'

    # Ensure the release_date column is in datetime format
    movie_master_dataset['release_date'] = pd.to_datetime(
        movie_master_dataset['release_date'], errors='coerce'
    )

    # Group by year and calculate the statistics for vote_count
    vote_count_stats = movie_master_dataset.groupby(
        movie_master_dataset['release_date'].dt.year
    )['vote_count'].agg(['mean', 'median', 'std', 'min', 'max'])

    # Filter out years before 1920
    vote_count_stats = vote_count_stats[vote_count_stats.index >= 1920]
    movie_master_dataset_filtered = movie_master_dataset[
        movie_master_dataset['release_date'].dt.year >= 1920
    ]

    # Create the plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7), dpi=300)
    fig.patch.set_facecolor(background_color)
    for ax in axes:
        ax.set_facecolor(background_color)

    # Main title
    fig.suptitle('Vote Count Analysis', fontsize=24, color='white')

    # Plot 1: Yearly Vote Count Statistics
    axes[0].fill_between(
        vote_count_stats.index, 
        vote_count_stats['mean'] - vote_count_stats['std'],
        vote_count_stats['mean'] + vote_count_stats['std'], 
        color='white', alpha=0.2, label='1 Std Dev'
    )
    axes[0].plot(
        vote_count_stats.index, 
        vote_count_stats['median'], 
        label='Median Vote Count', 
        linestyle='--', color='white', linewidth=3
    )
    axes[0].plot(
        vote_count_stats.index, 
        vote_count_stats['mean'], 
        label='Mean Vote Count', 
        color='orange', linewidth=3
    )
    axes[0].set_title('Yearly Vote Count Statistics (1920+)', fontsize=18, color='white')
    axes[0].set_xlabel('Year', fontsize=18, color='white')
    axes[0].set_ylabel('Vote Count', fontsize=18, color='white')
    axes[0].legend(fontsize=14, loc='upper left')
    axes[0].grid(True, color='gray')

    # Plot 2: Individual Vote Counts with Mean
    axes[1].plot(
        vote_count_stats.index, 
        vote_count_stats['mean'], 
        label='Mean Vote Count', 
        color='orange', linewidth=3
    )
    axes[1].scatter(
        movie_master_dataset_filtered['release_date'].dt.year, 
        movie_master_dataset_filtered['vote_count'], 
        color='magenta', alpha=0.1, label='Individual Vote Counts', s=30
    )
    axes[1].set_title('Vote Counts per Movie (log, 1920+)', fontsize=18, color='white')
    axes[1].set_xlabel('Year', fontsize=18, color='white')
    axes[1].set_ylabel('Vote Count (log)', fontsize=18, color='white')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=14, loc='upper left')
    axes[1].grid(True, color='gray')

    # Set tick parameters for both axes
    for ax in axes:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    plt.show()
