import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_revenue_statistics(movie_master_dataset):
    """
    Plot box office revenue statistics and individual movie revenues on a log scale.

    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata including 'release_date' and 'revenue'.
    """
    # Define background color
    background_color = '#1e1e1e'

    # Extract the release year
    movie_master_dataset['release_year'] = pd.to_datetime(
        movie_master_dataset['release_date'], errors='coerce'
    ).dt.year

    # Group by release year and calculate revenue statistics
    revenue_stats = movie_master_dataset.groupby('release_year')['revenue'].agg(['mean', 'std', 'median'])
    df_cleaned = movie_master_dataset[['release_year', 'revenue']].dropna()

    # Create the plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 9))
    fig.patch.set_facecolor(background_color)
    for ax in axes:
        ax.set_facecolor(background_color)

    # Main title
    fig.suptitle('Box Office Revenue Analysis', fontsize=24, color='white')

    # Plot 1: Revenue statistics
    axes[0].fill_between(
        revenue_stats.index, 
        revenue_stats['mean'] - revenue_stats['std'],
        revenue_stats['mean'] + revenue_stats['std'], 
        color='white', alpha=0.2, label='1 Std Dev'
    )
    axes[0].plot(
        revenue_stats.index, 
        revenue_stats['median'], 
        label='Median Revenue', 
        linestyle='--', color='white', linewidth=3
    )
    axes[0].plot(
        revenue_stats.index, 
        revenue_stats['mean'], 
        label='Mean Revenue', 
        color='lightgreen', linewidth=3
    )
    axes[0].set_title('Box Office Revenue Statistics', fontsize=18, color='white')
    axes[0].set_xlabel('Year', fontsize=18, color='white')
    axes[0].set_ylabel('Revenue [$]', fontsize=18, color='white')
    axes[0].legend(fontsize=14)
    axes[0].grid(True, color='gray')

    # Plot 2: Individual movie revenues vs mean revenue with log scale
    axes[1].plot(
        revenue_stats.index, 
        revenue_stats['mean'], 
        label='Mean Revenue', 
        color='lightgreen', linewidth=4
    )
    axes[1].scatter(
        df_cleaned['release_year'], 
        df_cleaned['revenue'], 
        color='magenta', alpha=0.1, label='Individual Revenues', s=30
    )
    axes[1].set_title('Box Office Revenue per Movie (log)', fontsize=18, color='white')
    axes[1].set_xlabel('Year', fontsize=18, color='white')
    axes[1].set_ylabel('Revenue [$] (log)', fontsize=18, color='white')
    axes[1].set_yscale('log')  # Set y-axis to log scale
    axes[1].legend(fontsize=14)
    axes[1].grid(True, color='gray')

    # Set tick parameters for both axes
    for ax in axes:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    plt.show()
