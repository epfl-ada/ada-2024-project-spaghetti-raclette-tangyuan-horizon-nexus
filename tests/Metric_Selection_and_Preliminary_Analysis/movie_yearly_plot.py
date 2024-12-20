
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


def plot_total_revenue(movie_master_dataset):
    """
    Plot the total yearly box office revenue from 1920 onward.

    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata including 'release_date' and 'revenue'.
    """
    # Compute log-revenue to avoid log(0)
    movie_master_dataset['log_revenue'] = np.log(
        movie_master_dataset['revenue'].replace(0, np.nan)
    )

    # Set up the background color for the plot
    plt.style.use('dark_background')
    background_color = '#1e1e1e'

    # Extract the release year from release_date
    movie_master_dataset['release_year'] = pd.to_datetime(
        movie_master_dataset['release_date'], errors='coerce'
    ).dt.year

    # Group by release year for total revenue and filter for years 1920 onwards
    total_revenue_per_year = movie_master_dataset.groupby('release_year')['revenue'].sum()
    total_revenue_per_year = total_revenue_per_year[total_revenue_per_year.index >= 1920]

    # Plot the total yearly box office revenue
    fig, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(background_color)

    ax2.set_facecolor(background_color)
    ax2.plot(
        total_revenue_per_year.index, 
        total_revenue_per_year.values, 
        color='lightgreen', 
        linewidth=2, 
        label='Total Box Office Revenue'
    )
    ax2.set_title('Total Yearly Box Office Revenue (1920+)', fontsize=16, color='white')
    ax2.set_xlabel('Year', fontsize=14, color='white')
    ax2.set_ylabel('Total Box Office Revenue [$] (log)', fontsize=14, color='white')
    ax2.set_yscale('log')
    ax2.grid(True, color='gray')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.legend(loc='upper left', fontsize=12)

    plt.show()