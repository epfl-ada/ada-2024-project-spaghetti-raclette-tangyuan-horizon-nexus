import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def plot_movies_released_yearly(movie_master_dataset):
    """
    Plot the total number of movies released yearly from 1920 onward.
    
    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata including 'release_date'.
    """
    # Set up the background color for the plot
    background_color = '#1e1e1e'
    plt.style.use('dark_background')

    # Extract the release year from release_date
    movie_master_dataset['release_year'] = pd.to_datetime(
        movie_master_dataset['release_date'], errors='coerce'
    ).dt.year

    # Group by release year for the number of movies and apply the cutoff at 1920
    movies_per_year = movie_master_dataset.groupby('release_year').size()
    movies_per_year = movies_per_year[movies_per_year.index >= 1920]

    # Plot the total number of movies released yearly
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(background_color)

    ax1.set_facecolor(background_color)
    ax1.plot(
        movies_per_year.index, 
        movies_per_year.values, 
        color='lightblue', 
        linewidth=2, 
        label='Number of Movies'
    )
    ax1.set_title('Total Number of Movies Released Yearly', fontsize=16, color='white')
    ax1.set_xlabel('Year', fontsize=14, color='white')
    ax1.set_ylabel('Number of Movies', fontsize=14, color='white')
    ax1.grid(True, color='gray')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend(loc='upper left', fontsize=12)

    plt.show()

