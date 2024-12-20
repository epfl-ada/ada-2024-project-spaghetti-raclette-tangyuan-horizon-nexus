import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Dictionary: language -> number of speakers (in millions)
language_population = {
    'English': 1515, 'Mandarin': 1140, 'Hindi': 609, 'Spanish': 560, 
    'French': 309, 'Arabic': 332, 'Bengali': 278, 'Portuguese': 264, 
    'Russian': 255, 'Urdu': 238, 'Indonesian': 199, 'German': 138, 
    'Japanese': 123, 'Turkish': 90, 'Vietnamese': 86, 'Korean': 81, 
    'Italien': 67, 'Thai': 61
}

# Add Language & Country Features and Plot
def add_language_country_features_and_plot(movie_master_dataset):
    movie_master_dataset['exposure'] = movie_master_dataset['languages'].apply(
        lambda x: sum(language_population.get(lang.strip(), 0) for lang in x.split(', ')) if isinstance(x, str) else 0)
    movie_master_dataset['num_genres'] = movie_master_dataset['genres'].apply(lambda x: len(x.split(',')))
    movie_master_dataset['num_languages'] = movie_master_dataset['languages'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    movie_master_dataset['num_countries'] = movie_master_dataset['countries'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    
    # Plot exposure
    plt.figure(figsize=(14, 8))
    plt.hist(movie_master_dataset['exposure'].dropna(), bins='auto', color='skyblue', edgecolor='black')
    plt.title('Histogram of Movie Exposure', fontsize=16)
    plt.xlabel('Exposure (Language Reach)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()
    return movie_master_dataset

# Add Holiday Feature and Plot
def add_holiday_feature_and_plot(movie_master_dataset):
    movie_master_dataset['release_month'] = pd.to_datetime(movie_master_dataset['release_date']).dt.month
    movie_master_dataset['holiday_release'] = movie_master_dataset['release_month'].apply(
        lambda x: 1 if x in [11, 12, 7] else 0)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=movie_master_dataset, x='holiday_release', y='success', errorbar=None)
    plt.title('Success by Holiday Release')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.show()
    return movie_master_dataset

# Add Genre Popularity and Plot
def add_genre_popularity_and_plot(movie_master_dataset):
    genre_success = movie_master_dataset.explode('genres')[['genres', 'success']].groupby('genres').mean()

    def assign_primary_genre(genres):
        if not isinstance(genres, str):
            return None
        genre_list = genres.split(',')
        genre_popularities = {genre.strip(): genre_success.loc[genre.strip(), 'success']
                              for genre in genre_list if genre.strip() in genre_success.index}
        return max(genre_popularities, key=genre_popularities.get) if genre_popularities else None

    def calculate_weighted_genre_popularity(genres):
        if not isinstance(genres, str):
            return np.nan
        genre_list = genres.split(',')
        genre_popularities = [genre_success.loc[genre.strip(), 'success']
                              for genre in genre_list if genre.strip() in genre_success.index]
        return sum(genre_popularities) / len(genre_popularities) if genre_popularities else np.nan

    movie_master_dataset['primary_genre'] = movie_master_dataset['genres'].apply(assign_primary_genre)
    movie_master_dataset['genre_popularity'] = movie_master_dataset['genres'].apply(calculate_weighted_genre_popularity)
    
    plt.figure(figsize=(14, 8))
    plt.hist(movie_master_dataset['genre_popularity'].dropna(), bins='auto', color='skyblue', edgecolor='black')
    plt.title('Histogram of Weighted Genre Popularity', fontsize=16)
    plt.xlabel('Weighted Genre Popularity', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()
    return movie_master_dataset

def add_sp_returns_and_plot(movie_master_dataset):
    """
    Add S&P 500 returns to the movie_master_dataset and plot the histogram.

    Args:
        movie_master_dataset (DataFrame): The main movie dataset.

    Returns:
        DataFrame: Updated movie_master_dataset with S&P 500 returns added.
    """
    # Fetch S&P 500 historical data
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(period="max")  # Fetch max available data

    # Resample to yearly frequency and compute yearly returns
    sp500_data['Year'] = sp500_data.index.year
    yearly_data = sp500_data.resample('YE').last()  # Take the last available price for each year
    yearly_data['Year'] = yearly_data.index.year
    yearly_data['Return'] = yearly_data['Close'].pct_change()

    # Create a dictionary of S&P 500 returns
    sp_returns = yearly_data.set_index('Year')['Return'].to_dict()

    # Ensure release_date is in datetime format and extract the year
    movie_master_dataset['release_year'] = pd.to_datetime(movie_master_dataset['release_date'], errors='coerce').dt.year

    # Map the S&P 500 returns to the release year
    movie_master_dataset['SP_return'] = movie_master_dataset['release_year'].map(sp_returns)

    # Plot the histogram
    plt.figure(figsize=(14, 8))
    plt.hist(movie_master_dataset['SP_return'].dropna(), bins='auto', color='skyblue', edgecolor='black')
    plt.title('Histogram of S&P 500 Returns', fontsize=16)
    plt.xlabel('S&P 500 Return', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    return movie_master_dataset

