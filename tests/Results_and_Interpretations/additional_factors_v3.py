import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import yfinance as yf

# Dictionary: language -> number of speakers (in millions)
language_population = {
    'English': 1515, 'Mandarin': 1140, 'Hindi': 609, 'Spanish': 560, 
    'French': 309, 'Arabic': 332, 'Bengali': 278, 'Portuguese': 264, 
    'Russian': 255, 'Urdu': 238, 'Indonesian': 199, 'German': 138, 
    'Japanese': 123, 'Turkish': 90, 'Vietnamese': 86, 'Korean': 81, 
    'Italien': 67, 'Thai': 61
}

# Create HTML saving directory
os.makedirs('html_plots', exist_ok=True)


### 1. Add Language & Country Features and Plot ###
def add_language_country_features_and_plot(movie_master_dataset):
    movie_master_dataset['exposure'] = movie_master_dataset['languages'].apply(
        lambda x: sum(language_population.get(lang.strip(), 0) for lang in x.split(', ')) if isinstance(x, str) else 0)
    movie_master_dataset['num_genres'] = movie_master_dataset['genres'].apply(lambda x: len(x.split(',')))
    movie_master_dataset['num_languages'] = movie_master_dataset['languages'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    movie_master_dataset['num_countries'] = movie_master_dataset['countries'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else 0)

    fig = go.Figure(data=go.Histogram(
        x=movie_master_dataset['exposure'].dropna(),
        marker_color='#FF7E1D',
    ))

    fig.update_layout(
        title='Histogram of Movie Exposure',
        xaxis_title='Exposure (Language Reach)',
        yaxis_title='Frequency',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    fig.show()
    fig.write_html('html_plots/histogram_movie_exposure.html')
    return movie_master_dataset


### 2. Add Holiday Feature and Plot ###
def add_holiday_feature_and_plot(movie_master_dataset):
    movie_master_dataset['release_month'] = pd.to_datetime(movie_master_dataset['release_date']).dt.month
    movie_master_dataset['holiday_release'] = movie_master_dataset['release_month'].apply(
        lambda x: 1 if x in [11, 12, 7] else 0)

    fig = go.Figure(data=go.Bar(
        x=['Non-Holiday', 'Holiday'],
        y=movie_master_dataset.groupby('holiday_release')['success'].mean(),
        marker_color='#FF7E1D'
    ))

    fig.update_layout(
        title='Success by Holiday Release',
        xaxis_title='Release Type',
        yaxis_title='Average Success',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    fig.show()
    fig.write_html('html_plots/success_by_holiday_release.html')
    return movie_master_dataset


### 3. Add Genre Popularity and Plot ###
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

    fig = go.Figure(data=go.Histogram(
        x=movie_master_dataset['genre_popularity'].dropna(),
        marker_color='#FF7E1D',
    ))

    fig.update_layout(
        title='Histogram of Weighted Genre Popularity',
        xaxis_title='Weighted Genre Popularity',
        yaxis_title='Frequency',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    fig.show()
    fig.write_html('html_plots/histogram_genre_popularity.html')
    return movie_master_dataset


### 4. Add S&P 500 Returns and Plot ###
def add_sp_returns_and_plot(movie_master_dataset):
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(period="max")

    sp500_data['Year'] = sp500_data.index.year
    yearly_data = sp500_data.resample('YE').last()
    yearly_data['Year'] = yearly_data.index.year
    yearly_data['Return'] = yearly_data['Close'].pct_change()

    sp_returns = yearly_data.set_index('Year')['Return'].to_dict()
    movie_master_dataset['release_year'] = pd.to_datetime(movie_master_dataset['release_date'], errors='coerce').dt.year
    movie_master_dataset['SP_return'] = movie_master_dataset['release_year'].map(sp_returns)

    fig = go.Figure(data=go.Histogram(
        x=movie_master_dataset['SP_return'].dropna(),
        marker_color='#FF7E1D',
    ))

    fig.update_layout(
        title='Histogram of S&P 500 Returns',
        xaxis_title='S&P 500 Return',
        yaxis_title='Frequency',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    fig.show()
    fig.write_html('html_plots/histogram_sp_returns.html')
    return movie_master_dataset
