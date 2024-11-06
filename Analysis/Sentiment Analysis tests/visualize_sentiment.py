# visualize_sentiment.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Load analyzed sentiment data
def load_analyzed_data(data_directory):
    file_path = os.path.join(data_directory, 'analyzed_sentiment_data.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file 'analyzed_sentiment_data.json' was not found in the directory: {data_directory}")
    with open(file_path, 'r') as f:
        analyzed_data = json.load(f)
    return analyzed_data

# Plot sentiment trajectory for a specific movie
def plot_sentiment_trajectory(movie_id, analyzed_data):
    if movie_id in analyzed_data:
        sentiment_scores = analyzed_data[movie_id]['trajectory']
        
        plt.figure(figsize=(12, 6))
        plt.plot(sentiment_scores, marker='o', color='b')
        plt.title(f'Sentiment Trajectory for Movie ID: {movie_id}')
        plt.xlabel('Sentence Index')
        plt.ylabel('Sentiment Score')
        plt.axhline(0, color='grey', linestyle='--')  # Neutral sentiment line
        plt.show()
    else:
        print(f"Movie ID {movie_id} not found in analyzed data.")

# Aggregate sentiment scores across all movies to get a distribution
def plot_sentiment_distribution(analyzed_data):
    avg_sentiments = [info['average_sentiment'] for info in analyzed_data.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(avg_sentiments, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Average Sentiment Scores Across Movies")
    plt.xlabel("Average Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

# Boxplot of sentiment variability
def plot_sentiment_variability(analyzed_data, num_movies=10):
    sample_movies = list(analyzed_data.keys())[:num_movies]
    sentiment_trajectories = [analyzed_data[movie_id]['trajectory'] for movie_id in sample_movies]
    
    plt.figure(figsize=(12, 8))
    plt.boxplot(sentiment_trajectories, vert=False, patch_artist=True)
    plt.yticks(range(1, len(sample_movies) + 1), sample_movies)
    plt.xlabel("Sentiment Score")
    plt.title("Sentiment Variability Across Sample Movies")
    plt.show()

# Main function for loading and visualizing data
def main(data_directory, movie_id=None):
    # Load analyzed data
    analyzed_data = load_analyzed_data(data_directory)

    # Individual movie plot (if movie_id is specified)
    if movie_id:
        plot_sentiment_trajectory(movie_id, analyzed_data)

    # Distribution of average sentiment scores across all movies
    plot_sentiment_distribution(analyzed_data)

    # Boxplot for variability across sample movies
    plot_sentiment_variability(analyzed_data)

# Set up the data directory and movie_id to visualize
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, '..', 'Data')
movie_id_to_visualize = '31186339'  # Replace with the movie_id you'd like to view

# Visualize the results
main(data_directory, movie_id=movie_id_to_visualize)
