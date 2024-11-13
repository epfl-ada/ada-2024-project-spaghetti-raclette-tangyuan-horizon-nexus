# visualize_single_movie.py

import os
import json
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
def plot_single_movie_sentiment(movie_id, analyzed_data):
    if movie_id in analyzed_data:
        sentiment_scores = analyzed_data[movie_id].get('trajectory', [])
        
        plt.figure(figsize=(10, 5))
        plt.plot(sentiment_scores, marker='o', color='b')
        plt.title(f'Sentiment by Sentence for Movie ID: {movie_id}')
        plt.xlabel('Sentence Number')
        plt.ylabel('Sentiment Score')
        plt.axhline(0, color='grey', linestyle='--')  # Neutral sentiment line
        plt.tight_layout()
        plt.show()
    else:
        print(f"Movie ID {movie_id} not found in analyzed data.")

# Main function to load and plot
def main(data_directory, movie_id):
    # Load analyzed data
    analyzed_data = load_analyzed_data(data_directory)

    # Plot sentiment for a single movie
    plot_single_movie_sentiment(movie_id, analyzed_data)

# Set up data directory and specify the movie ID to plot
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, '..', 'Data')
movie_id_to_plot = '31186339'  # Replace with the movie ID you'd like to view

# Execute main function
main(data_directory, movie_id=movie_id_to_plot)
