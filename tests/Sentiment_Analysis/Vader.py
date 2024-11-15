import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load movie master dataset
def load_movie_master_dataset():
    # Adjust the path to navigate to the data directory from the current script's location
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, '../../data')
    file_path = os.path.join(data_directory, 'movie_master_dataset.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file 'movie_master_dataset.csv' was not found in the directory: {data_directory}")
    
    df_movie_master = pd.read_csv(file_path)
    
    # Perform sentiment analysis on the plot_summary column
    df_movie_master['sentiment'] = df_movie_master['plot_summary'].apply(lambda x: analyzer.polarity_scores(str(x)))
    
    return df_movie_master

# Link metadata to sentiment data
def link_metadata(df_movie_master):
    linked_data = []
    for _, row in tqdm(df_movie_master.iterrows(), desc="Linking Metadata", total=len(df_movie_master)):
        genres = row['genres']  # Extract genres as a simple string
        
        linked_data.append({
            'movie_id': row['movie_id'],
            'average_sentiment': row['sentiment']['compound'],
            'neg_sentiment': row['sentiment']['neg'],
            'neu_sentiment': row['sentiment']['neu'],
            'pos_sentiment': row['sentiment']['pos'],
            'genres': genres,
            'revenue': row['revenue'],
            'runtime': row['runtime']
        })
    return pd.DataFrame(linked_data)

# Analyze trends in sentiment by genre
def analyze_genre_trends(linked_data):
    linked_data['average_sentiment'] = pd.to_numeric(linked_data['average_sentiment'], errors='coerce')
    linked_data['neg_sentiment'] = pd.to_numeric(linked_data['neg_sentiment'], errors='coerce')
    linked_data['neu_sentiment'] = pd.to_numeric(linked_data['neu_sentiment'], errors='coerce')
    linked_data['pos_sentiment'] = pd.to_numeric(linked_data['pos_sentiment'], errors='coerce')
    linked_data['revenue'] = pd.to_numeric(linked_data['revenue'], errors='coerce')
    linked_data['runtime'] = pd.to_numeric(linked_data['runtime'], errors='coerce')

    genre_analysis = linked_data.groupby('genres').agg({
        'average_sentiment': 'mean',
        'neg_sentiment': 'mean',
        'neu_sentiment': 'mean',
        'pos_sentiment': 'mean',
        'revenue': 'mean'
    }).sort_values(by='revenue', ascending=False)

    print("\nGenre-Based Analysis:")
    print(genre_analysis)
    return genre_analysis

# Clustering of sentiment trajectories
def perform_clustering(linked_data, num_clusters=4):
    numeric_data = linked_data[['average_sentiment', 'neg_sentiment', 'neu_sentiment', 'pos_sentiment']].dropna()
    filtered_linked_data = linked_data.loc[numeric_data.index]

    features = StandardScaler().fit_transform(numeric_data)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    filtered_linked_data['cluster'] = labels

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustering of Movies by Sentiment Trajectories')
    plt.colorbar(scatter)
    plt.show()

    print("\nCluster Analysis Summary:")
    print(filtered_linked_data.groupby('cluster').mean(numeric_only=True))
    return filtered_linked_data

# Main function to execute the pipeline
def main():
    print("Loading movie master dataset...")
    df_movie_master = load_movie_master_dataset()
    print("Movie master dataset loaded.")

    print("Linking metadata to sentiment data...")
    linked_data = link_metadata(df_movie_master)
    print("Metadata linked to sentiment data.")

    print("Analyzing genre-based sentiment trends...")
    genre_trends = analyze_genre_trends(linked_data)
    print("Genre-based sentiment trend analysis complete.")

    print("Performing clustering on sentiment features...")
    clustered_data = perform_clustering(linked_data)
    print("Clustering on sentiment features complete.")

    print("Saving genre trend and cluster results...")
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, '../../data')
    clustered_data.to_csv(os.path.join(data_directory, 'sentiment_genre_Vader_analysis.csv'), index=False)
    print("Saved genre and cluster analysis results to sentiment_genre_Vader_analysis.csv")

# Run the main function
if __name__ == "__main__":
    main()
