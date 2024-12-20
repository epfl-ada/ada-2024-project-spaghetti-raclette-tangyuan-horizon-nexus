import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.stats import f_oneway
import os


def analyze_narrative_types_with_clustering(movie_master_dataset):
    """
    Perform clustering on sentiment arcs for all movies, assign narrative types, and perform ANOVA testing.
    
    Args:
    - movie_master_dataset (DataFrame): Main dataset containing movie metadata.

    Returns:
    - movie_master_with_clusters (DataFrame): Updated movie master dataset with assigned clusters and narrative types.
    """

    # Load VADER sentiment analysis results
    vader_sentiment_path = os.path.join('data/sentence_sentimental_analysis_Vader.csv')
    vader_df = pd.read_csv(vader_sentiment_path)

    # Prepare data for clustering
    vader_df['genres'] = vader_df['genres'].fillna('')
    vader_df['genres'] = vader_df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    exploded_df = vader_df.explode('genres')

    def normalize_sentiment_arc(sentiments, num_points=200):
        x = np.linspace(0, len(sentiments) - 1, num_points)
        return np.interp(x, np.arange(len(sentiments)), sentiments)

    def prepare_data_for_clustering_all(exploded_df, normalize_arc_length=200):
        normalized_arcs = []
        movie_ids = []

        for _, row in exploded_df.iterrows():
            try:
                movie_sentiments = eval(row['sentence_sentiments'])
                compound_scores = [sentiment['compound'] for sentiment in movie_sentiments]
                normalized_arc = normalize_sentiment_arc(compound_scores, num_points=normalize_arc_length)
                normalized_arcs.append(normalized_arc)
                movie_ids.append(row['movie_id'])
            except Exception:
                continue

        return np.array(normalized_arcs), movie_ids

    print("Preparing data for clustering...")
    normalized_arcs, movie_ids = prepare_data_for_clustering_all(exploded_df)

    def cluster_all_movies(normalized_arcs, n_clusters=6):
        clustering_data = normalized_arcs.reshape((normalized_arcs.shape[0], normalized_arcs.shape[1], 1))
        scaler = TimeSeriesScalerMeanVariance()
        clustering_data_scaled = scaler.fit_transform(clustering_data)

        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=42)
        cluster_labels = kmeans.fit_predict(clustering_data_scaled)
        barycenters = kmeans.cluster_centers_

        plt.figure(figsize=(10, 6))
        cluster_labels_mapping = {
            1: "Man in Hole", 2: "Rags to Riches", 3: "Riches to Rags",
            4: "Cinderella", 5: "Icarus", 6: "Oedipus"
        }

        for i, barycenter in enumerate(barycenters):
            plt.plot(barycenter.ravel(), label=f"{cluster_labels_mapping[i+1]} (Cluster {i+1})", linewidth=2)
        plt.title("Representative Story Arcs for All Movies")
        plt.xlabel("Normalized Sentence Index")
        plt.ylabel("Sentiment Score")
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

        return cluster_labels, barycenters

    print("Clustering all movies...")
    cluster_labels, barycenters = cluster_all_movies(normalized_arcs, n_clusters=6)

    print("Assigning cluster labels to movies...")
    movie_clusters_df = pd.DataFrame({"movie_id": movie_ids, "cluster": cluster_labels})

    movie_master_with_clusters = movie_master_dataset.merge(movie_clusters_df, on="movie_id", how="left")

    def map_narrative_type(row):
        cluster_labels_mapping = {
            1: "Man in Hole", 2: "Rags to Riches", 3: "Riches to Rags",
            4: "Cinderella", 5: "Icarus", 6: "Oedipus"
        }
        if pd.isna(row["cluster"]):
            return None
        return cluster_labels_mapping.get(row["cluster"] + 1, f"Cluster {int(row['cluster'])}")

    movie_master_with_clusters["narrative_type"] = movie_master_with_clusters.apply(map_narrative_type, axis=1)
    movie_master_with_clusters.dropna(subset=["narrative_type"], inplace=True)

    print("Calculating average success by narrative type...")
    cluster_success_summary = movie_master_with_clusters.groupby("narrative_type")["success"].mean().reset_index()
    cluster_success_summary.rename(columns={"success": "average_success"}, inplace=True)

    plt.figure(figsize=(10, 6))
    plt.bar(cluster_success_summary["narrative_type"], cluster_success_summary["average_success"], 
            color="skyblue", edgecolor="black")
    plt.title("Average Success by Narrative Type for All Movies")
    plt.xlabel("Narrative Type")
    plt.ylabel("Average Success")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Cluster Success Summary with Narrative Types:")
    print(cluster_success_summary)

    def perform_anova(movie_master_with_clusters):
        narrative_groups = [
            movie_master_with_clusters[movie_master_with_clusters["narrative_type"] == narrative]["success"]
            for narrative in ["Man in Hole", "Rags to Riches", "Riches to Rags", "Cinderella", "Icarus", "Oedipus"]
        ]
        f_stat, p_value = f_oneway(*narrative_groups)
        return f_stat, p_value

    f_stat, p_value = perform_anova(movie_master_with_clusters)
    print("\nANOVA Results for Narrative Types:")
    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value:.4e}")

    if p_value < 0.05:
        print("There is a statistically significant difference in success across narrative types.")
    else:
        print("There is no statistically significant difference in success across narrative types.")

    return movie_master_with_clusters
