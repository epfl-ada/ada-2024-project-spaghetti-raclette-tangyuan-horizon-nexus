import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_cluster_narratives(cluster_results):
    """
    Analyze and visualize cluster sizes across genres with narrative types.

    Args:
    - cluster_results (dict): Dictionary containing cluster labels and barycenters for each genre.
    """
    # Define a mapping of clusters to narrative types (Kurt Vonnegut's labels)
    cluster_labels_mapping = {
        "Action": {
            "Cluster 1": "Cinderella",
            "Cluster 2": "Oedipus",
            "Cluster 3": "Man in a Hole",
            "Cluster 4": "Icarus"
        },
        "Horror": {
            "Cluster 1": "Fall from Grace",
            "Cluster 2": "Man in a Hole",
            "Cluster 3": "Oedipus",
            "Cluster 4": "Icarus"
        },
        "Drama": {
            "Cluster 1": "Man in a Hole",
            "Cluster 2": "Cinderella",
            "Cluster 3": "Oedipus",
            "Cluster 4": "Fall from Grace"
        },
        "Comedy": {
            "Cluster 1": "Man in a Hole",
            "Cluster 2": "Rags to Riches",
            "Cluster 3": "Oedipus",
            "Cluster 4": "Icarus"
        }
    }

    # Prepare data for the stacked bar chart
    cluster_summary = []

    for genre, result in cluster_results.items():
        cluster_labels = result["labels"]
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            cluster_name = cluster_labels_mapping[genre].get(f"Cluster {cluster_id + 1}", f"Cluster {cluster_id + 1}")
            cluster_summary.append({
                "Genre": genre,
                "Cluster": cluster_name,
                "Count": count
            })

    cluster_summary_df = pd.DataFrame(cluster_summary)

    # Pivot the data to prepare for the stacked bar chart
    pivot_table = cluster_summary_df.pivot(index="Genre", columns="Cluster", values="Count").fillna(0)

    # Plot the stacked bar chart
    plt.figure(figsize=(12, 8))
    pivot_table.plot(kind="bar", stacked=True, figsize=(12, 8), cmap="tab10")

    # Customize the chart
    plt.title("Cluster Sizes Across Genres with Narrative Types")
    plt.xlabel("Movie Genre")
    plt.ylabel("Number of Movies")
    plt.legend(title="Narrative Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Show the chart
    plt.show()

    # Display the pivot table for reference
    print("Cluster summary table with narrative types:")
    print(pivot_table)
