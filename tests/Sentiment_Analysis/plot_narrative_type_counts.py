import pandas as pd
import matplotlib.pyplot as plt

def plot_narrative_type_counts(movie_master_with_clusters):
    """
    Count and plot the number of movies per narrative type.

    Parameters:
    - movie_master_with_clusters (DataFrame): Updated movie dataset with narrative types.

    Returns:
    - None (Plots the bar chart)
    """
    # Count the number of movies in each narrative type
    print("Counting the number of movies per narrative type...")
    narrative_type_counts = movie_master_with_clusters["narrative_type"].value_counts().reset_index()
    narrative_type_counts.columns = ["narrative_type", "movie_count"]

    # Display the counts
    print("Number of Movies per Narrative Type:")
    print(narrative_type_counts)

    # Plot the counts
    print("Plotting the number of movies per narrative type...")
    plt.figure(figsize=(10, 6))
    plt.bar(narrative_type_counts["narrative_type"], narrative_type_counts["movie_count"], 
            color="lightcoral", edgecolor="black")
    plt.title("Number of Movies per Narrative Type", fontsize=16)
    plt.xlabel("Narrative Type", fontsize=14)
    plt.ylabel("Number of Movies", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
