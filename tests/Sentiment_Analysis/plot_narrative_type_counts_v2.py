import pandas as pd
import os
import plotly.graph_objects as go

def plot_narrative_type_counts(movie_master_with_clusters):
    """
    Count and plot the number of movies per narrative type as an interactive HTML plot.

    Parameters:
    - movie_master_with_clusters (DataFrame): Updated movie dataset with narrative types.

    Returns:
    - None (Displays and saves the bar chart as HTML)
    """
    # Count the number of movies in each narrative type
    print("Counting the number of movies per narrative type...")
    narrative_type_counts = movie_master_with_clusters["narrative_type"].value_counts().reset_index()
    narrative_type_counts.columns = ["narrative_type", "movie_count"]

    # Display the counts
    print("Number of Movies per Narrative Type:")
    print(narrative_type_counts)

    # Create a bar chart using Plotly
    print("Plotting the number of movies per narrative type...")
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=narrative_type_counts["narrative_type"],
        y=narrative_type_counts["movie_count"],
        marker_color='#FF7E1D',  # Consistent orange color for all bars
    ))

    fig.update_layout(
        title="Number of Movies per Narrative Type",
        xaxis_title="Narrative Type",
        yaxis_title="Number of Movies",
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        xaxis=dict(tickangle=45)  # Rotate x-axis labels for better readability
    )

    # Display the plot in the notebook
    fig.show()

    # Save the plot as an HTML file
    os.makedirs('html_plots', exist_ok=True)
    fig.write_html('html_plots/number_of_movies_by_narrative_type.html')
