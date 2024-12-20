import pandas as pd
import numpy as np
import os
import networkx as nx
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess
from tests.Network_Analysis.network_founctions_v21 import *
# Create directory for saving HTML plots
os.makedirs('html_plots', exist_ok=True)

# Update the main analysis function
def analyze_best_actor_fame(movie_master_dataset):
    """
    Analyze best actor fame based on network analysis and update the movie_master_dataset.

    Args:
        movie_master_dataset (DataFrame): Main dataset containing movie metadata.

    Returns:
        DataFrame: Updated movie_master_dataset with best actor fame.
    """
    # Load and preprocess data
    net_movie_df, net_character_df = load_data()
    merged_data = preprocess_data(net_movie_df, net_character_df)
    print('here1')
    # Build the graph
    G = build_graph(merged_data)
    print('here2')
    # Extract top actors and compute average success
    G_sub = get_top_actors(G, top_n=60)
    actor_avg_success = compute_actor_avg_success(merged_data, min_movies=5, top_success=10)

    # Update graph with attributes
    G_sub = update_graph_with_attributes(G_sub, actor_avg_success)
    G_sub = normalize_attributes(G_sub)
    pos = generate_positions(G_sub)

    # Create network visualization
    edge_trace, node_trace = prepare_plotly_traces(G_sub, pos)
    print('here3')
    visualize_network_plotly(edge_trace, node_trace)

    # Collect degree centrality and success
    degrees_all, avg_successes_all = collect_degrees_and_success(G, actor_avg_success)

    # Filter and smooth data
    degrees_all, avg_successes_all = filter_data(degrees_all, avg_successes_all)
    degrees_fit, avg_successes_fit = apply_lowess(degrees_all, avg_successes_all, frac=0.3)

    # Generate and save statistics plots
    print('here4')
    plot_matplotlib_statistics(degrees_all, avg_successes_all, degrees_fit, avg_successes_fit)

    # Compute Best Actor Fame
    actors_list = list(G.nodes)
    fame_scores = dict(zip(actors_list, degrees_all))

    # Find best actor per movie
    movie_best_actor = {
        movie: max(
            {actor: fame_scores.get(actor, 0) for actor in merged_data[merged_data['movie_name'] == movie]['actor_name']}.items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]
        for movie in net_movie_df['movie_name'].unique()
    }

    # Create a DataFrame for best actor fame
    movie_fame = pd.DataFrame({
        "movie_name": list(movie_best_actor.keys()),
        "best_actor_fame": [actor_avg_success.get(actor, None) for actor in movie_best_actor.values()]
    })

    # Merge with main dataset
    movie_master_dataset = pd.merge(movie_master_dataset, movie_fame, on='movie_name', how='left')
    movie_master_dataset = movie_master_dataset.drop_duplicates(subset='movie_id')

    # Create histogram of best actor fame
    fig = go.Figure(data=[go.Histogram(
        x=movie_master_dataset['best_actor_fame'].dropna(),
        marker_color='#FF7E1D', opacity=0.75
    )])

    fig.update_layout(
        title="Distribution of Best Actor Fame Across Movies",
        xaxis_title="Best Actor Fame",
        yaxis_title="Number of Movies",
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )

    fig.show()
    fig.write_html('html_plots/best_actor_fame_distribution.html')

    return movie_master_dataset
