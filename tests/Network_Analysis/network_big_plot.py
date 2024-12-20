import pandas as pd
import matplotlib.pyplot as plt
from tests.Network_Analysis.network_founctions import *

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

    # Build the graph
    G = build_graph(merged_data)

    # Extract top actors and compute average success
    G_sub = get_top_actors(G, top_n=60)
    actor_avg_success = compute_actor_avg_success(merged_data, min_movies=5, top_success=10)

    # Update graph with attributes
    G_sub = update_graph_with_attributes(G_sub, actor_avg_success)
    G_sub = normalize_attributes(G_sub)
    pos = generate_positions(G_sub)

    # Create network and visualize
    edge_trace, node_trace = prepare_plotly_traces(G_sub, pos)
    visualize_network_plotly(edge_trace, node_trace)

    # Collect degree centrality and success
    degrees_all, avg_successes_all = collect_degrees_and_success(G, actor_avg_success)

    # Compute Best Actor Fame
    actors_list = list(G.nodes)  # List of actor names in the graph
    fame_scores = dict(zip(actors_list, degrees_all))

    # Find best actor per movie
    movie_best_actor = {}
    for movie in net_movie_df['movie_name'].unique():
        actors_in_movie = merged_data[merged_data['movie_name'] == movie]['actor_name']
        movie_fame_scores = {actor: fame_scores.get(actor, 0) for actor in actors_in_movie}
        best_actor = max(movie_fame_scores, key=movie_fame_scores.get, default=None)
        movie_best_actor[movie] = best_actor

    # Create a DataFrame for best actor fame
    movie_fame = pd.DataFrame({
        "movie_name": list(movie_best_actor.keys()),
        "best_actor_fame": [actor_avg_success.get(actor, None) for actor in movie_best_actor.values()]
    })

    # Merge with main dataset
    movie_master_dataset = pd.merge(movie_master_dataset, movie_fame, on='movie_name', how='left')

    # Remove duplicates
    movie_master_dataset = movie_master_dataset.drop_duplicates(subset='movie_id')

    # Plot the distribution of best actor fame
    plt.figure(figsize=(14, 8))
    plt.hist(movie_master_dataset['best_actor_fame'].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Best Actor Fame Across Movies', fontsize=16)
    plt.show()

    return movie_master_dataset
