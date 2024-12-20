import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tests.Network_Analysis.network_founctions_v6 import *
from statsmodels.nonparametric.smoothers_lowess import lowess
from tests.Network_Analysis.network_big_plot_v3 import analyze_best_actor_fame

def perform_lowess_analysis(movie_master_dataset):
    """
    Performs LOWESS analysis on actor degree vs average success 
    and updates the movie_master_dataset.

    Args:
        movie_master_dataset (DataFrame): Main movie metadata dataset.

    Returns:
        DataFrame: Updated movie_master_dataset after LOWESS analysis.
    """
    # Analyze best actor fame
    movie_master_dataset = analyze_best_actor_fame(movie_master_dataset)

    # Build and preprocess network data
    net_movie_df, net_character_df = load_data()
    merged_data = preprocess_data(net_movie_df, net_character_df)
    G = build_graph(merged_data)

    # Collect degrees and average successes
    degrees_all, avg_successes_all = collect_degrees_and_success(G, compute_actor_avg_success(merged_data))

    # Filter and apply LOWESS
    movie_master_dataset = movie_master_dataset.dropna()
    degrees_all, avg_successes_all = filter_data(degrees_all, avg_successes_all)
    degrees_fit, avg_successes_fit = apply_lowess(degrees_all, avg_successes_all, frac=0.3)

    # Create figure
    plot_matplotlib_statistics(degrees_all, avg_successes_all, degrees_fit, avg_successes_fit)

    return movie_master_dataset
