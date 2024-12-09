import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def load_data(movie_path='data/movie_metadata_cleaned.csv', character_path='data/character_metadata_cleaned.csv'):
    net_movie_df = pd.read_csv(movie_path)
    net_character_df = pd.read_csv(character_path)
    return net_movie_df, net_character_df

def preprocess_data(net_movie_df, net_character_df):
    merged_data = pd.merge(net_character_df, net_movie_df, on='movie_id', how='inner')
    merged_data = merged_data[merged_data['actor_name'].notna()]
    merged_data = merged_data[merged_data['actor_name'] != '']
    merged_data['success'] = np.where(
        merged_data['rating'].notna(),
        merged_data['rating'] * np.log(merged_data['vote_count'] + 1),
        np.nan
    )
    return merged_data

def build_graph(merged_data):
    G = nx.Graph()
    movie_actor_group = merged_data.groupby('movie_id')['actor_name'].apply(list)
    for actors in movie_actor_group:
        unique_actors = list(set(actors))
        for i in range(len(unique_actors)):
            for j in range(i + 1, len(unique_actors)):
                G.add_edge(unique_actors[i], unique_actors[j])
    return G

def get_top_actors(G, top_n=100):
    degree_dict = dict(G.degree())
    nx.set_node_attributes(G, degree_dict, 'degree')
    top_actors = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_n]
    G_sub = G.subgraph(top_actors).copy()
    return G_sub

def compute_actor_avg_success(merged_data, min_movies=5, top_success=10):
    actor_movie_success = merged_data.groupby('actor_name')['success'].apply(list)
    actor_avg_success = {}
    for actor, success_list in actor_movie_success.items():
        success_list = [s for s in success_list if not pd.isna(s)]
        if len(success_list) >= min_movies:
            avg_success = np.mean(sorted(success_list, reverse=True)[:top_success])
            actor_avg_success[actor] = avg_success
    return actor_avg_success

def update_graph_with_attributes(G_sub, actor_avg_success):
    for node in G_sub.nodes():
        G_sub.nodes[node]['avg_success'] = actor_avg_success.get(node, None)
        G_sub.nodes[node]['degree'] = G_sub.degree[node]
    nodes_to_remove = [node for node, attr in G_sub.nodes(data=True) if attr['avg_success'] is None]
    G_sub.remove_nodes_from(nodes_to_remove)
    nodes_to_remove = [node for node, degree in G_sub.degree() if degree == 0]
    G_sub.remove_nodes_from(nodes_to_remove)
    return G_sub

def normalize_attributes(G_sub):
    avg_successes = [attr['avg_success'] for _, attr in G_sub.nodes(data=True)]
    min_success = min(avg_successes)
    max_success = max(avg_successes)
    for node in G_sub.nodes():
        avg_success = G_sub.nodes[node]['avg_success']
        norm_success = (avg_success - min_success) / (max_success - min_success) if max_success != min_success else 0
        G_sub.nodes[node]['norm_success'] = norm_success
    degrees = [G_sub.degree[node] for node in G_sub.nodes()]
    min_degree = min(degrees)
    max_degree = max(degrees)
    for node in G_sub.nodes():
        degree = G_sub.degree[node]
        norm_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree != min_degree else 0
        G_sub.nodes[node]['norm_degree'] = norm_degree
    return G_sub

def generate_positions(G_sub, k=0.5, iterations=50, seed=42):
    pos = nx.spring_layout(G_sub, k=k, iterations=iterations, seed=seed)
    return pos

def prepare_plotly_traces(G_sub, pos):
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_size = []
    node_color = []
    for node in G_sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        size = ((G_sub.nodes[node]['norm_degree']) ** 2 + 0.1) * 100
        node_size.append(size)
        node_color.append(G_sub.nodes[node]['norm_success'])
        node_text.append(f"{node}")
        node_hovertext.append(
            f"Degree: {G_sub.nodes[node]['degree']}<br>Avg Success: {G_sub.nodes[node]['avg_success']:.2f}"
        )
    edge_x = []
    edge_y = []
    for edge in G_sub.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        hovertext=node_hovertext,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Inferno',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Normalized Avg Success',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        textposition='middle center',
        textfont_size=10
    )
    return edge_trace, node_trace

def visualize_network_plotly(edge_trace, node_trace):
    net = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Top Actors Network Graph',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Size: Degree (Higher Contrast), Color: Normalized Avg Success",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    net.show()

def collect_degrees_and_success(G, actor_avg_success):
    degrees_all = []
    avg_successes_all = []
    for node in G.nodes():
        degree = G.degree[node]
        avg_success = actor_avg_success.get(node, None)
        if avg_success is not None:
            degrees_all.append(degree)
            avg_successes_all.append(avg_success)
    degrees_all = np.array(degrees_all)
    avg_successes_all = np.array(avg_successes_all)
    return degrees_all, avg_successes_all

def filter_data(degrees_all, avg_successes_all):
    valid_indices = (
            (~np.isnan(degrees_all)) &
            (~np.isnan(avg_successes_all)) &
            (~np.isinf(degrees_all)) &
            (~np.isinf(avg_successes_all))
    )
    degrees_all = degrees_all[valid_indices]
    avg_successes_all = avg_successes_all[valid_indices]
    mean_success = np.mean(avg_successes_all)
    std_success = np.std(avg_successes_all)
    valid_indices = (
            (avg_successes_all > mean_success - 3 * std_success) &
            (avg_successes_all < mean_success + 3 * std_success)
    )
    degrees_all = degrees_all[valid_indices]
    avg_successes_all = avg_successes_all[valid_indices]
    return degrees_all, avg_successes_all

def apply_lowess(degrees_all, avg_successes_all, frac=0.3):
    sorted_indices = np.argsort(degrees_all)
    degrees_sorted = degrees_all[sorted_indices]
    avg_successes_sorted = avg_successes_all[sorted_indices]
    lowess_smoothed = lowess(avg_successes_sorted, degrees_sorted, frac=frac)
    degrees_fit = lowess_smoothed[:, 0]
    avg_successes_fit = lowess_smoothed[:, 1]
    return degrees_fit, avg_successes_fit

def plot_matplotlib_statistics(degrees_all, avg_successes_all, degrees_fit, avg_successes_fit):
    background_color = '#1e1e1e'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 9))
    fig.patch.set_facecolor(background_color)
    for ax in axes:
        ax.set_facecolor(background_color)
    fig.suptitle('Actor Collaboration and Success Analysis', fontsize=24, color='white')
    axes[0].hist(degrees_all, bins=50, color='lightgreen', alpha=0.7)
    axes[0].set_title('Distribution of Actor Degrees', fontsize=18, color='white')
    axes[0].set_xlabel('Degree', fontsize=18, color='white')
    axes[0].set_ylabel('Number of Actors', fontsize=18, color='white')
    axes[0].grid(True, color='gray')
    axes[0].tick_params(axis='x', colors='white', labelsize=14)
    axes[0].tick_params(axis='y', colors='white', labelsize=14)
    axes[1].scatter(degrees_all, avg_successes_all, color='magenta', alpha=0.7, label='Actors', s=30)
    axes[1].plot(degrees_fit, avg_successes_fit, color='lightgreen', linewidth=3, label='LOWESS Curve')
    axes[1].set_title('Actor Degree vs Average Success', fontsize=18, color='white')
    axes[1].set_xlabel('Degree', fontsize=18, color='white')
    axes[1].set_ylabel('Average Success', fontsize=18, color='white')
    axes[1].legend(fontsize=14)
    axes[1].grid(True, color='gray')
    axes[1].tick_params(axis='x', colors='white', labelsize=14)
    axes[1].tick_params(axis='y', colors='white', labelsize=14)
    plt.show()
