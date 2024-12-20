import pandas as pd
import matplotlib.pyplot as plt

def plot_actor_occurrences_vs_age(character_metadata_df, background_color='#1e1e1e'):
    """
    Plot the total number of occurrences of an actor vs. the youngest age at first occurrence.
    
    Parameters:
    - character_metadata_df (DataFrame): The dataset containing actor names and ages.
    - background_color (str): Background color for the plot (default is dark).
    """
    # Calculate the total number of occurrences of each actor
    actor_occurrences = character_metadata_df['actor_name'].value_counts()

    # Calculate the youngest age at which each actor first appeared
    youngest_age_first_occurrence = character_metadata_df.groupby('actor_name')['actor_age'].min()

    # Merge the two series into a DataFrame
    actor_stats = pd.DataFrame({
        'total_occurrences': actor_occurrences,
        'youngest_age_first_occurrence': youngest_age_first_occurrence
    }).dropna()

    # Calculate the mean occurrences for each youngest age at first occurrence
    mean_occurrences_per_age = actor_stats.groupby('youngest_age_first_occurrence')['total_occurrences'].mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Scatter plot
    ax.scatter(actor_stats['youngest_age_first_occurrence'], actor_stats['total_occurrences'], 
               alpha=0.02, color='magenta', s=70)

    # Plot the mean occurrences per age
    ax.plot(mean_occurrences_per_age.index, mean_occurrences_per_age.values, 
            color='cyan', linewidth=2, label='Mean Occurrences per Age')

    # Add reference vertical lines
    critical_ages = [1, 3, 5, 15, 17, 19]
    line_styles = ['--', '-', '--', '--', '-', '--']
    line_colors = ['white', 'yellow', 'white', 'white', 'yellow', 'white']

    for age, style, color in zip(critical_ages, line_styles, line_colors):
        ax.axvline(x=age, color=color, linestyle=style, linewidth=0.5 if style == '--' else 1)

    # Plot settings
    ax.set_title('Total Number of Occurrences of an Actor vs. Youngest Age at First Occurrence', 
                 fontsize=16, color='white')
    ax.set_xlabel('Youngest Age at First Occurrence', fontsize=11, color='white')
    ax.set_ylabel('Total Number of Occurrences', fontsize=11, color='white')
    ax.set_yscale('log')
    ax.grid(True, color='gray')
    ax.legend()

    # Set tick parameters
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.show()
