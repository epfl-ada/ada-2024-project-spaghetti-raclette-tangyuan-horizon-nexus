# Stars or Storyline: How Actor Fame and Sentiment Trajectories Shape a Movie's Critical Success

## Abstract

This project aims to uncover the relative influence of two key factors on a movie's critical success: actor fame and the narrative sentiment arc of the plot. We hypothesize that a film's success, as reflected in its ratings, might be influenced not only by the popularity of its cast but also by the emotional journey it offers. To explore this, we employ two distinct approaches: sentiment analysis on plot summaries to assess emotional complexity and patterns, and social network analysis to evaluate actor prominence and connectivity in the industry. Ultimately, this analysis seeks to provide a nuanced understanding of whether a star-studded cast or a compelling emotional narrative contributes more significantly to a film’s acclaim, offering insights into storytelling and casting strategies that could enhance a film's impact.

## Key Questions We Aim to Answer

- How can we evaluate a movie’s success (e.g., audience rating, revenue)?
- How do different sentiment features (e.g., overall positivity, variability, peaks) within movie plots correlate with our success metric?
- Are certain emotional arcs consistently linked to successful movies? How do they align with the six classic narrative archetypes proposed by Kurt Vonnegut?
- How does the presence of well-established actors in a film’s network affect the movie’s critical success?
- Are certain communities of actors who frequently collaborate more impactful for a movie's success?
- Can we identify an ideal combination of sentiment arc and actor influence that predicts higher audience ratings for movies?

## Additional Datasets

To enrich our analysis and fill in any missing data, we incorporated two external data sources for critical information on movie ratings and revenues:

- **Movie Ratings from The Movie Database (TMDb)**: Accessed via the [TMDb API](https://api.themoviedb.org/3/search/movie) to retrieve ratings for each movie. Although the API provides a range of data, we focused on ratings relevant to our project. The API returns data in CSV format, which we parse and integrate into our existing dataset.

---

## Method

### 1. Data Cleaning & Preprocessing

- **Cleaning Functions**: We created a function to clean each of the primary datasets (movie metadata, plot summaries, etc.), standardizing formats, handling missing values, and removing unnecessary columns.
- **Adding Ratings**: Ratings data were added using the TMDb API, providing a consistent metric for movie success.
- **Handling Missing Data**: Essential missing values were filled, and non-essential fields were removed.
- **Master Dataset**: A single, comprehensive dataset was built containing plot summaries, ratings, and movie metadata.

### 2. Metric Selection & Preliminary Analysis

- **Success Metric**: Ratings from TMDb were chosen as our primary success metric, providing a stable measure of audience perception over time. The reasons for this choice are detailed in `results.ipynb`.
- **Preliminary Analysis**:
  - **Ratings**: Calculated mean, standard deviation, min, and max to understand rating distribution.
  - **Release Trends**: Examined annual movie releases.
  - **Languages**: Analyzed the number of languages per movie and the most common ones.
  - **Actor Origins**: Investigated actor country distribution for geographic diversity.

### 3. Sentiment Analysis of Plot Summaries
*We explored VADER and DistilBERT for sentiment analysis and ultimately chose VADER for its efficiency and suitability with sentence-level plot summaries.*

1. **Sentence Segmentation & Sentiment Scoring (completed)**:
   - Each plot summary was segmented into sentences, and VADER was applied to score each sentence, creating an emotional trajectory for each movie.
   
2. **Planned Analysis (to do)**:
   - **Success Correlation Analysis**: Investigate how sentiment patterns (e.g., variability, peaks) correlate with movie success metrics.
   - **Archetype Comparison**: Classify movies by narrative archetype and assess which structures achieve the highest ratings.
   - **Predictive Modeling (if time permits)**: Use sentiment and archetypes to forecast movie success based on emotional trajectory.

### 4. Actor Fame & Network Analysis

1. **Actor Network Visualization (completed)**:
   - A network graph with nodes representing actors (scaled by movie rating) was generated, illustrating the influence of collaborations. Edges represent shared movie appearances.

2. **Planned Analysis (to do)**:
   - **Centrality Metrics**: Compute metrics (e.g., degree, betweenness, closeness) to create an Actor Fame Score.
   - **Community Detection**: Identify prominent actor communities and assess their impact on ratings.
   - **Predictive Modeling (if time permits)**: Suggest potential actor networks for maximizing a movie’s success.

### 5. Results & Interpretation (to do)
We will compile insights from sentiment patterns and actor networks to define the optimal mix of emotional arcs and actor collaborations for high movie ratings.

## Timeline

- **15.11.2024**: Milestone 2 Submission - Complete initial data handling, preprocessing, and preliminary analysis.
- **29.11.2024**: Task 3 (Sentiment Analysis) and Task 4 (Actor Network Analysis) - Focus on in-depth sentiment scoring, correlation with ratings, and network analysis of actor collaborations.
- **06.12.2024**: Task 5 (Results and Interpretation) - Integrate insights to determine the optimal patterns for high movie ratings.
- **13.12.2024**: Report Writing & Webpage Development - Finalize the report and webpage for presentation.
- **20.12.2024**: Milestone 3 Submission - Submit complete report and webpage.

## Team Organization

- **Task 3**: Alessio & Leonardo  
- **Task 4**: Quentin & Jiayi  
- **Task 5**: Pierre  
- **Report & Webpage**: All team members

