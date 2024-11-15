# Stars or Storyline: How Actor Fame and Sentiment Trajectories Shape a Movie's Critical Success

## Abstract

This project aims to uncover the relative influence of two key factors on a movie's critical success: actor fame and the narrative sentiment arc of the plot. We hypothesize that a film's success, might be influenced not only by the popularity of its cast but also by the emotional journey it offers. To explore this, we employ two distinct approaches: sentiment analysis on plot summaries to assess emotional complexity and patterns, and social network analysis to evaluate actor prominence and connectivity in the industry. Ultimately, this analysis seeks to provide a nuanced understanding of whether a star-studded cast or a compelling emotional narrative contributes more significantly to a film’s acclaim, offering insights into storytelling and casting strategies that could enhance a film's impact.

## Key Questions We Aim to Answer

- How can we evaluate a movie’s success (e.g., audience rating, revenue, other metric)?
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
- **Master Dataset**: A single, comprehensive dataset was built containing plot summaries, ratings, and movie metadata, while keeping the character dataset separated for convenience.

### 2. Metric Selection & Preliminary Analysis
- **Preliminary Analysis**:
   - **Release Trends**: Examined annual movie releases to understand trends.
   - **Revenue**: Diverse statistics.
   - **Ratings & Number of Ratings**: Diverse statistics.
   - **Success Metric**: Ratings from TMDb were chosen as our primary success metric, providing a stable measure of audience perception. Diverse statisctics and hypothetical correlation with Revenue to evaluate robustness of definition.
   - **Actor's begining age vs. life experience**: Actor's age of first apparition to see likeliness of it building further experience after.

### 3. Sentiment Analysis of Plot Summaries
*We explored VADER and DistilBERT for sentiment analysis and ultimately chose VADER for its efficiency and suitability with sentence-level plot summaries.*

1. **Sentence Segmentation & Sentiment Scoring (completed)**:
   - Each plot summary was segmented into sentences, and VADER was applied to score each sentence. This will allow us to compute an emotional trajectory for each movie. (Same goes for DistilBERT)
   
2. **Planned Analysis (to do)**:
   - **Success Correlation Analysis**: Investigate how sentiment patterns (e.g., variability, peaks) correlate with movie success metrics.
   - **Archetype Comparison**: Classify movies by narrative archetype and assess which structures achieve the highest ratings.
   - **Predictive Modeling (if time permits)**: Use sentiment and archetypes to forecast movie success based on emotional trajectory.

### 4. Actor Fame & Network Analysis

1. **Actor Network Visualization (completed)**:
   - A network graph with nodes representing actors (scaled by our success metric) was generated, illustrating the influence of collaborations. Edges represent shared movie appearances.

2. **Planned Analyses (to do)**:
   - **Community Structure Analysis**: Examine if certain actor communities have higher success rates by segmenting communities within the network graph.
   - **Cut Set Analysis**: Assess if key connectors bridging multiple communities achieve more success, using cut set identification within the community graph.

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

## Questions fo the TAs

- **Success outliers**: Success = Rating*log(No. of Ratings)
                        Only 1 Rating brings Success to 0 whatever the Rating ==> differentiation of these movies impossible. Solution would be to add an offset for ex: log(No. Ratings + 1).
                        Moreover, some movies have overall rating = 0 while revenue = millions ==> unlikely.
                        For now, movies with No. of Ratings < 2 are not taken into account as we consider them unreliable. What do you think?
               
- **Movie dataset & Character dataset**:  Once fully cleaned, the movie dataset = ~5000 movies. However, 
                                          for actor's connections we use all the actors from all the movies (>>5000 movies). Should we instead limit to the same dataset? 