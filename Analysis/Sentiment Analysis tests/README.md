#### Sentiment Analysis Pipeline for Movie Plot Summaries

### Overview

This section of our project focuses on analyzing the sentiment trajectories of movie plot summaries to understand how emotional arcs might correlate with various movie characteristics, such as revenue, runtime, and genre. Our pipeline uses **DistilBERT**, a distilled version of BERT, to perform sentiment analysis on segmented movie plots. We analyze these trajectories across genres and cluster movies based on these sentiment patterns.

---

### Data Pipeline

The data pipeline for our sentiment analysis project consists of three main phases: **Data Preprocessing**, **Sentiment Analysis**, and **Analysis & Clustering**.

#### 1. Data Preprocessing

This phase includes loading, cleaning, and structuring the movie plot data, as well as the accompanying metadata. The steps involved are:

- **Load Movie Plot Summaries**: The raw plot summaries are loaded from a text file containing movie plot descriptions.
  
- **Sentence Segmentation**: Each plot summary is segmented into sentences using NLTK’s sentence tokenizer to capture sentence-by-sentence sentiment evolution.
  
- **Load Metadata**: Metadata (e.g., movie genre, runtime, revenue) is loaded.

#### 2. Sentiment Analysis

In this phase, we apply sentiment analysis to each segmented sentence to derive a sentiment trajectory for each movie:

- **Load Sentiment Analysis Model**: We use the DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) to analyze the sentiment of each sentence.
  
- **Calculate Sentiment Scores**: For each sentence, we calculate a sentiment score ranging from -1 (negative) to +1 (positive) based on the model output.

- **Store Sentiment Trajectories**: Each movie's sentiment trajectory (a sequence of sentiment scores for its sentences) is stored in a JSON file (`analyzed_sentiment_data.json`).

#### 3. Key Analysis & Clustering

This phase focuses on analyzing sentiment patterns and narrative archetypes to identify elements that may drive movie revenue.

- **Link Metadata and Sentiment Data**: Merge sentiment trajectories (from DistilBERT outputs) with metadata (e.g., runtime, release date) for each movie. This creates a comprehensive dataset that combines each movie’s sentiment trajectory with essential attributes, allowing for more granular analysis.

- **Classify Movies by Narrative Archetypes**:
    - **Sentiment Arcs as Narrative Types**: We categorize each movie’s sentiment trajectory into one of six common plot archetypes:
        - "Rags to Riches" (steady rise in sentiment)
        - "Riches to Rags" (steady decline)
        - "Man in a Hole" (fall then rise)
        - "Icarus" (rise then fall)
        - "Cinderella" (rise, fall, then rise)
        - "Oedipus" (fall, rise, then fall)
    - **Method**: For each movie, we analyze the sequence of sentiment scores (e.g., -1 for negative, +1 for positive) calculated per sentence. By observing the peaks and valleys in sentiment over time, we compare each movie’s sentiment sequence with idealized patterns representing each narrative archetype.
    - **Implementation**: We use time-series comparison methods (such as Dynamic Time Warping) to match each movie’s sentiment sequence with the closest archetype. This method accommodates variations in length and intensity, finding the best fit based on overall trend.

- **Cluster Movies by Sentiment Patterns**:
    - **Sentiment Features for Clustering**: We create feature vectors for each movie using key sentiment metrics:
        - **Average Sentiment**: Overall sentiment across sentences.
        - **Sentiment Variability**: Standard deviation of sentiment scores.
        - **Peaks and Valleys**: Number of significant highs (positive peaks) and lows (negative valleys).
    - **K-Means Clustering**: Using these features, we apply K-Means clustering to group movies based on similar sentiment trajectories. This groups movies with similar emotional flows, regardless of narrative archetype.
    - **PCA Visualization**: To visualize the clusters, we use PCA (Principal Component Analysis) to reduce dimensions and plot clusters. This allows us to observe distinct groups and compare their sentiment trajectories.

- **Save Analysis Results**: We save the archetype classifications, sentiment clusters, and linked metadata to `sentiment_archetype_cluster_analysis.csv`. This sets up a structured dataset for in-depth revenue correlation.

---

#### 4. Revenue Correlation Analysis

In this phase, we explore the relationship between sentiment patterns, narrative archetypes, and movie revenue, aiming to identify high-impact storytelling elements that drive box office success.

- **Revenue and Sentiment Correlation**: 
    - **Sentiment Metrics and Revenue**: Calculate correlation coefficients between revenue and sentiment metrics (average sentiment, variability, number of peaks/valleys). This determines if specific sentiment trends align with higher revenue.
    - **Visualize Correlation**: Plot sentiment metrics against revenue to visualize patterns. For instance, we may find that movies with high sentiment variability perform better financially, indicating that emotional ups and downs engage audiences.

- **Revenue Comparison Across Narrative Archetypes**:
    - **Archetype and Revenue Analysis**: Calculate average revenue for each narrative archetype. For example, does the "Cinderella" arc (multiple highs and lows) generate higher revenue than simpler arcs like "Rags to Riches"?
    - **Statistical Tests**: Use ANOVA or similar statistical tests to assess if revenue differences across archetypes are statistically significant.

- **Identify Revenue-Optimized Sentiment Clusters**:
    - **Cluster Revenue Analysis**: Calculate average revenue for each sentiment-based cluster to identify which sentiment patterns are associated with higher earnings.
    - **Cluster Comparison**: Compare clusters to find out if certain emotional journeys are more profitable. For instance, a "roller-coaster" trajectory (high variability) might attract larger audiences than a steady arc.

- **Save Revenue Analysis Results**: Save the results of this analysis to `sentiment_revenue_analysis.csv`, providing insights into how sentiment patterns and narrative archetypes impact box office success.

This pipeline is designed to uncover high-impact storytelling elements by examining which combination of sentiment patterns and narrative archetypes are most strongly associated with revenue outcomes.

#### 5. Predictive Modeling for Revenue Optimization (Maybe)

If time permits, we will build a predictive model to identify optimal sentiment patterns for maximizing movie revenue. This model aims to suggest sentiment arcs and narrative structures that align with higher box office performance.

- **Feature Engineering for Predictive Model**:
    - **Input Features**: Use the key sentiment features identified in prior analysis:
        - **Average Sentiment**: Overall tone across the plot.
        - **Sentiment Variability**: Standard deviation of sentiment, capturing emotional fluctuations.
        - **Peaks and Valleys**: Count of significant emotional highs and lows.
        - **Narrative Archetype**: Encoded archetype classification (e.g., "Cinderella," "Rags to Riches").
    - **Revenue as Target**: Revenue will be the target variable, allowing the model to learn patterns that correlate with financial success.

- **Modeling Approach**:
    - **Regression Models**: We will try multiple regression models (e.g., Linear Regression, Random Forest, XGBoost) to predict revenue based on sentiment and archetype features.
    - **Evaluation**: Use cross-validation to measure model performance, assessing metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) to determine predictive accuracy.

- **Feature Importance Analysis**:
    - **Interpret Results**: Identify which features most influence revenue, allowing us to isolate the sentiment patterns and archetypes with the highest impact on financial outcomes.
    - **Revenue Optimization Insights**: This analysis will reveal the sentiment traits that contribute most significantly to revenue, helping to refine storytelling strategies.

- **Save Model and Findings**: 
    - Store the final model and insights on feature importance in `revenue_prediction_model.pkl` and `revenue_optimization_summary.csv`. This will provide a predictive tool and insights on sentiment-driven revenue maximization.

This predictive modeling step will allow us to validate our findings and offer actionable recommendations for optimizing movie sentiment patterns and narrative structures to maximize revenue potential.
