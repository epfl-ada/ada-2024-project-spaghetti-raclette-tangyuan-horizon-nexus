# 🎥 Stars or Storyline: How Actor Fame and Sentiment Trajectories Shape a Movie's Critical Success

Welcome to **MovieKinsey Analytics**, a consulting startup dedicated to decoding the secrets behind a movie's critical success. Our story-driven analysis blends **actor network influence** and **narrative sentiment arcs** to uncover how casting decisions and plot structures impact a movies success.  

Explore the full analysis on our interactive project site:  
👉 **[Project Webpage](https://pierrelts.github.io/ada-template-website/)**  

---

## **Abstract**
This project investigates the relative influence of two key factors on a movie's success: **actor fame** and the **emotional trajectory** of its plot. We hypothesize that a film’s success is shaped not only by its cast's prominence but also by the emotional journey it offers.  

Our approach combines:  
- **Sentiment Analysis:** Evaluating movie plot summaries to identify emotional complexity and narrative arcs.  
- **Social Network Analysis:** Measuring actor prominence through industry connections.  
- **Predictive Modeling:** Building a predictive model that forecasts movie success based on these features.  

---

## **Key Findings**
1. **Sentiment Analysis of Movie Plots:**  
   - We applied **VADER Sentiment Analysis** to movie summaries, tracking emotional highs and lows throughout the story.  
   - We identified **Kurt Vonnegut’s six classic narrative archetypes**, such as "Cinderella" and "Man in a Hole," and linked them to higher success.

2. **Actor Network Analysis:**  
   - We constructed a **collaboration network** of actors, where edges indicate shared movie appearances.  

3. **Predictive Model:**  
   - Using features from sentiment trajectories, actor prominence, and movie metadata, we built a predictive model that estimates a movie’s critical success.  

---

## **Methodology Overview**
### **1. Data Collection & Cleaning**
- Merged multiple datasets, including movie summaries, ratings, and metadata.  
- Extracted sentiment scores from plot summaries and built an actor collaboration network.  

### **2. Analysis Techniques**
- **Sentiment Features:** Extracted metrics like overall sentiment, variability, emotional peaks, and arc patterns.  
- **Actor Fame Metric:** Calculated degree centrality to estimate actor prominence.  
- **Community Detection:** Applied clustering techniques to identify influential actor groups.  

### **3. Prediction Model**
- Built a regression-based model combining sentiment features, actor fame, and metadata.  
- Validated performance through cross-validation, achieving meaningful predictive accuracy.  

---

## **Contributions**
- **Alessio:** Developed the data processing pipelines, conducted sentiment analysis using VADER, and built the emotional arc analysis framework.  
- **Leonardo:** Designed and implemented the predictive model, handling feature extraction and success metric definition.  
- **Quentin:** Built the actor collaboration network using graph theory, conducted community detection, and analyzed network centrality.  
- **Jiayi:** Performed data cleaning, dataset merging, and integrated metadata from external APIs like TMDb.  
- **Pierre:** Led data visualization, created interactive plots for the webpage, and managed the project’s front-end web development.  

---

We hope this project inspires future explorations into the art of filmmaking, data-driven storytelling, and predictive modeling for the entertainment industry. 🎬
