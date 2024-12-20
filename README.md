# 🎥 Stars or Storyline: Decoding Movie Success Through Sentiment and Networks

Welcome to **MovieKinsey Analytics**, a consulting startup that dives into the complex world of movie success. We aim to uncover the nuanced interplay between **narrative dynamics**, **actor influence**, and other factors that shape a movie's critical success. By leveraging data-driven insights, we provide studios, producers, and filmmakers with actionable strategies for their projects.  

Explore the full analysis on our interactive project site:  
👉 **[Project Webpage](https://pierrelts.github.io/ada-template-website/)**  

---

## **Abstract**
Our project explores the multifaceted factors that contribute to a movie's success, focusing on two major aspects:  
1. **The Narrative Arc** – How emotional trajectories in plots engage audiences.  
2. **Actor Influence** – How prominent actors impact success through their industry networks.  

While we uncovered fascinating relationships between features like sentiment arcs and actor fame, we found that predicting success remains a challenging task. Success is inherently complex, influenced by countless factors beyond measurable data.  

---

## **Key Findings**
1. **Sentiment Analysis of Movie Plots:**  
   - We analyzed plot summaries to extract **emotional trajectories** using **VADER sentiment analysis**.  
   - Movies following **complex arcs** like “Cinderella” and “Oedipus” performed better, suggesting audiences favor stories with dynamic emotional shifts.  

2. **Actor Network Analysis:**  
   - We built a **social network of actors**, calculating metrics like **degree centrality** to measure fame and collaboration influence.  
   - Prominent actors (those with higher network centrality) consistently correlated with higher movie success.  

3. **Economic and Temporal Factors:**  
   - Features like **holiday releases**, **exposure (language reach)**, and **genre popularity** also played a role in success.  

4. **Predictive Modeling Challenges:**  
   - Despite meaningful features, predictive performance remains modest.  
   - Our best-performing models (linear regression and decision trees) reached an R² of ~38–39%, underscoring the complexity of predicting movie success.  

---

## **Methodology Overview**
### **1. Data Collection & Feature Engineering**
- Collected data on **movie plots**, **actor collaborations**, and **metadata** like release dates, languages, and genres.  
- Extracted sentiment metrics such as **amplitude**, **variability**, and **emotional peaks**.  
- Built **actor collaboration networks** to quantify influence and identify patterns in casting decisions.  

### **2. Narrative Analysis**
- Identified **Kurt Vonnegut’s six narrative archetypes** in sentiment trajectories.  
- Linked narrative types like “Cinderella” and “Man in Hole” to higher levels of audience engagement.  

### **3. Actor Fame & Networks**
- Constructed actor networks where edges represented shared projects.  
- Measured **network centrality** to quantify actor prominence and examined its correlation with movie success.  

### **4. Prediction Models**
- Built models combining features from sentiment analysis, actor networks, and metadata.  
- Used techniques like **OLS regression** and **decision trees** to explore linear and non-linear relationships.  
- Validated models through **cross-validation** and assessed predictive accuracy.  

---

## **Insights and Reflections**
- **Features Matter, But Prediction is Tough:** While features like **sentiment amplitude**, **variability**, and **actor fame** showed clear correlations with success, predicting it with high accuracy remains elusive.  
- **Complex Stories Perform Better:** Narrative arcs with more emotional variation (e.g., “Cinderella” and “Oedipus”) consistently outperformed simpler arcs.  
- **Networks and Context Count:** Actor influence and contextual factors like release timing (e.g., holiday periods) also played significant roles.  

Our findings highlight the complexity of predicting success, suggesting that while features offer valuable insights, the nuances of human behavior and creative appeal make success prediction inherently uncertain.  

---

## **Contributions**
- **Alessio:** Designed and implemented the sentiment analysis pipeline and narrative arc clustering.  
- **Leonardo:** Explored predictive modeling and regression techniques, combining feature engineering with success metrics.  
- **Quentin:** Developed the actor collaboration network, applying graph metrics and community detection.  
- **Jiayi:** Focused on data cleaning, dataset integration, and metadata enrichment.  
- **Pierre:** Created visualizations, managed the project's web interface, and ensured an engaging presentation of findings.  

---

This project serves as a stepping stone for future analyses of movie success, blending storytelling, data science, and industry insights. Whether you’re a filmmaker, producer, or a curious data enthusiast, we hope this work inspires new ways to connect art with analytics. 🎬  
