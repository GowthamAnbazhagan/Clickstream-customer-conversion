# Clickstream-customer-conversion
This project analyzes clickstream data to predict customer conversion, estimate revenue, and segment users for e-commerce. A Streamlit app uses ML models for real-time insights, enhancing engagement and sales. Tools: Python, Scikit-learn, XGBoost, TensorFlow


# Customer Conversion Analysis for Online Shopping Using Clickstream Data

## 📌 Project Overview
This project aims to analyze clickstream data to predict customer conversion, estimate potential revenue, and segment customers based on their browsing behavior. The solution is deployed as an interactive **Streamlit web application**, enabling real-time insights for e-commerce businesses.

## 🏆 Key Objectives
- **Classification**: Predict whether a customer will complete a purchase (1) or not (0).
- **Regression**: Estimate the revenue a customer is likely to generate.
- **Clustering**: Segment customers into groups for targeted marketing and personalization.

## 📊 Business Use Cases
- **Customer Conversion Prediction**: Helps businesses target potential buyers more effectively.
- **Revenue Forecasting**: Assists in optimizing pricing and marketing strategies.
- **Customer Segmentation**: Enables businesses to provide personalized experiences.
- **Churn Reduction**: Detects users likely to abandon their carts and re-engages them.
- **Product Recommendations**: Provides tailored suggestions based on browsing patterns.

## 🔍 Approach
### 1️⃣ Data Preprocessing
- Handled missing values using mean/median for numerical and mode for categorical features.
- Encoded categorical variables using One-Hot Encoding/Label Encoding.
- Scaled numerical features using MinMaxScaler/StandardScaler.

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed session duration, page views, and bounce rates.
- Created visualizations such as histograms, bar charts, and correlation heatmaps.
- Extracted time-based features (hour, day of the week, browsing duration).

### 3️⃣ Feature Engineering
- Created session metrics (session length, clicks, time spent per category).
- Derived behavioral metrics (bounce rates, revisit patterns).
- Tracked click sequences to identify browsing paths.

### 4️⃣ Balancing Techniques for Classification
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes.
- Used **undersampling** to remove excess majority-class samples.
- Adjusted class weights during model training.

### 5️⃣ Model Building
#### Supervised Learning:
- **Classification Models**: Logistic Regression, Decision Trees, Random Forest, XGBoost, Neural Networks.
- **Regression Models**: Linear Regression, Ridge, Lasso, Gradient Boosting Regressors.
#### Unsupervised Learning:
- **Clustering Models**: K-Means, DBSCAN, Hierarchical Clustering.
- Built a **Scikit-learn Pipeline** for automated processing, training, and evaluation.

### 6️⃣ Model Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Regression**: RMSE, MAE, R-squared.
- **Clustering**: Silhouette Score, Davies-Bouldin Index.

### 7️⃣ Deployment with Streamlit
- Users can **upload CSV files** or manually input data.
- Key features:
  - Customer conversion predictions (classification)
  - Revenue estimation (regression)
  - Customer segmentation visualization (clustering)
  - Interactive visualizations (bar charts, histograms, etc.)

## 📌 Results
✅ High accuracy in predicting customer conversions.  
✅ Reliable revenue estimation from clickstream data.  
✅ Meaningful customer segmentation for personalized marketing.  
✅ Fully functional Streamlit app for real-time insights.

## 🛠 Tech Stack
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Supervised & Unsupervised models
- **Deployment**: Streamlit

## 📂 Dataset
- **Source**: UCI Machine Learning Repository - [Clickstream Data](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping)
- **Files**:
  - `train.csv` - Training dataset.
  - `test.csv` - Testing dataset.

## 📜 Project Deliverables
✔️ **Source Code**: Preprocessing, modeling, deployment scripts.  
✔️ **Streamlit App**: Interactive tool for real-time predictions.  
✔️ **Documentation**: Methodology, approaches, and results.  
✔️ **Presentation Deck**: Summarized findings and visualizations.  


## 🔗 References
- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [MLFlow Documentation](https://mlflow.org/docs/latest/getting-started/index.html)
- [Project Live Evaluation Guide](https://docs.google.com/document/d/1gbhLvJYY7J73lu1g9c6C9LRJvYemiDOdRDAEMe632w8/edit)

## 📅 Timeline
- **Project Completion**: 1 week  

---
🎯 **Created by:** Gowtham Anbazhagan  
📧 **Contact:** gowthamanbazhagan@gmail.com

