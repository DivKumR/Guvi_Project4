# Guvi_Project4
Project Description: **Swiggy’s Restaurant Recommendation System using Streamlit **
This project builds a personalized restaurant recommendation system using clustering and similarity techniques. Powered by Python and Streamlit, it enables users to discover top-rated restaurants tailored to their preferences.

📄 Detailed Project Description:https://docs.google.com/document/d/1sai8gqvaoS6FinDVs3BvzTs8yBmiFH4SPFT8y8KujNA/edit?tab=t.0

🚀Project Structure  
├── cleaned_data.csv  
├── encoded_data.csv  
├── clustered_data.csv  
├── encoder.pkl  
├── app.py                 # Streamlit UI  
├── README.md  
- User Preferences: Inputs include city, cuisine, rating, and budget.
- Encoding & Scaling:- HashingEncoder transforms categorical features.
- StandardScaler scales features to improve model performance.
- Clustering:- K-Means groups restaurants into 10 clusters based on shared traits (e.g. rating, cost).
- Clusters help narrow recommendation candidates for faster and more relevant matches.
- Cosine Similarity:- Compares user input against clustered, standardized restaurant data.
- Top 5 similar restaurants are returned as recommendations.

📊 Data Cleaning
- Remove duplicates
- Impute missing values
- Save clean data → cleaned_data.csv
  
🧹 Data Preprocessing Workflow
Prior to building recommendations, the dataset undergoes extensive cleaning and transformation:
- 🔁 Deduplication: Removes redundant rows for consistency:
        df_swiggy = df_swiggy.drop_duplicates()
- 💰 Cost Cleaning
Strips currency symbols and converts cost to numeric:
        df_swiggy['cost'] = df_swiggy['cost'].replace('[₹,]', '', regex=True)
        df_swiggy['cost'] = pd.to_numeric(df_swiggy['cost'], errors='coerce')
- 🧮 Missing Value Imputation
Fills numerical missing values using median:
      df_swiggy['cost'].fillna(df_swiggy['cost'].median(), inplace=True)
      df_swiggy['rating'].fillna(df_swiggy['rating'].median(), inplace=True)
- ✂️ Drop Irrelevant & Incomplete Entries
      df_swiggy.dropna(subset=['name', 'city', 'cuisine'], inplace=True)
      df_swiggy.drop(columns=['id', 'link', 'menu'], inplace=True)

📊 Encoding Strategy
📌 One-Hot Encoding
Generates binary columns for each unique value in 'city' and 'cuisine':
    df_encoded = pd.get_dummies(df_swiggy, columns=['city', 'cuisine'], dtype=int)
    df_encoded.to_csv("encoded_data.csv", index=False)
    
💾 Encoder Export
Saves the trained encoder for reuse:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_swiggy[['city', 'cuisine']])
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    
📦 Installation
Install dependencies:
pip install streamlit pandas numpy scikit-learn category_encoders

▶️ Running the App
streamlit run app.py

Clustering Logic
- Used K-Means to group restaurants into 10 clusters.
- Removed non-numerical columns (city, cuisine, license, address) before clustering.
- Filled missing values with median and standardized features for consistency.
- Cluster assignment stored in df_encoded['cluster'].
Recommendation Logic
- Compared user input (e.g., preferred budget, rating) to restaurant feature vectors.
- Ranked all restaurants using cosine similarity.
- Returned the top 5 closest matches in the same or related clusters.

✨ Future Enhancements
- Interactive maps for restaurant locations
- Deeper filtering (e.g. delivery, open hours)
- Real-time location-based suggestions
- Cloud deployment (e.g. Azure, Streamlit Cloud)
