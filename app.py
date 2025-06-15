import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from category_encoders import HashingEncoder

# Load data
df_cleaned = pd.read_csv(r"C:\Users\v-dhramaraj\Desktop\Python\Projects\Assignment4_RestRecomendation\cleaned_data.csv")
df_encoded = pd.read_csv(r"C:\Users\v-dhramaraj\Desktop\Python\Projects\Assignment4_RestRecomendation\encoded_data.csv")

# Load precomputed encoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Apply Hashing Encoding to limit feature space
hashing_encoder = HashingEncoder(n_components=256)
df_encoded = hashing_encoder.fit_transform(df_cleaned[['city', 'cuisine']])

# Ensure numeric data before scaling
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(df_encoded.median())

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Streamlit UI
st.title("Restaurant Recommendation System")
st.sidebar.header("Enter Your Preferences")

city = st.sidebar.selectbox("Select City", df_cleaned['city'].unique())
cuisine = st.sidebar.selectbox("Select Cuisine", df_cleaned['cuisine'].unique())
rating = st.sidebar.slider("Minimum Rating", float(df_cleaned['rating'].min()), float(df_cleaned['rating'].max()), float(df_cleaned['rating'].median()))
price = st.sidebar.slider("Budget (₹)", float(df_cleaned['cost'].min()), float(df_cleaned['cost'].max()), float(df_cleaned['cost'].median()))

# Process Input & Compute Recommendations Immediately
input_data = pd.DataFrame([[city, cuisine]], columns=['city', 'cuisine'])
input_encoded = hashing_encoder.transform(input_data)

# Ensure transformed input is numeric before scaling
input_encoded = input_encoded.apply(pd.to_numeric, errors='coerce').fillna(input_encoded.median())

# Standardize input
input_vector = scaler.transform(input_encoded)

# Compute Similarity
similarities = cosine_similarity(input_vector, df_scaled)
recommendations = df_cleaned.iloc[similarities.argsort()[0][-5:]]

# # Store address selection in session state
# if "selected_address" not in st.session_state:
#     st.session_state.selected_address = None

# Display Recommendations Table with clickable addresses
st.subheader("Top Recommended Restaurants")

# selected_address = st.selectbox("Click an address to view location", recommendations['address'])

# Show recommendations table
st.write(recommendations[['name', 'city', 'rating', 'cost', 'cuisine', 'address']])

# # Display map below when an address is clicked
# if selected_address:
#     restaurant_info = recommendations[recommendations['address'] == selected_address]
#     st.subheader(f"Location of {selected_address}")
#     st.map(pd.DataFrame({'lat': restaurant_info['latitude'], 'lon': restaurant_info['longitude']}))

# Generate Summary Report
st.write("### Recommendation Methodology:")
st.write("- Used **K-Means Clustering** for grouping restaurants")
st.write("- Applied **Cosine Similarity** for personalized recommendations")

st.write("### Key Results:")
st.write(f"- Processed `{len(df_cleaned)}` restaurants")
st.write(f"- Found `{recommendations.shape[0]}` matching recommendations")