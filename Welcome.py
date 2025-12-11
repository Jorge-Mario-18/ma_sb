import streamlit as st
from PIL import Image
# Set page config with soccer ball icon - must be first Streamlit command
st.set_page_config(
    page_title="Welcome",
    page_icon="⚽"
)

st.header("Player Analysis App")
logo = Image.open('ball.jpg')
st.sidebar.image(logo)

# Display image (place your image file in the project root or create an 'images' folder)
# Supported formats: JPG, PNG, GIF, WebP
try:
    st.image("soccer_analytics.jpg", use_container_width=True)  # Adjust path as needed
except FileNotFoundError:
    # If image not found, you can use a URL instead or comment out this section
    # st.image("https://example.com/your-image.jpg", use_container_width=True)
    pass


st.write("""
This app enables you to analyze how individual player statistics influence team-level performance metrics. 
By merging team-level target variables (such as Goal Difference, OBV, Possession%, and 90+ other metrics) 
with player-level data, you can build machine learning models to identify which player attributes most 
significantly impact team success.

**Key Features:**
- **Target Variable Selection**: Choose from 90+ team-level metrics to predict
- **Multiple ML Models**: Compare linear models (LassoCV, ElasticNetCV, RidgeCV) and non-linear models 
  (GradientBoosting, Random Forest, XGBoost) with customizable hyperparameters
- **Feature Importance Analysis**: Discover which player attributes (shots, passes, defensive actions, etc.) 
  have the strongest correlation with your chosen target variable
- **Model Performance Visualization**: View actual vs predicted plots, residual analysis, and feature 
  importance rankings to understand model accuracy and identify key drivers of team performance
""")
st.write("Data is based on team and player metrics from the USL 2025 season and La Liga 2024/2025 season")

st.write("By Jorge Mario Restrepo")
st.write("""
         Email: coachjorgemario@gmail.com""")

