import streamlit as st
from PIL import Image
import pandas as pd


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📑",
)

# --- HEADER ---
st.title("📊 Customer Segmentation Dashboard")
st.markdown("""Segment customers using **MLRFM** and **LRFM** analysis with clustering algorithms: **K-Means**, **DBSCAN**, and **K-Medoids**.""")

# --- INTRO SECTION ---
st.header("🔍 What is Customer Segmentation?")
st.markdown("""
Customer segmentation is the process of dividing customers into groups based on common characteristics.
This dashboard uses two powerful scoring methods:
- **LRFM**: Length, Recency, Frequency, Monetary
- **MLRFM**: Multi-Layer Recency, Frequency, Monetary

And applies three clustering techniques:
-  **K-Means Clustering**
-  **DBSCAN (Density-Based Spatial Clustering)**
-  **K-Medoids Clustering**

**C14210240 - Delon Teddy Gunawan** """)

if "df" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Data uploaded successfully!")
else:
    st.write("Data already loaded.")
    st.dataframe(st.session_state.df)