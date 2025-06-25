import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from datetime import datetime
from functools import reduce
from itertools import combinations
import random

st.title("Clustering ")

# Upload file
if "df" in st.session_state:
    df = st.session_state.df
    st.dataframe(df.head())
else:
    st.warning("Please upload your CSV file on the Home page first.")

if "df" in st.session_state:
    # Drop unnamed extra columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    try:
        df["tanggal"] = pd.to_datetime(df["tanggal"], format='mixed')
        df["qty"] = df["qty"].astype(str).str.replace(',', '').astype(int)
        df["harga"] = df["harga"].astype(str).str.replace(',', '').astype(float)
        df["total_harga"] = df["total_harga"].astype(str).str.replace(',', '').astype(float)
        df["diskon"] = pd.to_numeric(df["diskon"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    except Exception as e:
        st.error(f"Data preprocessing failed: {e}")
        st.stop()

    # Rename
    df.rename(columns={"nama_customer": "customer_id"}, inplace=True)

    #Year Selection
    # Extract year and determine range
    df['year'] = df['tanggal'].dt.year
    min_year, max_year = int(df['year'].min()), int(df['year'].max())

    # Allow user to choose a year range
    selected_year_range = st.slider(
        "Select year range to include in analysis",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Filter data by selected year range
    df = df[(df['year'] >= selected_year_range[0]) & (df['year'] <= selected_year_range[1])]

    # Handle empty selection
    if df.empty:
        st.warning("No data available in the selected year range.")
        st.stop()

    # --- Select Method ---
    st.subheader("Select Method and Fetures")
    method = st.selectbox("Select Clustering Method", ["K-Means", "K-Medoids", "DBSCAN"])
    features = st.selectbox("Select Features", ["LRFM", "MLRFM"])

    if method == "K-Means" and features == "LRFM":
        # --- Compute LRFM ---
        st.subheader("K-Means Clustering using LRFM")

        latest_date = df["tanggal"].max()
        
        customer_df = df.groupby("customer_id").agg({
            "tanggal": [
                lambda x: (latest_date - x.min()).days + 1,  # Length
                lambda x: (latest_date - x.max()).days   # Recency
            ],
            "qty": "count",            # Frequency
            "total_harga": "sum"       # Monetary
        })

        customer_df.columns = ['Length', 'Recency', 'Frequency', 'Monetary']
        customer_df = customer_df.reset_index()

        #Display LRFM feature
        st.dataframe(customer_df.head(10))

        # --- Apply Yeo-Johnson + MinMax ---
        st.subheader("Applying Yeo-Johnson Transformation and MinMax Scaling")

        lrfm = customer_df[['Length', 'Recency', 'Frequency', 'Monetary']]
        pt = PowerTransformer(method='yeo-johnson')
        lrfm_transformed = pt.fit_transform(lrfm)

        scaler = MinMaxScaler()
        lrfm_scaled = scaler.fit_transform(lrfm_transformed)

        # Reverse Recency (2nd column)
        lrfm_scaled[:, 1] = 1 - lrfm_scaled[:, 1]

        st.subheader("LRFM Feature Weighting")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            w_L = st.number_input("Weight for Length", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col2:
            w_R = st.number_input("Weight for Recency", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col3:
            w_F = st.number_input("Weight for Frequency", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col4:
            w_M = st.number_input("Weight for Monetary", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        # Normalize weights if they don't sum to 1
        weights = np.array([w_L, w_R, w_F, w_M])
        weights_sum = weights.sum()
        if weights_sum != 1.0:
            st.warning("Weights don't sum to 1. Make sure input again!")

        # Binning thresholds and corresponding scores
        bins = [0.00, 0.10, 0.30, 0.50, 0.70, 1.00]
        scores = [1, 2, 3, 4, 5]

        # Apply binning using numpy.digitize and subtract 1 to get 0-based index
        binned = np.digitize(lrfm_scaled, bins, right=True)
        binned[binned == 0] = 1       

        # Map the binned index to scores
        mapped_scores = np.array([scores[i-1] for i in binned.flatten()]).reshape(lrfm_scaled.shape)

        # Multiply each score by weight
        lrfm_scaled = mapped_scores * weights # dibuat input

        lrfm_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, lrfm_scaled)),
            columns=['customer_id', 'Length', 'Recency', 'Frequency', 'Monetary']
        )
        st.dataframe(lrfm_df.head(10))


        # set maximum k for evaluation
        max_k = st.slider("Maximum number of clusters to test", min_value=3, max_value=15, value=10)

        inertias = []
        X = lrfm_scaled  # Scaled LRFM data

        K_range = range(2, max_k + 1)
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            inertias.append(model.inertia_)

        # --- Plot Elbow Chart ---
        st.write("### Elbow Method (SSE)")
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o', color='blue')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("SSE score")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig_elbow)

        # --- Show Evaluation Table ---
        st.write("### Elbow method SSE")
        scores_df = pd.DataFrame({
            'k': list(K_range),
            'SSE Score': inertias
        })
        st.dataframe(scores_df.style.format(precision=3))

        # --- K-Means ---
        st.subheader("K-Means Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        lrfm_df['Cluster'] = kmeans.fit_predict(lrfm_scaled)

        # --- Visualization ---
        st.subheader("Cluster Visualization")
        # Ambil semua kombinasi 3 fitur dari ['L', 'R', 'F', 'M']
        features_visualitazion = ['Length', 'Recency', 'Frequency', 'Monetary']
        combinations_3d = list(combinations(features_visualitazion, 3))

        # selectbox
        combination_labels = [f"{x[0]}, {x[1]}, {x[2]}" for x in combinations_3d]
        selected_label = st.selectbox("Pick Combination for 3D Scatter Plot", combination_labels)

        # pick combination
        selected_features = selected_label.split(", ")  # hasil: ['L', 'R', 'F']

        # Plot 3D 
        fig = px.scatter_3d(
            lrfm_df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='Cluster',
            hover_data=lrfm_df.columns,
            title=f"3D Scatter Plot by Feature: {selected_features[0]} vs {selected_features[1]} vs {selected_features[2]}"
        )

        st.plotly_chart(fig)

        # --- Show clustered data ---
        st.subheader("Clustered Customer Table")
        st.dataframe(lrfm_df)

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(lrfm_scaled, lrfm_df['Cluster'])
        db_score = davies_bouldin_score(lrfm_scaled, lrfm_df['Cluster'])

        # --- Show Evaluation Metrics ---
        st.subheader("Evaluation Metrics for K-Means")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        
        # --- Segmentation profile ---
        avg_length = lrfm_df['Length'].mean()
        avg_recency = lrfm_df['Recency'].mean()
        avg_frequency = lrfm_df['Frequency'].mean()
        avg_monetary = lrfm_df['Monetary'].mean()

        st.subheader("Global Average LRFM Values")
        global_avg = {
            'Length': lrfm_df['Length'].mean(),
            'Recency': lrfm_df['Recency'].mean(),
            'Frequency': lrfm_df['Frequency'].mean(),
            'Monetary': lrfm_df['Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = lrfm_df.groupby('Cluster')[['Length', 'Recency', 'Frequency', 'Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Length', 'Recency', 'Frequency', 'Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]
        
        # Segment mapping
        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Length_high', 'Recency_high', 'Frequency_high', 'Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )

        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Length', 'Recency', 'Frequency', 'Monetary', 'Segment']])
        st.bar_chart(lrfm_df['Cluster'].value_counts())

        # --- Download result ---
        csv = lrfm_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "K-Means_clustered_customers.csv", "text/csv")
    

    # --- Kmeans MLRFM ---
    elif method == "K-Means" and features == "MLRFM":
        st.subheader("K-Means Clustering using MLRFM")
        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)
        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]
        
        #Periods
        periods = {'365d': 365, '730d': 730, 'All': None}

        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()  # All data
                
            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })

            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        st.dataframe(final_mlrfm.head(10))    

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        
        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = ['Recency', 'Frequency_365d', 'Frequency_730d', 'Frequency_All',
                            'Monetary_365d', 'Monetary_730d', 'Monetary_All']

        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))  

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # Normalize weights
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        # Compute combined Frequency & Monetary
        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )
        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer','Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))  

        # Bins
        bins = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]
        labels = [1, 2, 3, 4, 5]

        # Note: Recency Reverse
        mlrfm_scaled_df['Recency'] = pd.cut(mlrfm_scaled_df['Recency'], bins=bins, labels=labels[::-1], include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Frequency'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Frequency'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Monetary'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Monetary'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df)  

        K_range = range(2, 11)

        st.subheader("Elbow Method")

        # Allow user to define max k
        max_k = st.slider("Maximum number of clusters to test (Elbow)", min_value=3, max_value=15, value=10)

        inertias = []
        X = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]

        K_range = range(2, max_k + 1)
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            inertias.append(model.inertia_)
        
        # Plot Elbow
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o')
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("SSE Score")
        ax.set_title("Elbow Method For Optimal k")
        st.pyplot(fig_elbow)

        # Display scores for each k
        scores_df = pd.DataFrame({
            'k': list(K_range),
            'SSE Score': inertias,
        })
        st.dataframe(scores_df.style.format(precision=3))

        #--- K-Means Cluster ---
        st.subheader("K-Means Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)
        X = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        mlrfm_scaled_df['Cluster'] = kmeans.fit_predict(X)

        st.subheader("3D Cluster Visualization")
        fig = px.scatter_3d(
            mlrfm_scaled_df,
            x='Recency',
            y='Multi_Layer_Frequency',
            z='Multi_Layer_Monetary',
            color='Cluster',
            hover_data=['nama_customer'],
            title="3D Customer Segmentation"
        )
        st.plotly_chart(fig)

        st.subheader("Clustered Data Table")
        st.dataframe(mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']])

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        db_score = davies_bouldin_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])

        # ---Silhouette, Davies-Bouldin ---
        st.subheader("Evaluation Metrics for K-Means")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")

        # --- Segmentation profile ---
        avg_recency = mlrfm_scaled_df['Recency'].mean()
        avg_frequency = mlrfm_scaled_df['Multi_Layer_Frequency'].mean()
        avg_monetary = mlrfm_scaled_df['Multi_Layer_Monetary'].mean()

        st.subheader("Global Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[[ 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers",           # R↓ F↑ M↑ 
            (False, False,  True):  "Lost customers"           # R↓ F↓ M↑ 
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "mlrfm_clustered.csv", "text/csv")
            
    elif method == "K-Medoids" and features == "LRFM":
        st.subheader("K-Mediods Clustering using LRFM")

        latest_date = df["tanggal"].max()
        
        customer_df = df.groupby("customer_id").agg({
            "tanggal": [
                lambda x: (latest_date - x.min()).days + 1,  # Length
                lambda x: (latest_date - x.max()).days   # Recency
            ],
            "qty": "count",            # Frequency
            "total_harga": "sum"       # Monetary
        })

        customer_df.columns = ['Length', 'Recency', 'Frequency', 'Monetary']
        customer_df = customer_df.reset_index()

        #Display LRFM feature
        st.dataframe(customer_df.head(10))

        # --- Apply Yeo-Johnson + MinMax ---
        st.subheader("Applying Yeo-Johnson Transformation and MinMax Scaling")

        lrfm = customer_df[['Length', 'Recency', 'Frequency', 'Monetary']]
        pt = PowerTransformer(method='yeo-johnson')
        lrfm_transformed = pt.fit_transform(lrfm)

        scaler = MinMaxScaler()
        lrfm_scaled = scaler.fit_transform(lrfm_transformed)

        # Reverse Recency (2nd column)
        lrfm_scaled[:, 1] = 1 - lrfm_scaled[:, 1]

        st.subheader("LRFM Feature Weighting")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            w_L = st.number_input("Weight for Length", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col2:
            w_R = st.number_input("Weight for Recency", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col3:
            w_F = st.number_input("Weight for Frequency", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col4:
            w_M = st.number_input("Weight for Monetary", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        # Normalize weights if they don't sum to 1
        weights = np.array([w_L, w_R, w_F, w_M])
        weights_sum = weights.sum()
        if weights_sum != 1.0:
            st.warning("Weights don't sum to 1. Make sure input again!")

        # Binning thresholds and corresponding scores
        bins = [0.00, 0.10, 0.30, 0.50, 0.70, 1.00]
        scores = [1, 2, 3, 4, 5]

        # Apply binning 
        binned = np.digitize(lrfm_scaled, bins, right=True)
        binned[binned == 0] = 1  # Handle any 0.00 edge cases

        # Map the binned index to scores
        mapped_scores = np.array([scores[i-1] for i in binned.flatten()]).reshape(lrfm_scaled.shape)

        # Multiply each score by 0.25
        lrfm_scaled = mapped_scores * weights

        lrfm_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, lrfm_scaled)),
            columns=['customer_id', 'Length', 'Recency', 'Frequency', 'Monetary']
        )

        st.dataframe(lrfm_df.head(10))     

        st.subheader("Clustering Evaluation")

        max_k = st.slider("Maximum number of clusters to test", min_value=3, max_value=15, value=10)

        inertias = []
        K_range = range(2, max_k + 1)

        distance_matrix = calculate_distance_matrix(lrfm_scaled)

        for k in K_range:
            try:
                initial_medoids_k = random.sample(range(len(lrfm_scaled)), k)
                kmedoids_eval = kmedoids(distance_matrix, initial_medoids_k, data_type='distance_matrix')
                kmedoids_eval.process()
                clusters_k = kmedoids_eval.get_clusters()

                labels_k = np.zeros(len(lrfm_scaled), dtype=int)
                for cid, clus in enumerate(clusters_k):
                    for idx in clus:
                        labels_k[idx] = cid

                # Calculate inertia manually (sum of distances to medoid)
                inertia_k = 0
                for cid, clus in enumerate(clusters_k):
                    medoid_idx = kmedoids_eval.get_medoids()[cid]
                    inertia_k += sum(np.linalg.norm(lrfm_scaled[i] - lrfm_scaled[medoid_idx]) for i in clus)

                inertias.append(inertia_k)

            except Exception as e:
                inertias.append(None)
                st.warning(f"Clustering failed for k={k}: {e}")

        # --- Plot Elbow ---
        st.write("### Elbow Method (SSE)")
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o', color='blue')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("SSE Score")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig_elbow)

        # --- Show Evaluation Table ---
        st.write("### SSE Score")
        scores_df = pd.DataFrame({
            'k': list(K_range),
            'SSE Score': inertias
        })
        st.dataframe(scores_df.style.format(precision=3))

        # --- K-Medoids ---
        st.subheader("K-Medoids Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)

        # Select initial medoids 
        initial_medoids = list(range(n_clusters))

        # Run K-Medoids
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        # Assign cluster labels
        labels = np.zeros(len(lrfm_scaled), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id
        lrfm_df['Cluster'] = labels

        # --- Visualization ---
        st.subheader("Cluster Visualization")
        # Ambil semua kombinasi 3 fitur dari ['L', 'R', 'F', 'M']
        features_visualitazion = ['Length', 'Recency', 'Frequency', 'Monetary']
        combinations_3d = list(combinations(features_visualitazion, 3))

        # selectbox
        combination_labels = [f"{x[0]}, {x[1]}, {x[2]}" for x in combinations_3d]
        selected_label = st.selectbox("Pick Combination for 3D Scatter Plot", combination_labels)

        # pick combination
        selected_features = selected_label.split(", ")  # hasil: ['L', 'R', 'F']

        # Plot 3D 
        fig = px.scatter_3d(
            lrfm_df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='Cluster',
            hover_data=lrfm_df.columns,
            title=f"3D Scatter Plot by Feature: {selected_features[0]} vs {selected_features[1]} vs {selected_features[2]}"
        )

        st.plotly_chart(fig)

        # --- Show clustered data ---
        st.subheader("Clustered Customer Table")
        st.dataframe(lrfm_df)

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(lrfm_scaled, lrfm_df['Cluster'])
        db_score = davies_bouldin_score(lrfm_scaled, lrfm_df['Cluster'])

        # --- Show Evaluation Metrics ---
        st.subheader("Evaluation Metrics for K-Medoids")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        
        # --- Segmentation profile ---
        avg_length = lrfm_df['Length'].mean()
        avg_recency = lrfm_df['Recency'].mean()
        avg_frequency = lrfm_df['Frequency'].mean()
        avg_monetary = lrfm_df['Monetary'].mean()

        st.subheader("Global Average LRFM Values")
        global_avg = {
            'Length': lrfm_df['Length'].mean(),
            'Recency': lrfm_df['Recency'].mean(),
            'Frequency': lrfm_df['Frequency'].mean(),
            'Monetary': lrfm_df['Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = lrfm_df.groupby('Cluster')[['Length', 'Recency', 'Frequency', 'Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Length', 'Recency', 'Frequency', 'Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]
        
        # Segment mapping
        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Length_high', 'Recency_high', 'Frequency_high', 'Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Length', 'Recency', 'Frequency', 'Monetary', 'Segment']])
        st.bar_chart(lrfm_df['Cluster'].value_counts())

        # --- Download result ---
        csv = lrfm_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "K-Medoids_LRFM_clustered_customers.csv", "text/csv")


    elif method == "K-Medoids" and features == "MLRFM":  
        st.subheader("K-Medoids Clustering using MLRFM")
        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)
        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]
        
        #Periods
        periods = {'365d': 365, '730d': 730, 'All': None}

        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()  # All data
                
            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })

            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        st.dataframe(final_mlrfm.head(10))    

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        
        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = ['Recency', 'Frequency_365d', 'Frequency_730d', 'Frequency_All',
                            'Monetary_365d', 'Monetary_730d', 'Monetary_All']

        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))  

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # Normalize weights
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        # Compute combined Frequency & Monetary
        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )
        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer','Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))  

        # Bins
        bins = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]
        labels = [1, 2, 3, 4, 5]

        # Note: Recency Reverse
        mlrfm_scaled_df['Recency'] = pd.cut(mlrfm_scaled_df['Recency'], bins=bins, labels=labels[::-1], include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Frequency'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Frequency'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Monetary'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Monetary'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df)  

        K_range = range(2, 11)

        st.subheader("Elbow Method")

        X = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].values.tolist()
        distance_matrix = calculate_distance_matrix(X)

        # Allow user to define max k
        max_k = st.slider("Maximum number of clusters to test (Elbow)", min_value=3, max_value=15, value=10)

        inertias = []
        X_scaled = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].values.tolist()

        K_range = range(2, max_k + 1)
        for k in K_range:
            try:
                initial_medoids_k = random.sample(range(len(X_scaled)), k)
                kmedoids_eval = kmedoids(distance_matrix, initial_medoids_k, data_type='distance_matrix')
                kmedoids_eval.process()
                clusters_k = kmedoids_eval.get_clusters()

                # Assign cluster labels
                labels_k = np.zeros(len(X_scaled), dtype=int)
                for cid, cluster_k in enumerate(clusters_k):
                    for idx in cluster_k:
                        labels_k[idx] = cid

                # Calculate inertia manually (sum of distances to medoid)
                inertia_k = 0
                for cid, cluster_k in enumerate(clusters_k):
                    medoid_idx = kmedoids_eval.get_medoids()[cid]
                    inertia_k += sum(np.linalg.norm(np.array(X_scaled[i]) - np.array(X_scaled[medoid_idx])) for i in cluster_k)

                inertias.append(inertia_k)

            except Exception as e:
                inertias.append(None)
                st.warning(f"Clustering failed for k={k}: {e}")

        # Plot Elbow
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o')
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("SSE Score")
        ax.set_title("Elbow Method For Optimal k")
        st.pyplot(fig_elbow)

        # Display scores for each k
        scores_df = pd.DataFrame({
            'k': list(K_range),
            'SSE Score': inertias,
        })
        st.dataframe(scores_df.style.format(precision=3))

        ## --- K-Medoids ---
        st.subheader("K-Medoids Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)

        # choose initial medoids
        initial_medoids = list(range(n_clusters))

        # Apply K-Medoids
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()

        # Assign cluster labels to each point
        cluster_labels = np.zeros(len(X), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                cluster_labels[index] = cluster_id

        mlrfm_scaled_df['Cluster'] = cluster_labels
        st.subheader("3D Cluster Visualization")
        fig = px.scatter_3d(
            mlrfm_scaled_df,
            x='Recency',
            y='Multi_Layer_Frequency',
            z='Multi_Layer_Monetary',
            color='Cluster',
            hover_data=['nama_customer'],
            title="3D Customer Segmentation"
        )
        st.plotly_chart(fig)

        st.subheader("Clustered Data Table")
        st.dataframe(mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']])

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        db_score = davies_bouldin_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])

        # ---Silhouette, Davies-Bouldin ---
        st.subheader("Evaluation Metrics for K-Medoids")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")

        # --- Segmentation profile ---
        avg_recency = mlrfm_scaled_df['Recency'].mean()
        avg_frequency = mlrfm_scaled_df['Multi_Layer_Frequency'].mean()
        avg_monetary = mlrfm_scaled_df['Multi_Layer_Monetary'].mean()

        st.subheader("Global Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[[ 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers",           # R↓ F↑ M↑ 
            (False, False,  True):  "Lost customers"           # R↓ F↓ M↑ 
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "K-Medoids_mlrfm_clustered.csv", "text/csv")

    elif method == "DBSCAN" and features == "LRFM":  
        # --- Compute LRFM ---
        st.subheader("DBSCAN Clustering using LRFM")

        latest_date = df["tanggal"].max()
        
        customer_df = df.groupby("customer_id").agg({
            "tanggal": [
                lambda x: (latest_date - x.min()).days + 1,  # Length
                lambda x: (latest_date - x.max()).days   # Recency
            ],
            "qty": "count",            # Frequency
            "total_harga": "sum"       # Monetary
        })

        customer_df.columns = ['Length', 'Recency', 'Frequency', 'Monetary']
        customer_df = customer_df.reset_index()

        #Display LRFM feature
        st.dataframe(customer_df.head(10))

        # --- Apply Yeo-Johnson + MinMax ---
        st.subheader("Applying Yeo-Johnson Transformation and MinMax Scaling")

        lrfm = customer_df[['Length', 'Recency', 'Frequency', 'Monetary']]
        pt = PowerTransformer(method='yeo-johnson')
        lrfm_transformed = pt.fit_transform(lrfm)

        # lrfm_transformed = lrfm
        # lrfm_scaled = lrfm

        scaler = MinMaxScaler()
        lrfm_scaled = scaler.fit_transform(lrfm_transformed)

        # Reverse Recency (2nd column)
        lrfm_scaled[:, 1] = 1 - lrfm_scaled[:, 1]
        st.subheader("LRFM Feature Weighting")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            w_L = st.number_input("Weight for Length", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col2:
            w_R = st.number_input("Weight for Recency", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col3:
            w_F = st.number_input("Weight for Frequency", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        with col4:
            w_M = st.number_input("Weight for Monetary", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        # Normalize weights if they don't sum to 1
        weights = np.array([w_L, w_R, w_F, w_M])
        weights_sum = weights.sum()
        if weights_sum != 1.0:
            st.warning("Weights don't sum to 1. Make sure input again!")

        # # Binning thresholds and corresponding scores
        # bins = [0.00, 0.10, 0.30, 0.50, 0.70, 1.00]
        # scores = [1, 2, 3, 4, 5]

        # # Apply binning using numpy.digitize and subtract 1 to get 0-based index
        # binned = np.digitize(lrfm_scaled, bins, right=True)
        # binned[binned == 0] = 1  # Handle any 0.00 edge cases

        # # Map the binned index to scores
        # mapped_scores = np.array([scores[i-1] for i in binned.flatten()]).reshape(lrfm_scaled.shape)

        # # Multiply each score by 0.25
        # lrfm_scaled = lrfm_scaled * weights

        lrfm_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, lrfm_scaled)),
            columns=['customer_id', 'Length', 'Recency', 'Frequency', 'Monetary']
        )
        st.dataframe(lrfm_df.head(10))

        # --- K-Distance (Elbow Method for eps)
        st.subheader("Elbow Method for Optimal eps (k-distance plot)")
        eps = st.slider("Set epsilon (eps)", min_value=0.01, max_value=2.0, value=0.16, step=0.01)
        min_samples = st.slider("Set min_samples", min_value=1, max_value=20, value=6)
        
        k = min_samples
        nn = NearestNeighbors(n_neighbors=k)
        nn_fit = nn.fit(lrfm_scaled)
        distances, _ = nn_fit.kneighbors(lrfm_scaled)
        distances = np.sort(distances[:, -1])  # sort k-th NN distances

        fig_eps, ax_eps = plt.subplots()
        ax_eps.plot(distances)
        ax_eps.set_title(f"k-Distance Graph (k={k})")
        ax_eps.set_xlabel("Data Points sorted by distance")
        ax_eps.set_ylabel(f"{k}-NN Distance")
        st.pyplot(fig_eps)

        # --- DBSCAN ---
        st.subheader("DBSCAN Clustering")

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(lrfm_scaled)

        lrfm_df['Cluster'] = labels

        # --- Visualization ---
        st.subheader("Cluster Visualization")
        # Ambil semua kombinasi 3 fitur dari ['L', 'R', 'F', 'M']
        features_visualitazion = ['Length', 'Recency', 'Frequency', 'Monetary']
        combinations_3d = list(combinations(features_visualitazion, 3))

        # selectbox
        combination_labels = [f"{x[0]}, {x[1]}, {x[2]}" for x in combinations_3d]
        selected_label = st.selectbox("Pick Combination for 3D Scatter Plot", combination_labels)

        # pick combination
        selected_features = selected_label.split(", ")  # hasil: ['L', 'R', 'F']

        # Plot 3D 
        fig = px.scatter_3d(
            lrfm_df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='Cluster',
            hover_data=lrfm_df.columns,
            title=f"3D Scatter Plot by Feature: {selected_features[0]} vs {selected_features[1]} vs {selected_features[2]}"
        )

        st.plotly_chart(fig)

        # --- Evaluation Scores (ignore noise label -1)
        valid_labels = lrfm_df['Cluster'].unique()
        valid_labels = valid_labels[valid_labels != -1]  # exclude noise
        n_clusters = len(valid_labels)

        st.subheader("Clustered Customer Table")
        st.dataframe(lrfm_df)

        # --- Silhoeutte score and davies bouldin
        st.write("### Evaluation Metrics")
        if n_clusters > 1:
            mask = lrfm_df['Cluster'] != -1
            sil_score = silhouette_score(lrfm_scaled[mask], labels[mask])
            db_score = davies_bouldin_score(lrfm_scaled[mask], labels[mask])
            st.markdown(f"**Number of Clusters (excluding noise):** {n_clusters}")
            st.success(f"**Silhouette Score:** {sil_score:.4f}")
            st.success(f"**Davies-Bouldin Score:** {db_score:.4f}")
        else:
            st.warning("Not enough clusters found (excluding noise) to calculate evaluation metrics.")

        # --- Segmentation profile ---
        avg_length = lrfm_df['Length'].mean()
        avg_recency = lrfm_df['Recency'].mean()
        avg_frequency = lrfm_df['Frequency'].mean()
        avg_monetary = lrfm_df['Monetary'].mean()

        st.subheader("Global Average LRFM Values")
        global_avg = {
            'Length': lrfm_df['Length'].mean(),
            'Recency': lrfm_df['Recency'].mean(),
            'Frequency': lrfm_df['Frequency'].mean(),
            'Monetary': lrfm_df['Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = lrfm_df.groupby('Cluster')[['Length', 'Recency', 'Frequency', 'Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Length', 'Recency', 'Frequency', 'Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]
        
        # Segment mapping
        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Length_high', 'Recency_high', 'Frequency_high', 'Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Length', 'Recency', 'Frequency', 'Monetary', 'Segment']])
        st.bar_chart(lrfm_df['Cluster'].value_counts())  

        # --- Download result
        st.write("### Download")
        csv = lrfm_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "clustered_customers_dbscan.csv", "text/csv")

    elif method == "DBSCAN" and features == "MLRFM":  
        st.subheader("DBSCAN Clustering using MLRFM")
        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)
        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]
        
        #Periods
        periods = {'365d': 365, '730d': 730, 'All': None}

        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()  # All data
                
            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })

            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        st.dataframe(final_mlrfm.head(10))    

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        
        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = ['Recency', 'Frequency_365d', 'Frequency_730d', 'Frequency_All',
                            'Monetary_365d', 'Monetary_730d', 'Monetary_All']

        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])
 
        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))  

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # Normalize weights
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        # Compute combined Frequency & Monetary
        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )
        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer','Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))  

        # # Bins
        # bins = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]
        # labels = [1, 2, 3, 4, 5]

        # # Note: Recency Reverse
        # mlrfm_scaled_df['Recency'] = pd.cut(mlrfm_scaled_df['Recency'], bins=bins, labels=labels[::-1], include_lowest=True).astype(int) * 0.33
        # mlrfm_scaled_df['Multi_Layer_Frequency'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Frequency'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33
        # mlrfm_scaled_df['Multi_Layer_Monetary'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Monetary'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33

        # mlrfm_scaled_df['Recency'] = mlrfm_scaled_df['Recency'] *  0.33
        # mlrfm_scaled_df['Multi_Layer_Frequency'] = mlrfm_scaled_df['Multi_Layer_Frequency'] * 0.33
        # mlrfm_scaled_df['Multi_Layer_Monetary'] = mlrfm_scaled_df['Multi_Layer_Monetary'] *0.33

        mlrfm_scaled_df['Recency'] = 1 - mlrfm_scaled_df['Recency'] #command if use bin

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df) 

        st.subheader("DBSCAN Clustering")
        X = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]

        min_samples = st.slider("Set min_samples", 2, 15, 6)

        # Epsilon Estimation via k-distance graph
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, min_samples - 1])

        fig_kdist, ax = plt.subplots()
        ax.plot(distances)
        ax.set_title("K-Distance Graph (Pick ε from elbow)")
        ax.set_xlabel("Data Points (sorted)")
        ax.set_ylabel(f"{min_samples}-NN Distance")
        st.pyplot(fig_kdist)

        epsilon = st.number_input("Set epsilon (based on the graph above)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

        db = DBSCAN(eps=epsilon, min_samples=min_samples)
        mlrfm_scaled_df['Cluster'] = db.fit_predict(X)

        st.subheader("3D Cluster Visualization")
        fig_3d = px.scatter_3d(
            mlrfm_scaled_df,
            x='Recency',
            y='Multi_Layer_Frequency',
            z='Multi_Layer_Monetary',
            color=mlrfm_scaled_df['Cluster'].astype(str),
            hover_data=['nama_customer'],
            title="DBSCAN Clusters"
        )
        st.plotly_chart(fig_3d)

        st.subheader("Clustered Data Table")
        st.dataframe(mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']])

        # --- Evaluation (ignoring noise cluster -1)
        st.write("### Evaluation Metrics")
        valid_clusters = mlrfm_scaled_df[mlrfm_scaled_df['Cluster'] != -1]
        if len(valid_clusters['Cluster'].unique()) > 1:
            sil = silhouette_score(valid_clusters[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']], valid_clusters['Cluster'])
            dbi = davies_bouldin_score(valid_clusters[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']], valid_clusters['Cluster'])
            st.success(f"Silhouette Score (excluding noise): {sil:.3f}")
            st.success(f"Davies-Bouldin Index (excluding noise): {dbi:.3f}")
        else:
            st.warning("Not enough clusters to compute Silhouette or DB Index (need >1 cluster excluding noise).")
        # --- Segmentation profile ---
        avg_recency = mlrfm_scaled_df['Recency'].mean()
        avg_frequency = mlrfm_scaled_df['Multi_Layer_Frequency'].mean()
        avg_monetary = mlrfm_scaled_df['Multi_Layer_Monetary'].mean()

        st.subheader("Global Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[[ 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers",           # R↓ F↑ M↑ 
            (False, False,  True):  "Lost customers"            # R↓ F↓ M↑ 
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "mlrfm_clustered.csv", "text/csv")


    #--- Segment Profiling
    st.title("Segment Profiling")
    seg_df = df.rename(columns={'nama_customer': 'customer_id'})
    if features == "LRFM":
        df_trans = seg_df.merge(lrfm_df[['customer_id', 'Cluster']], on='customer_id', how='left')
    else:
        mlrfm_scaled_df = mlrfm_scaled_df.rename(columns={'nama_customer': 'customer_id'})
        df_trans = seg_df.merge(mlrfm_scaled_df[["customer_id", 'Cluster']], on='customer_id', how='left')   
    
    df_trans['year'] = df_trans['tanggal'].dt.year
    df_trans['month'] = df_trans['tanggal'].dt.month
    df_trans['day'] = df_trans['tanggal'].dt.date
    df_trans['day_of_week'] = df_trans['tanggal'].dt.day_name()
        
    selected = st.selectbox("Select Cluster", df_trans['Cluster'].unique())

    filtered_df = df_trans[df_trans['Cluster'] == selected]

    st.subheader(f"Customer Analysis - Cluster {selected}")

    # Total pelanggan dalam cluster
    total_pelanggan = filtered_df['customer_id'].nunique()
    st.metric(label=f"👥 Total Pelanggan dalam Cluster {selected}", value=total_pelanggan)
   
    # Preferensi barang
    st.subheader("Preferensi Barang yang Dibeli")
    preferensi_barang = (
        filtered_df.groupby(['nama_barang'])
        .size()
        .reset_index(name='jumlah_beli')
        .sort_values('jumlah_beli', ascending=False)
    )
    st.dataframe(preferensi_barang)


    # 💰 Rata-rata uang per transaksi per customer
    uang_per_transaksi = (
        filtered_df.groupby(['customer_id', 'no_invoice'])['total_harga'].sum().reset_index()
    )
    avg_uang_per_customer = (
        uang_per_transaksi.groupby('customer_id')['total_harga'].mean().mean()
    )

    # 📦 Rata-rata qty per transaksi per customer
    produk_per_transaksi = (
        filtered_df.groupby(['no_invoice'])['qty'].sum().reset_index(name='total_qty')
    )
    avg_qty_per_customer = (
        produk_per_transaksi['total_qty'].mean().mean()
    )

    # ⏱️ Rata-rata rentang waktu antar transaksi per customer
    tanggal_per_trans = (
        filtered_df.sort_values('tanggal')
        .drop_duplicates(subset=['customer_id', 'no_invoice'])[['customer_id', 'tanggal']]
    )

    def avg_days_between(dates):
        if len(dates) < 2:
            return None
        diffs = dates.sort_values().diff().dropna()
        return diffs.mean().days

    avg_rentang_per_customer = (
        tanggal_per_trans.groupby('customer_id')['tanggal']
        .apply(avg_days_between)
        .dropna()
        .mean()
    )

    # Tampilkan hasil
    st.markdown(f"**💰 Average Money Spent per Transaction per Customer:** Rp {avg_uang_per_customer:,.0f}")
    st.markdown(f"**📦 Average Product Sold per Transacrion per Customer:** {avg_qty_per_customer:.2f} item")
    st.markdown(f"**⏱️ Average Days Between Each Transaction per Customer:** {avg_rentang_per_customer:.2f} hari") 

    st.subheader("📆 Total Transaction per Year")
    tahun_df = (
        filtered_df.groupby(['year'])['no_invoice']
        .nunique()
        .reset_index(name='jumlah_transaksi')
    )
    st.dataframe(tahun_df)
    st.bar_chart(tahun_df.set_index('year'))

    # Jumlah Transaksi per Bulan
    st.subheader("📅 Total Transaction for Each Month (All Years)")

    # Hitung jumlah transaksi per bulan
    bulan_df = (
        filtered_df.groupby(['month'])['no_invoice']
        .nunique()
        .reset_index(name='jumlah_transaksi')
    )

    # Tambahkan nama bulan
    bulan_df['month_name'] = bulan_df['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))

    # Urutkan bulan dari Januari - Desember
    bulan_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    bulan_df['month_name'] = pd.Categorical(bulan_df['month_name'], categories=bulan_order, ordered=True)
    bulan_df = bulan_df.sort_values('month_name')

    # Tampilkan tabel dan chart
    st.dataframe(bulan_df[['month_name', 'jumlah_transaksi']].rename(columns={'month_name': 'Bulan'}))
    st.bar_chart(bulan_df.set_index('month_name')['jumlah_transaksi'])

    # 📆 Jumlah Transaksi per Hari dalam Minggu

    st.subheader("📅 Total Transaction per Day (Monday - Sunday)")
    hari_df = (
        filtered_df.groupby('day_of_week')['no_invoice']
        .nunique()
        .reset_index(name='jumlah_transaksi')
    )

    # Agar urutan Senin ke Minggu
    order_hari = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hari_df['day_of_week'] = pd.Categorical(hari_df['day_of_week'], categories=order_hari, ordered=True)
    hari_df = hari_df.sort_values('day_of_week')

    hari_df['Hari'] = hari_df['day_of_week']

    st.dataframe(hari_df[['Hari', 'jumlah_transaksi']])
    st.bar_chart(hari_df.set_index('Hari')['jumlah_transaksi'])
