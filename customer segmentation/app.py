# ==========================================
# Customer Segmentation Application
# Backend + Frontend (Streamlit)
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# App Title & Description
# -----------------------------
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("Customer Segmentation Application")
st.markdown("""
This application performs **customer segmentation using K-Means clustering**.
Users can upload a retail dataset and visualize customer groups interactively.
""")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Customer Dataset (CSV format)", type=["csv"]
)

if uploaded_file is not None:

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # -----------------------------
    # Run Model Button
    # -----------------------------
    if st.button("Run Customer Segmentation"):

        # -----------------------------
        # Data Cleaning & Preprocessing
        # -----------------------------
        df = df.dropna()

        if 'CustomerID' in df.columns:
            df = df.drop(columns=['CustomerID'])

        if 'Gender' in df.columns:
            df = pd.get_dummies(df, drop_first=True)

        # -----------------------------
        # Feature Selection
        # -----------------------------
        try:
            X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        except KeyError:
            st.error("Required columns not found in dataset.")
            st.stop()

        # -----------------------------
        # Feature Scaling
        # -----------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------
        # Elbow Method
        # -----------------------------
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        st.subheader("Elbow Method")
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), wcss, marker='o')
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("WCSS")
        ax1.set_title("Elbow Method for Optimal K")
        st.pyplot(fig1)

        # -----------------------------
        # Final K-Means Model
        # -----------------------------
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        # -----------------------------
        # Cluster Visualization
        # -----------------------------
        st.subheader("Customer Segmentation Visualization")
        fig2, ax2 = plt.subplots()
        ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
        ax2.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X'
        )
        ax2.set_xlabel("Annual Income (Scaled)")
        ax2.set_ylabel("Spending Score (Scaled)")
        ax2.set_title("Customer Clusters")
        st.pyplot(fig2)

        # -----------------------------
        # Cluster Analysis
        # -----------------------------
        st.subheader("Cluster Analysis Summary")
        st.write(df.groupby('Cluster').mean())

        # -----------------------------
        # Business Insights
        # -----------------------------
        st.subheader("Business Insights")
        st.markdown("""
- **Cluster 0:** Low income – Low spending customers  
- **Cluster 1:** High income – High spending (Premium customers)  
- **Cluster 2:** High income – Low spending (Potential customers)  
- **Cluster 3:** Low income – High spending (Impulse buyers)  
- **Cluster 4:** Average income – Average spending customers  
""")

        st.success("Customer segmentation completed successfully!")

else:
    st.info("Please upload a CSV file to start.")
