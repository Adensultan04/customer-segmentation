import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation Analysis using K-Means")
st.write("This application segments customers based on their purchasing behavior using unsupervised learning.")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # DATA CLEANING & PREPROCESSING
    # -------------------------------
    st.subheader("Data Cleaning & Preprocessing")

    st.write("Checking missing values:")
    st.write(df.isnull().sum())

    # Select numeric features only
    features = df.select_dtypes(include=["int64", "float64"])

    st.write("Selected Numerical Features:")
    st.dataframe(features.head())

    # -------------------------------
    # FEATURE SCALING
    # -------------------------------
    st.subheader("Feature Scaling")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    st.write("Features scaled using StandardScaler (mean=0, std=1).")

    # -------------------------------
    # ELBOW METHOD
    # -------------------------------
    st.subheader("Elbow Method to Determine Optimal Clusters")

    wcss = []
    K = range(1, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(K, wcss, marker='o')
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Method")
    st.pyplot(fig1)

    st.write("The elbow point indicates the optimal number of clusters.")

    # -------------------------------
    # K-MEANS MODEL
    # -------------------------------
    st.sidebar.header("Model Configuration")
    k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    df["Cluster"] = clusters

    # -------------------------------
    # RESULTS
    # -------------------------------
    st.subheader("Clustered Dataset")
    st.dataframe(df.head())

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    st.subheader("Cluster Visualization")

    if features.shape[1] >= 2:
        fig2, ax2 = plt.subplots()
        ax2.scatter(
            scaled_data[:, 0],
            scaled_data[:, 1],
            c=df["Cluster"],
        )
        ax2.set_xlabel(features.columns[0])
        ax2.set_ylabel(features.columns[1])
        ax2.set_title("Customer Segments")
        st.pyplot(fig2)
    else:
        st.warning("Need at least 2 numerical features for visualization.")

    # -------------------------------
    # CLUSTER INTERPRETATION
    # -------------------------------
    st.subheader("Cluster Interpretation & Business Insights")

    for i in sorted(df["Cluster"].unique()):
        st.markdown(f"### Cluster {i}")
        st.write(df[df["Cluster"] == i][features.columns].mean())

    st.success("Customer Segmentation Completed Successfully ✅")

else:
    st.info("Please upload a CSV file to begin.")
mm
