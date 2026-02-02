import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("Customer Segmentation Analysis")

st.write("Upload customer data (CSV)")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.write("Data Preview")
    st.dataframe(df.head())

    features = df.select_dtypes(include=["int64", "float64"])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    k = st.slider("Number of clusters", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = model.fit_predict(scaled_data)

    st.write("Clustered Data")
    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=df["Cluster"])
    st.pyplot(fig)
