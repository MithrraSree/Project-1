import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Project 1")


st.title("Decoding Phone Usage Patterns in India")
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an option:", ['EDA', 'Classification', 'Clustering'])

df = pd.read_csv("C:/Users/Mithrra Sree/Downloads/phone_usage_india.csv")

# EDA Section
if option == 'EDA':
    st.header("Exploratory Data Analysis")
    st.write("### Sample Data")
    st.write(df.head())

    # Visualization Example
    st.write("### Data Usage vs. Screen Time")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Data Usage (GB/month)', y='Screen Time (hrs/day)', data=df)
    st.pyplot(plt)

# Classification Section
elif option == 'Classification':
    st.header("Predict Primary Use")

    # User Input
    st.write("Enter device details for prediction:")
    screen_time = st.number_input("Screen Time (hrs/day)", min_value=0.0, max_value=24.0)
    data_usage = st.number_input("Data Usage (GB/month)", min_value=0.0)
    apps_installed = st.number_input("Number of Apps Installed", min_value=0)
    recharge_cost = st.number_input("Monthly Recharge Cost (INR)", min_value=0)

    # Simple Model for Demonstration
    X = df[['Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Number of Apps Installed', 'Monthly Recharge Cost (INR)']]
    y = df['Primary Use']
    model = LogisticRegression()
    model.fit(X, y)

    if st.button("Predict"):
        prediction = model.predict([[screen_time, data_usage, apps_installed, recharge_cost]])
        st.success(f"The predicted primary use is: **{prediction[0]}**")

# Clustering Section
elif option == 'Clustering':
    st.header("Clustering Analysis")

    # Perform Clustering
    X_clustering = df[['Screen Time (hrs/day)', 'Data Usage (GB/month)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clustering)

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    st.write("### Cluster Visualization")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Data Usage (GB/month)', y='Screen Time (hrs/day)', hue='Cluster', data=df, palette='viridis')
    st.pyplot(plt)