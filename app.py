import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math


@st.cache_data
def load_data():
    iris = load_iris()
    # and let's just pick out the sepal length and width, 2 rows for now
    return pd.DataFrame(iris.data[:, :2])


@st.cache_data
def k_means_clustering(data, n_cluster: int, n_iterations: int):
    """
    takes in data and n_cluster, return a df with an extra column in data to specify the cluster it belongs
    data must be 2D

    representative dictionary should have the following structure:
    {
        1: (x, y),
        2: (x, y)...
    }
    """
    # choose random representatives
    x_min, x_max = data[0].min(), data[0].max()
    y_min, y_max = data[1].min(), data[1].max()
    representatives = {}
    results = []
    for i in range(n_cluster):
        representatives[i] = (
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max),
        )

    for _ in range(n_iterations):
        # points choose group based on who is the closest representative
        output = data.copy()
        output["group"] = -1
        for idx, point in output.iterrows():
            distances = [
                math.sqrt(
                    (point[0] - representatives[representative][0]) ** 2
                    + (point[1] - representatives[representative][1]) ** 2
                )
                for representative in representatives
            ]
            output.at[idx, "group"] = distances.index(min(distances))
        # representatives find their member's mean coordinate and update
        for i in range(n_cluster):
            filtered_df = output[output["group"] == i]
            representatives[i] = (filtered_df[0].mean(), filtered_df[1].mean())
        # store results
        results.append((output.copy(), representatives.copy()))

    return results


iris_df = load_data()

n_iterations = 20
n_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
iteration = st.sidebar.slider("Iteration", 0, n_iterations - 1, 0)

if "clear_cache" not in st.session_state:
    st.session_state.clear_cache = False

if st.sidebar.button("Clear cache"):
    k_means_clustering.clear()
    st.session_state.clear_cache = True

if st.session_state.clear_cache:
    results = k_means_clustering(iris_df, n_clusters, n_iterations)
    st.session_state.clear_cache = False
else:
    results = k_means_clustering(iris_df, n_clusters, n_iterations)

clustered_df, representatives = results[iteration]

fig, ax = plt.subplots()
scatter = ax.scatter(
    clustered_df[0], clustered_df[1], c=clustered_df["group"], cmap="viridis"
)
for rep in representatives.values():
    ax.scatter(rep[0], rep[1], c="red", marker="x", s=100)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Scatter plot of Sepal Length vs Sepal Width with Clusters")
legend1 = ax.legend(*scatter.legend_elements(), title="Groups")
ax.add_artist(legend1)
st.pyplot(fig)
