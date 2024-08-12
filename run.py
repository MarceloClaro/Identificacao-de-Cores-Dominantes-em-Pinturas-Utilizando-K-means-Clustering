import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Function to preprocess the image
def preprocess_image(image_path, apply_pca=False, n_components=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_small = cv2.resize(image, (100, 100))
    pixels = image_small.reshape(-1, 3)

    if apply_pca:
        pca = PCA(n_components=n_components)
        pixels = pca.fit_transform(pixels)
        # After PCA, clip the values to ensure they're in a valid RGB range
        pixels = np.clip(pixels, 0, 255)

    return pixels

# Function to analyze the image and identify dominant colors
def analyze_image(image_path, n_clusters=5, apply_pca=False):
    pixels = preprocess_image(image_path, apply_pca)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    counts = np.bincount(labels)
    percentages = counts / len(labels)

    # Clip and convert colors to valid integers
    colors = np.clip(colors, 0, 255).astype(int)

    return colors, percentages

# Function to plot the dominant colors
def plot_dominant_colors(colors, percentages):
    fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    ax.imshow([colors], aspect='auto')
    plt.title("Cores Dominantes")
    st.pyplot(fig)

# Function to create a pie chart of the dominant colors
def plot_pie_chart(colors, percentages):
    colors_hex = [f'#{int(r):02x}{int(g):02x}{int(b):02x}' for r, g, b in colors]
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages],
                                      colors=colors_hex, autopct='%1.1f%%', startangle=140)
    for text in texts:
        text.set_color('grey')
    for autotext in autotexts:
        autotext.set_color('white')
    plt.title("Distribuição das Cores Dominantes")
    st.pyplot(fig)

# Streamlit configuration
st.title("Identificação de Cores Dominantes em Pinturas")
st.markdown("Este aplicativo identifica as cores dominantes em uma pintura usando o algoritmo K-means.")

# Upload the image
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    image_path = "uploaded_image.jpg"

    # Select the number of clusters
    n_clusters = st.slider("Número de Clusters", min_value=2, max_value=10, value=5)

    # Apply PCA or not
    apply_pca = st.checkbox("Aplicar PCA para redução de dimensionalidade")

    # Analyze the image
    colors, percentages = analyze_image(image_path, n_clusters=n_clusters, apply_pca=apply_pca)

    # Display the dominant colors
    plot_dominant_colors(colors, percentages)
    plot_pie_chart(colors, percentages)

    st.success("Análise Concluída!")
    st.write("Cores dominantes e suas porcentagens:")
    for color, percentage in zip(colors, percentages):
        st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")
