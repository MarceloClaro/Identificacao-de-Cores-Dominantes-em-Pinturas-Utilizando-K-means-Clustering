import streamlit as st
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, scale=100):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (scale, scale))
    pixels = image.reshape(-1, 3)
    return image, pixels

# Function to apply K-means clustering
def apply_kmeans_clustering(pixels, n_clusters=5):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    return kmeans

# Function to apply Gaussian Mixture Model
def apply_gmm_clustering(pixels, n_clusters=5):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(pixels)
    return gmm

# Function to apply PCA
def apply_pca(pixels, n_components=3):
    pca = PCA(n_components=n_components)
    transformed_pixels = pca.fit_transform(pixels)
    return transformed_pixels

# Function to normalize and convert colors to valid RGB values
def normalize_colors(colors):
    colors = np.clip(colors, 0, 255)
    return colors.astype(int)

# Function to plot the dominant colors
def plot_dominant_colors(colors, percentages, title="Cores Dominantes"):
    colors = normalize_colors(colors)
    if colors.ndim == 2 and colors.shape[1] == 3:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
        ax.imshow([colors], aspect='auto')
        plt.title(title)
        st.pyplot(fig)
    else:
        st.error("Formato de dados inválido para a exibição de cores dominantes.")

# Function to create a pie chart of dominant colors
def plot_pie_chart(colors, percentages):
    colors = normalize_colors(colors)
    if colors.ndim == 2 and colors.shape[1] == 3:
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages],
                                          colors=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors],
                                          autopct='%1.1f%%', startangle=140)
        for text in texts:
            text.set_color('grey')
        for autotext in autotexts:
            autotext.set_color('white')
        plt.title("Distribuição das Cores Dominantes")
        st.pyplot(fig)
    else:
        st.error("Formato de dados inválido para a exibição de cores no gráfico de pizza.")

# Main function to process and analyze the image
def analyze_image(image_path, n_clusters=5, use_gmm=False, use_pca=False):
    image, pixels = load_and_preprocess_image(image_path)

    if use_pca:
        pixels = apply_pca(pixels, n_components=3)  # Ensure we keep 3 components for RGB

    if use_gmm:
        model = apply_gmm_clustering(pixels, n_clusters)
        labels = model.predict(pixels)
        colors = model.means_
    else:
        model = apply_kmeans_clustering(pixels, n_clusters)
        labels = model.labels_
        colors = model.cluster_centers_

    counts = Counter(labels)
    total_count = sum(counts.values())
    percentages = [count / total_count for count in counts.values()]

    plot_dominant_colors(colors, percentages)
    plot_pie_chart(colors, percentages)

    return colors, percentages

# Streamlit configuration
st.title("Identificação de Cores Dominantes em Pinturas")
st.markdown("Este aplicativo identifica as cores dominantes em uma pintura utilizando técnicas de clustering.")

# Upload image
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Select the number of clusters
n_clusters = st.slider("Número de Clusters", 2, 10, 5)

# Choose clustering algorithm
algorithm = st.selectbox("Escolha o Algoritmo de Clustering", ["K-means", "Gaussian Mixture Model"])

# Apply PCA option
use_pca = st.checkbox("Aplicar PCA para Redução de Dimensionalidade")

# Analyze the image when uploaded
if uploaded_file is not None:
    with st.spinner('Processando...'):
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        use_gmm = algorithm == "Gaussian Mixture Model"
        colors, percentages = analyze_image(image_path, n_clusters=n_clusters, use_gmm=use_gmm, use_pca=use_pca)

        st.success("Análise Concluída!")
        st.write("Cores dominantes e suas porcentagens:")
        for color, percentage in zip(colors, percentages):
            st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")

# Footer
st.sidebar.markdown("""
Projeto de Arteterapia
""")
