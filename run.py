import streamlit as st
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# Função para carregar e processar a imagem
def load_and_preprocess_image(image_path, scale=100):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (scale, scale))
    pixels = image.reshape(-1, 3)
    return image, pixels

# Função para aplicar o K-means clustering
def apply_kmeans_clustering(pixels, n_clusters=5):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    return kmeans

# Função para aplicar o Gaussian Mixture Model
def apply_gmm_clustering(pixels, n_clusters=5):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(pixels)
    return gmm

# Função para aplicar PCA
def apply_pca(pixels, n_components=2):
    pca = PCA(n_components=n_components)
    transformed_pixels = pca.fit_transform(pixels)
    return transformed_pixels

# Função para plotar as cores dominantes
def plot_dominant_colors(colors, percentages, title="Cores Dominantes"):
    # Verificar se o formato do array de cores é correto
    if colors.ndim == 2 and colors.shape[1] == 3:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
        ax.imshow([colors], aspect='auto')
        plt.title(title)
        st.pyplot(fig)
    else:
        st.error("Formato de dados inválido para a exibição de cores dominantes.")

# Função para criar um gráfico de pizza das cores dominantes
def plot_pie_chart(colors, percentages):
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

# Função principal para processamento e análise da imagem
def analyze_image(image_path, n_clusters=5, use_gmm=False, use_pca=False):
    image, pixels = load_and_preprocess_image(image_path)

    if use_pca:
        pixels = apply_pca(pixels, n_components=2)

    if use_gmm:
        model = apply_gmm_clustering(pixels, n_clusters)
        labels = model.predict(pixels)
        colors = model.means_.astype(int)
    else:
        model = apply_kmeans_clustering(pixels, n_clusters)
        labels = model.labels_
        colors = model.cluster_centers_.astype(int)

    counts = Counter(labels)
    total_count = sum(counts.values())
    percentages = [count / total_count for count in counts.values()]

    plot_dominant_colors(colors, percentages)
    plot_pie_chart(colors, percentages)

    return colors, percentages

# Configuração do Streamlit
st.title("Identificação de Cores Dominantes em Pinturas")
st.markdown("Este aplicativo identifica as cores dominantes em uma pintura utilizando técnicas de clustering.")

# Carregar imagem do usuário
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Selecionar o número de clusters
n_clusters = st.slider("Número de Clusters", 2, 10, 5)

# Escolher algoritmo de clustering
algorithm = st.selectbox("Escolha o Algoritmo de Clustering", ["K-means", "Gaussian Mixture Model"])

# Escolher se deseja usar PCA
use_pca = st.checkbox("Aplicar PCA para Redução de Dimensionalidade")

# Executar a análise quando a imagem for carregada
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
