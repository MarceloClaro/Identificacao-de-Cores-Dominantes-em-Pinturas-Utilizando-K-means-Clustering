import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Função para processar a imagem e identificar cores dominantes
def process_image(image, n_clusters):
    # Redimensionar a imagem para acelerar o processamento
    image_small = cv2.resize(image, (100, 100))
    
    # Converter a imagem para um array 2D
    pixels = image_small.reshape(-1, 3)
    
    # Aplicar K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Calcular a porcentagem de cada cor
    counts = np.bincount(labels)
    percentages = counts / len(labels)
    
    # Converter cores para valores inteiros
    colors = colors.astype(int)
    
    return colors, percentages

# Configurar a barra lateral do Streamlit
st.sidebar.title("Análise de Cores Dominantes")
st.sidebar.write("Faça upload da sua imagem e escolha o número de clusters.")

# Upload da imagem
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Seleção do número de clusters
n_clusters = st.sidebar.slider("Número de clusters", min_value=1, max_value=10, value=5)

# Botão para executar a análise
if st.sidebar.button("Executar Análise"):
    if uploaded_file is not None:
        # Ler a imagem usando OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Processar a imagem
        colors, percentages = process_image(image, n_clusters)
        
        # Mostrar as cores dominantes
        st.image(image, caption="Imagem Original", use_column_width=True)
        
        # Plotar as cores dominantes como uma barra
        fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
        for sp in ax.spines.values():
            sp.set_visible(False)
        
        ax.imshow([colors], aspect='auto')
        st.pyplot(fig)
        
        # Plotar gráfico de pizza das cores dominantes
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(percentages,
                                          labels=[f'{int(p*100)}%' for p in percentages],
                                          colors=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors],
                                          autopct='%1.1f%%', startangle=140)
        
        for text in texts:
            text.set_color('grey')
        
        for autotext in autotexts:
            autotext.set_color('white')
        
        st.pyplot(fig)
        
        # Exibir as cores dominantes e suas porcentagens
        st.write("Cores dominantes e suas porcentagens:")
        for color, percentage in zip(colors, percentages):
            st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")
    else:
        st.write("Por favor, faça o upload de uma imagem.")