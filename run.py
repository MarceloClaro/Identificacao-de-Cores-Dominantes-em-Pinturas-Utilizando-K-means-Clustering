import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats

# Função para garantir que as cores estejam no formato adequado para o matplotlib
def validate_color(color):
    color = np.round(color).astype(int)
    if len(color) == 2:
        color = np.append(color, 0)  # Adiciona o valor do azul como 0 se a cor tiver apenas dois componentes
    elif len(color) == 1:
        color = np.append(color, [0, 0])  # Adiciona valores para verde e azul se a cor tiver apenas um componente
    return np.clip(color, 0, 255) / 255  # Normaliza as cores para o intervalo [0, 1]

# Função para mapear cores para a interpretação psicológica
def interpret_color_psychology(color):
    r, g, b = validate_color(color)
    if r > 0.59 and g < 0.39 and b < 0.39:
        return "Vermelho: Amor, Ódio, Perigo, Dinamismo"
    elif b > 0.59 and g < 0.39 and r < 0.39:
        return "Azul: Calma, Harmonia, Fidelidade"
    elif r > 0.59 and g > 0.59 and b < 0.39:
        return "Amarelo: Otimismo, Traição, Inteligência"
    elif g > 0.59 and r < 0.39 and b < 0.39:
        return "Verde: Fertilidade, Esperança, Saúde"
    elif r < 0.2 and g < 0.2 and b < 0.2:
        return "Preto: Poder, Morte, Elegância"
    elif r > 0.78 and g > 0.78 and b > 0.78:
        return "Branco: Inocência, Pureza, Bondade"
    else:
        return "Cor não identificada. Consulte manualmente."

st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes e Psicologia das Cores</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader("Escolha até 10 imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)
apply_pca = st.sidebar.checkbox("Aplicar PCA para Redução de Dimensionalidade")

def calculate_statistics(pixels, labels, cluster_centers):
    statistics = []
    for i in range(len(cluster_centers)):
        cluster_pixels = pixels[labels == i]
        mean_color = np.mean(cluster_pixels, axis=0)
        std_dev = np.std(cluster_pixels, axis=0)
        margin_of_error = stats.sem(cluster_pixels, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(cluster_pixels)-1)
        conf_interval = np.vstack((mean_color - margin_of_error, mean_color + margin_of_error)).T
        statistics.append({
            'mean': mean_color,
            'std_dev': std_dev,
            'margin_of_error': margin_of_error,
            'confidence_interval': conf_interval
        })
    return statistics

if st.sidebar.button("Executar"):
    if len(uploaded_files) >= 1:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_small = cv2.resize(image, (100, 100))
            pixels = image_small.reshape(-1, 3)

            if apply_pca:
                pca = PCA(n_components=2)
                pixels = pca.fit_transform(pixels)
                st.write("PCA aplicada para redução de dimensionalidade.")

            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_

            counts = np.bincount(labels)
            percentages = counts / len(labels)

            statistics = calculate_statistics(pixels, labels, colors)
            colors = colors.astype(int)

            st.image(image, caption='Imagem Analisada', use_column_width=True)

            dominant_colors = []
            interpretations = []
            for color, percentage in zip(colors, percentages):
                color = validate_color(color)  # Certifique-se de que a cor esteja no formato correto
                dominant_colors.append((color, percentage))
                interpretations.append(interpret_color_psychology(color))

            fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
            for sp in ax.spines.values():
                sp.set_visible(False)
            bar_width = 1
            index = np.arange(len(colors))
            ax.bar(index, [1] * len(colors), color=[validate_color(color) for color in colors], width=bar_width)
            ax.set_xticks(index)
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            plt.title("Cores Dominantes")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages],
                                              colors=[validate_color(color) for color in colors],
                                              autopct='%1.1f%%', startangle=140)
            for text in texts:
                text.set_color('grey')
            for autotext in autotexts:
                autotext.set_color('white')
            plt.title("Distribuição das Cores Dominantes")
            st.pyplot(fig)

            st.write("Cores dominantes e interpretações psicológicas:")
            for i, (color, percentage) in enumerate(dominant_colors):
                st.write(f"**Cor {i+1}:** {color} - {percentage:.2%}")
                st.write(f"**Interpretação Psicológica:** {interpretations[i]}")

    else:
        st.error("Por favor, faça o upload de pelo menos uma imagem.")
