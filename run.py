import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats

# Corrigindo o erro de descompactação da cor
def map_color_to_archetype(color):
    color = np.round(color).astype(int)  # Garante que os valores da cor sejam inteiros
    
    # Verifica a quantidade de dimensões
    if len(color) == 3:
        r, g, b = color
    elif len(color) == 2:
        r, g = color
        b = 0  # Define um valor padrão para o canal azul
    else:
        return "Cor não identificada. Consulte manualmente."
    
    if r > 150 and g < 100 and b < 100:
        return "Paixão, Ação, Energia (Vermelho)"
    elif b > 150 and g < 100 and r < 100:
        return "Calma, Sabedoria, Proteção (Azul)"
    elif r > 150 and g > 150 and b < 100:
        return "Otimismo, Criatividade, Poder (Amarelo)"
    elif g > 150 and r < 100 and b < 100:
        return "Crescimento, Cura, Abundância (Verde)"
    elif r < 50 and g < 50 and b < 50:
        return "Mistério, Inconsciente, Transformação (Preto)"
    elif r > 200 and g > 200 and b > 200:
        return "Pureza, Clareza, Novos Começos (Branco)"
    else:
        return "Outros - Interpretar com base em contexto"

# Adicionando mapeamento baseado na Psicologia das Cores
def interpret_color_psychology(color):
    color = np.round(color).astype(int)
    
    # Verifica a quantidade de dimensões
    if len(color) == 3:
        r, g, b = color
    elif len(color) == 2:
        r, g = color
        b = 0  # Define um valor padrão para o canal azul
    else:
        return "Cor não identificada. Consulte manualmente."
    
    if r > 150 and g < 100 and b < 100:
        return "Vermelho: Amor, Ódio, Perigo, Dinamismo"
    elif b > 150 and g < 100 and r < 100:
        return "Azul: Calma, Harmonia, Fidelidade"
    elif r > 150 and g > 150 and b < 100:
        return "Amarelo: Otimismo, Traição, Inteligência"
    elif g > 150 and r < 100 and b < 100:
        return "Verde: Fertilidade, Esperança, Saúde"
    elif r < 50 and g < 50 and b < 50:
        return "Preto: Poder, Morte, Elegância"
    elif r > 200 and g > 200 and b > 200:
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
            archetypes = []
            interpretations = []
            for color, percentage in zip(colors, percentages):
                dominant_colors.append((color, percentage))
                archetypes.append(map_color_to_archetype(color))
                interpretations.append(interpret_color_psychology(color))

            fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
            for sp in ax.spines.values():
                sp.set_visible(False)
            bar_width = 1
            index = np.arange(len(colors))
            ax.bar(index, [1] * len(colors), color=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors], width=bar_width)
            ax.set_xticks(index)
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            plt.title("Cores Dominantes")
            st.pyplot(fig)

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

            st.write("Cores dominantes, arquétipos e interpretações psicológicas:")
            for i, (color, percentage) in enumerate(dominant_colors):
                st.write(f"**Cor {i+1}:** {color} - {percentage:.2%}")
                st.write(f"**Arquétipo:** {archetypes[i]}")
                st.write(f"**Interpretação Psicológica:** {interpretations[i]}")

    else:
        st.error("Por favor, faça o upload de pelo menos uma imagem.")
