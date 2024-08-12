import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats

# Título e descrição
st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes em Pinturas</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("𝐂𝐨𝐧𝐡𝐞𝐜̧𝐚 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞𝐨𝐫𝐢𝐚𝐬, 𝐝𝐨𝐦𝐢𝐧𝐞 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞́𝐜𝐧𝐢𝐜𝐚𝐬, 𝐦𝐚𝐬 𝐚𝐨 𝐭𝐨𝐜𝐚𝐫 𝐮𝐦𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚, 𝐬𝐞𝐣𝐚 𝐚𝐩𝐞𝐧𝐚𝐬 𝐨𝐮𝐭𝐫𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚 (𝐂.𝐆. 𝐉𝐮𝐧𝐠)")
st.markdown("<hr>", unsafe_allow_html=True)

# Instruções na barra lateral
st.sidebar.image("psicologia.jpg", width=200)
with st.sidebar.expander("Instruções"):
    st.markdown("""
    **Passos:**
    1. Faça o upload de até 10 imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Se desejar, ative a opção de PCA para redução de dimensionalidade.
    4. Clique no botão "Executar" para processar as imagens.

    **Nota:** Evite usar PCA se a precisão das cores for crucial para a análise.
    """)

# Função para garantir que as cores estejam no formato adequado para o matplotlib
def validate_color(color):
    color = np.clip(np.round(color), 0, 255).astype(int)
    return color[0], color[1], color[2]  # Retorna r, g, b

# Função para calcular a distância euclidiana
def euclidean_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

# Função para encontrar a cor mais próxima e sua interpretação psicológica
def interpret_color_psychology(color):
    r, g, b = validate_color(color)
    colors_db = [
        {'color': (1, 0, 0), 'name': 'Vermelho', 'interpretation': 'Representa paixão, energia, e agressividade.'},
        {'color': (0, 0, 1), 'name': 'Azul', 'interpretation': 'Simboliza calma, confiança e harmonia.'},
        {'color': (1, 1, 0), 'name': 'Amarelo', 'interpretation': 'Ligado à criatividade, intelecto e alegria.'},
        {'color': (0, 1, 0), 'name': 'Verde', 'interpretation': 'Associado à natureza, crescimento e estabilidade.'},
        {'color': (0, 0, 0), 'name': 'Preto', 'interpretation': 'Representa poder, mistério, e morte.'},
        {'color': (1, 1, 1), 'name': 'Branco', 'interpretation': 'Simboliza pureza, inocência, e renovação.'},
        {'color': (0.5, 0.5, 0.5), 'name': 'Cinza', 'interpretation': 'Simboliza neutralidade, sabedoria e maturidade.'},
        {'color': (0.5, 0, 0.5), 'name': 'Roxo', 'interpretation': 'Ligado à espiritualidade, transformação e mistério.'},
        {'color': (1, 0.5, 0), 'name': 'Laranja', 'interpretation': 'Representa energia, criatividade, e entusiasmo.'},
        {'color': (0.5, 0.75, 0.5), 'name': 'Verde-claro', 'interpretation': 'Associado à tranquilidade, frescor e harmonia.'},
        {'color': (0.4, 0.7, 0.6), 'name': 'Turquesa', 'interpretation': 'Representa equilíbrio emocional e tranquilidade.'},
        {'color': (0.7, 0.2, 0.2), 'name': 'Carmesim', 'interpretation': 'Representa paixão, intensidade e força.'},
        {'color': (0.6, 0.5, 0.8), 'name': 'Violeta', 'interpretation': 'Associado à intuição, inovação, e misticismo.'},
        {'color': (0.7, 0.7, 0.7), 'name': 'Prata', 'interpretation': 'Conectado à pureza, precisão, e integridade.'},
        {'color': (0.8, 0.4, 0), 'name': 'Âmbar', 'interpretation': 'Simboliza calor, segurança e aconchego.'},
        {'color': (0.2, 0.8, 0.2), 'name': 'Verde-oliva', 'interpretation': 'Representa paz, diplomacia e harmonia.'},
        {'color': (0.4, 0.2, 0.2), 'name': 'Marrom', 'interpretation': 'Simboliza estabilidade, confiabilidade e segurança.'},
        {'color': (0.5, 0.4, 0.4), 'name': 'Bege', 'interpretation': 'Representa simplicidade, confiabilidade e tradição.'},
        {'color': (1, 0.4, 0.7), 'name': 'Rosa', 'interpretation': 'Representa carinho, afeto e vulnerabilidade.'},
        {'color': (0.6, 0.4, 0.2), 'name': 'Sépia', 'interpretation': 'Evoke nostalgia and antiquity.'},
        {'color': (0.4, 0.2, 0.6), 'name': 'Lavanda', 'interpretation': 'Represents serenity, grace, and elegance.'},
        {'color': (0.3, 0.3, 0.7), 'name': 'Índigo', 'interpretation': 'Associated with deep thoughts and spirituality.'},
        {'color': (0.3, 0.6, 0.3), 'name': 'Verde-musgo', 'interpretation': 'Represents resilience, endurance, and balance.'},
    ]
    
    closest_color = min(colors_db, key=lambda c: euclidean_distance((r, g, b), c['color']))
    return closest_color  # Retorna o dicionário inteiro

# Configuração do streamlit
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
                from sklearn.decomposition import PCA
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
                color = validate_color(color)
                dominant_colors.append((color, percentage))
                closest_color_info = interpret_color_psychology(color)
                interpretations.append(closest_color_info)

            # Visualização das cores dominantes
            fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
            bar_width = 1
            index = np.arange(len(colors))
            ax.bar(index, [1] * len(colors), color=[(r, g, b) for (r, g, b) in colors], width=bar_width)
            ax.set_xticks(index)
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            plt.title("Cores Dominantes")
            st.pyplot(fig)

            # Gráfico de pizza das cores dominantes
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages],
                                              colors=[(r, g, b) for (r, g, b) in colors],
                                              autopct='%1.1f%%', startangle=140)
            for text in texts:
                text.set_color('grey')
            for autotext in autotexts:
                autotext.set_color('white')
            plt.title("Distribuição das Cores Dominantes")
            st.pyplot(fig)

            # Exibir cores dominantes e suas interpretações psicológicas
            st.write("**Cores dominantes e interpretações psicológicas:**")
            for i, (color, percentage) in enumerate(dominant_colors):
                color_info = interpretations[i]
                st.write(f"**Cor {i+1}:** {color_info['name']} ({color}) - {percentage:.2%}")
                st.write(f"**Interpretação Psicológica:** {color_info['interpretation']}")
                st.write("<hr>", unsafe_allow_html=True)

    else:
        st.error("Por favor, faça o upload de pelo menos uma imagem.")

# Informações adicionais na barra lateral
st.sidebar.image("logo.png", width=80)
st.sidebar.write("""
Projeto Arteterapia 
- Professores: Marcelo Claro (Coorientador).

Graduanda: Nadielle Darc Batista Dias
Whatsapp: (88)981587145

Instagram: [Equipe de Psicologia 5º Semestre](https://www.instagram.com/_psicologias/)
""")





                                 
