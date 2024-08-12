import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats

# Título e descrição
st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes em Pinturas</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("𝐂𝐨𝐧𝐡𝐞𝐜̧𝐚 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞𝐨𝐫𝐢𝐚𝐬, 𝐝𝐨𝐦𝐢𝐧𝐞 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞́𝐜𝐧𝐢𝐜𝐚𝐬, 𝐦𝐚𝐬 𝐚𝐨 𝐭𝐨𝐜𝐚𝐫 𝐮𝐦𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚, 𝐬𝐞𝐣𝐚 𝐚𝐩𝐞𝐧𝐚𝐬 𝐨𝐮𝐭𝐫𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚 (𝐂.𝐆. 𝐉𝐮𝐧𝐠)")
st.markdown("<hr>", unsafe_allow_html=True)

# Instruções na barra lateral
st.sidebar.image("psicologia.jpg", width=200)
with st.sidebar.expander("Instruções"):
    st.markdown("""
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando o algoritmo K-means Clustering, Análise de Componentes Principais (PCA) e Rede Neural. Siga as instruções abaixo para usar o aplicativo:

    **Passos:**
    1. Faça o upload de até 10 imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Escolha se deseja aplicar PCA para redução de dimensionalidade ou usar uma Rede Neural para classificação.
    4. Clique no botão "Executar" para processar as imagens.

    **Inovações:**
    - Integração de técnicas de ciência de dados para análise de imagens.
    - Interface interativa que permite personalização pelo usuário.

    **Pontos Positivos:**
    - Fácil de usar e intuitivo, mesmo para usuários sem experiência prévia em processamento de imagens.
    - Resultados visuais claros e informativos.

    **Limitações:**
    - O tempo de processamento pode variar dependendo do tamanho da imagem.
    - A precisão da segmentação pode ser afetada por imagens com muitas cores semelhantes.

    Este aplicativo é uma ferramenta poderosa para análise de cores em pinturas, utilizando técnicas avançadas de aprendizado de máquina para fornecer resultados precisos e visualmente agradáveis.
    """)

# Upload das imagens pelo usuário (aceitando de 1 a 10 imagens)
uploaded_files = st.sidebar.file_uploader("Escolha de 1 a 10 imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Selecionar o número de clusters
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)

# Checkbox para aplicar PCA
apply_pca = st.sidebar.checkbox("Aplicar PCA para redução de dimensionalidade", value=True)

# Checkbox para usar Rede Neural
use_nn = st.sidebar.checkbox("Usar Rede Neural para classificação de cores", value=False)

# Função para calcular a margem de erro e outros estatísticos
def calculate_statistics(pixels, labels, cluster_centers):
    statistics = []
    for i in range(cluster_centers.shape[0]):
        cluster_pixels = pixels[labels == i]
        mean_color = cluster_centers[i]
        std_dev = np.std(cluster_pixels, axis=0)
        margin_of_error = stats.sem(cluster_pixels, axis=0) * stats.t.ppf((1 + 0.95) / 2., cluster_pixels.shape[0]-1)
        conf_interval = [mean_color - margin_of_error, mean_color + margin_of_error]
        statistics.append({
            'mean': mean_color,
            'std_dev': std_dev,
            'margin_of_error': margin_of_error,
            'conf_interval': conf_interval
        })
    return statistics

# Botão para executar a análise
if st.sidebar.button("Executar"):
    if 1 <= len(uploaded_files) <= 10:
        for uploaded_file in uploaded_files:
            # Ler a imagem do upload
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Redimensionar a imagem para acelerar o processamento
            image_small = cv2.resize(image, (100, 100))

            # Converter a imagem para um array 2D
            pixels = image_small.reshape(-1, 3)

            if apply_pca:
                pca = PCA(n_components=2)
                pixels = pca.fit_transform(pixels)

            if use_nn:
                # Aplicar Rede Neural para identificar as cores dominantes
                nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
                nn_model.fit(pixels, np.random.randint(0, num_clusters, size=pixels.shape[0]))
                labels = nn_model.predict(pixels)
                colors = np.array([pixels[labels == i].mean(axis=0) for i in range(num_clusters)])
            else:
                # Aplicar K-means clustering para identificar as cores dominantes
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_

            if apply_pca:
                colors = pca.inverse_transform(colors)

            colors = np.clip(colors, 0, 255)  # Garantir que os valores estão dentro da faixa RGB

            # Calcular a porcentagem de cada cor
            counts = np.bincount(labels)
            percentages = counts / len(labels)

            # Calcular estatísticas
            statistics = calculate_statistics(pixels, labels, colors)

            # Converter cores para valores inteiros
            colors = colors.astype(int)

            # Mostrar a imagem original
            st.image(image, caption='Imagem Analisada', use_column_width=True)

            # Mostrar as cores dominantes e suas porcentagens
            dominant_colors = []
            for color, percentage in zip(colors, percentages):
                dominant_colors.append((color, percentage))

            # Plotar as cores dominantes como uma barra
            fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.imshow([colors], aspect='auto')
            plt.title("Cores Dominantes")
            st.pyplot(fig)

            # Plotar gráfico de pizza das cores dominantes
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

            # Exibir as cores dominantes, suas porcentagens, e estatísticas
            st.write("Cores dominantes, suas porcentagens e estatísticas:")
            for i, (color, percentage) in enumerate(dominant_colors):
                st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")
                st.write(f" - Desvio Padrão: {statistics[i]['std_dev']}")
                st.write(f" - Margem de Erro: {statistics[i]['margin_of_error']}")
                st.write(f" - Intervalo de Confiança (95%): {statistics[i]['conf_interval']}")

    else:
        st.error("Por favor, faça o upload de 1 a 10 imagens.")

# Informações adicionais na barra lateral
st.sidebar.image("logo.png", width=80)
st.sidebar.write("""
Projeto Arteterapia 
- Professores: Marcelo Claro (Coorientador).

Graduanda: Nadielle Darc Batista Dias
Whatsapp: (88)981587145

Instagram: [Equipe de Psicologia 5º Semestre](https://www.instagram.com/_psicologias/)
""")
