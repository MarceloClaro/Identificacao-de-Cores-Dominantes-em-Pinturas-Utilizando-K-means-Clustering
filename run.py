import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Título e descrição
st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes em Pinturas</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("𝐂𝐨𝐧𝐡𝐞𝐜̧𝐚 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞𝐨𝐫𝐢𝐚𝐬, 𝐝𝐨𝐦𝐢𝐧𝐞 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞́𝐜𝐧𝐢𝐜𝐚𝐬, 𝐦𝐚𝐬 𝐚𝐨 𝐭𝐨𝐜𝐚𝐫 𝐮𝐦𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚, 𝐬𝐞𝐣𝐚 𝐚𝐩𝐞𝐧𝐚𝐬 𝐨𝐮𝐭𝐫𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚 (𝐂.𝐆. 𝐉𝐮𝐧𝐠)")
st.markdown("<hr>", unsafe_allow_html=True)

# Instruções na barra lateral
st.sidebar.image("psicologia.jpg", width=200)
with st.sidebar.expander("Instruções"):
    st.markdown("""
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando diferentes técnicas de machine learning. Siga as instruções abaixo para usar o aplicativo:

    **Passos:**
    1. Faça o upload de duas imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Escolha o modelo para a análise (K-means, PCA, ou Rede Neural).
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

# Upload das imagens pelo usuário
uploaded_files = st.sidebar.file_uploader("Escolha duas imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Selecionar o número de clusters
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)

# Selecionar o método de análise
model_choice = st.sidebar.selectbox("Escolha o Modelo de Análise", ("K-means", "PCA", "Rede Neural"))

# Botão para executar a análise
if st.sidebar.button("Executar"):
    if len(uploaded_files) == 2:
        for uploaded_file in uploaded_files:
            # Ler a imagem do upload
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Redimensionar a imagem para acelerar o processamento
            image_small = cv2.resize(image, (100, 100))

            # Converter a imagem para um array 2D
            pixels = image_small.reshape(-1, 3)

            # Aplicar o modelo selecionado
            if model_choice == "K-means":
                model = KMeans(n_clusters=num_clusters, random_state=42)
                model.fit(pixels)
                colors = model.cluster_centers_
            elif model_choice == "PCA":
                pca = PCA(n_components=2)
                pixels_pca = pca.fit_transform(pixels)
                model = KMeans(n_clusters=num_clusters, random_state=42)
                model.fit(pixels_pca)
                colors = model.cluster_centers_
                colors = pca.inverse_transform(colors)
            elif model_choice == "Rede Neural":
                # Reduzir para 2D para visualização com PCA antes de usar MLPClassifier
                pca = PCA(n_components=2)
                pixels_pca = pca.fit_transform(pixels)
                model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
                model.fit(pixels_pca, np.zeros(pixels_pca.shape[0]))  # Usamos zeros como dummy target
                labels = model.predict(pixels_pca)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_

            # Garantir que os valores estão dentro da faixa RGB
            colors = np.clip(colors, 0, 255)
            colors = colors.astype(int)

            # Calcular a porcentagem de cada cor
            labels = model.predict(pixels)
            counts = np.bincount(labels)
            percentages = counts / len(labels)

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

            # Exibir as cores dominantes e suas porcentagens
            st.write("Cores dominantes e suas porcentagens:")
            for color, percentage in dominant_colors:
                st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")
    else:
        st.error("Por favor, faça o upload de duas imagens.")

# Informações adicionais na barra lateral
st.sidebar.image("logo.png", width=80)
st.sidebar.write("""
Projeto Arteterapia 
- Professores: Marcelo Claro (Coorientador).

Graduanda: Nadielle Darc Batista Dias
Whatsapp: (88)981587145

Instagram: [Equipe de Psicologia 5º Semestre](https://www.instagram.com/_psicologias/)
""")
