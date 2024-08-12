import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando os algoritmos K-means Clustering e PCA. Siga as instruções abaixo para usar o aplicativo:

    **Passos:**
    1. Faça o upload de duas imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters e componentes principais para a segmentação de cores utilizando os controles deslizantes.
    3. Clique no botão "Executar" para processar as imagens.

    **Detalhes Técnicos:**
    - **Upload da Imagem:** O aplicativo aceita imagens nos formatos JPG, JPEG e PNG.
    - **Número de Clusters e Componentes Principais:** Você pode selecionar entre 1 e 10 clusters para identificar diferentes cores dominantes na imagem e entre 1 e 3 componentes principais para simplificação da imagem usando PCA.
    - **Resultados:** O aplicativo exibirá uma barra com as cores dominantes e um gráfico de pizza mostrando a distribuição percentual de cada cor para cada imagem.

    **Inovações:**
    - Utilização de técnicas de ciência de dados para análise de imagens.
    - Interface interativa que permite personalização pelo usuário.

    **Pontos Positivos:**
    - Fácil de usar e intuitivo, mesmo para usuários sem experiência prévia em processamento de imagens.
    - Resultados visuais claros e informativos.

    **Limitações:**
    - O tempo de processamento pode variar dependendo do tamanho da imagem.
    - A precisão da segmentação pode ser afetada por imagens com muitas cores semelhantes.

    **Importância de Ter Instruções:**
    - As instruções claras garantem que o aplicativo possa ser utilizado eficientemente por qualquer pessoa, independentemente do seu nível de conhecimento técnico.

    Em resumo, este aplicativo é uma ferramenta poderosa para análise de cores em pinturas, utilizando técnicas avançadas de aprendizado de máquina para fornecer resultados precisos e visualmente agradáveis.
    """)

# Upload das imagens pelo usuário
uploaded_files = st.sidebar.file_uploader("Escolha duas imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Selecionar o número de clusters e componentes principais
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)
num_components = st.sidebar.slider("Número de Componentes Principais (PCA)", 1, 3, 3)

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

            # Aplicar PCA para redução de dimensionalidade
            pca = PCA(n_components=num_components)
            pca_pixels = pca.fit_transform(pixels)

            # Aplicar K-means clustering para identificar as cores dominantes
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(pca_pixels)
            colors = pca.inverse_transform(kmeans.cluster_centers_)
            labels = kmeans.labels_

            # Calcular a porcentagem de cada cor
            counts = np.bincount(labels)
            percentages = counts / len(labels)

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
