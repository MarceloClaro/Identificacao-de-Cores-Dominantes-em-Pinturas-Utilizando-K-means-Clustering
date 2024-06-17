import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.title('Identificação de Cores Dominantes em Pinturas')

# Carregar a imagem a partir do upload do usuário
st.sidebar.image("psicologia.jpg", width=200)
# Instruções na barra lateral
with st.sidebar.expander("Instruções"):
    st.markdown("""
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando o algoritmo K-means Clustering. Siga as instruções abaixo para usar o aplicativo:

    **Passos:**
    1. Faça o upload de uma imagem utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Clique no botão "Executar" para processar a imagem.

    **Detalhes Técnicos:**
    - **Upload da Imagem:** O aplicativo aceita imagens nos formatos JPG, JPEG e PNG.
    - **Número de Clusters:** Você pode selecionar entre 1 e 10 clusters para identificar diferentes cores dominantes na imagem.
    - **Resultados:** O aplicativo exibirá uma barra com as cores dominantes e um gráfico de pizza mostrando a distribuição percentual de cada cor.

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
    
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

# Selecionar o número de clusters


num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)

# Botão para executar a análise
if st.sidebar.button("Executar"):
    if uploaded_file is not None:
        # Ler a imagem do upload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Redimensionar a imagem para acelerar o processamento
        image_small = cv2.resize(image, (100, 100))

        # Converter a imagem para um array 2D
        pixels = image_small.reshape(-1, 3)

        # Aplicar K-means clustering para identificar as cores dominantes
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Calcular a porcentagem de cada cor
        counts = np.bincount(labels)
        percentages = counts / len(labels)

        # Converter cores para valores inteiros
        colors = colors.astype(int)

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
        st.error("Por favor, faça o upload de uma imagem.")
    
st.sidebar.image("logo.png", width=80)
st.sidebar.write("""
Projeto Arteterapia 
- Professores: Marcelo Claro (Coorientador).

Graduanda: Nadielle Darc Batista Dias
Whatsapp: (88)981587145

Instagram: [Equipe de Psicologia 5º Semestre](https://www.instagram.com/_psicologias/)
    """)
