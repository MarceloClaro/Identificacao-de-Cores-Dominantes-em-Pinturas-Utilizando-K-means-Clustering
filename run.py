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
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando o algoritmo K-means Clustering. Siga as instruções abaixo para usar o aplicativo:

    **Passos:**
    1. Faça o upload de uma ou até dez imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Clique no botão "Executar" para processar as imagens.

    **Detalhes Técnicos:**
    - **Upload da Imagem:** O aplicativo aceita imagens nos formatos JPG, JPEG e PNG.
    - **Número de Clusters:** Você pode selecionar entre 1 e 10 clusters para identificar diferentes cores dominantes na imagem.
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
uploaded_files = st.sidebar.file_uploader("Escolha até 10 imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Selecionar o número de clusters
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)

# Função para calcular estatísticas
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

# Função para verificar se a imagem foi carregada corretamente
def load_image(uploaded_file):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None:
            st.error("Não foi possível processar a imagem.")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
        return None

# Botão para executar a análise
if st.sidebar.button("Executar"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = load_image(uploaded_file)
            if image is None:
                continue

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

            # Calcular estatísticas
            statistics = calculate_statistics(pixels, labels, colors)

            # Converter cores para valores inteiros
            colors = colors.astype(int)

            # Mostrar a imagem original
            st.image(image, caption='Imagem Analisada', use_column_width=True)

            # Mostrar as cores dominantes e suas porcentagens
            dominant_colors = [(color, percentage) for color, percentage in zip(colors, percentages)]

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

            # Mostrar estatísticas adicionais com explicações
            st.write("### Estatísticas das Cores Dominantes:")

            # Gráfico da média das cores
            fig, ax = plt.subplots(figsize=(10, 5))
            mean_colors = np.array([stats['mean'] for stats in statistics])
            ax.bar(range(num_clusters), mean_colors[:, 0], color=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors], label='R')
            ax.bar(range(num_clusters), mean_colors[:, 1], color=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors], bottom=mean_colors[:, 0], label='G')
            ax.bar(range(num_clusters), mean_colors[:, 2], color=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors], bottom=mean_colors[:, 0] + mean_colors[:, 1], label='B')
            ax.set_xticks(range(num_clusters))
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            ax.set_ylabel('Valor de Cor (RGB)')
            ax.set_title('Média das Cores Dominantes')
            ax.legend(loc='upper right')
            st.pyplot(fig)

            # Gráfico do desvio padrão das cores
            fig, ax = plt.subplots(figsize=(10, 5))
            std_devs = np.array([stats['std_dev'] for stats in statistics])
            ax.bar(range(num_clusters), std_devs[:, 0], color='r', label='Desvio Padrão R')
            ax.bar(range(num_clusters), std_devs[:, 1], color='g', bottom=std_devs[:, 0], label='Desvio Padrão G')
            ax.bar(range(num_clusters), std_devs[:, 2], color='b', bottom=std_devs[:, 0] + std_devs[:, 1], label='Desvio Padrão B')
            ax.set_xticks(range(num_clusters))
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            ax.set_ylabel('Desvio Padrão de Cor (RGB)')
            ax.set_title('Desvio Padrão das Cores Dominantes')
            ax.legend(loc='upper right')
            st.pyplot(fig)

            # Gráfico do intervalo de confiança das cores
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, stats in enumerate(statistics):
                ci = stats['confidence_interval']
                ax.errorbar(i, stats['mean'][0], yerr=[[stats['mean'][0] - ci[0, 0]], [ci[1, 0] - stats['mean'][0]]], fmt='o', color='r', label='CI R' if i == 0 else "")
                ax.errorbar(i, stats['mean'][1], yerr=[[stats['mean'][1] - ci[0, 1]], [ci[1, 1] - stats['mean'][1]]], fmt='o', color='g', label='CI G' if i == 0 else "")
                ax.errorbar(i, stats['mean'][2], yerr=[[stats['mean'][2] - ci[0, 2]], [ci[1, 2] - stats['mean'][2]]], fmt='o', color='b', label='CI B' if i == 0 else "")
            ax.set_xticks(range(num_clusters))
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            ax.set_ylabel('Valor de Cor (RGB)')
            ax.set_title('Intervalo de Confiança das Cores Dominantes (95%)')
            ax.legend(loc='upper right')
            st.pyplot(fig)

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

