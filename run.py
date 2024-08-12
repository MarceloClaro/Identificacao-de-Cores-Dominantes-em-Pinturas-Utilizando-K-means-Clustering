import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats

# Título e descrição
st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes em Pinturas com Estatísticas</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("𝐂𝐨𝐧𝐡𝐞𝐜̧𝐚 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞𝐨𝐫𝐢𝐚𝐬, 𝐝𝐨𝐦𝐢𝐧𝐞 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞́𝐜𝐧𝐢𝐜𝐚𝐬, 𝐦𝐚𝐬 𝐚𝐨 𝐭𝐨𝐜𝐚𝐫 𝐮𝐦𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚, 𝐬𝐞𝐣𝐚 𝐚𝐩𝐞𝐧𝐚𝐬 𝐨𝐮𝐭𝐫𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚 (𝐂.𝐆. 𝐉𝐮𝐧𝐠)")
st.markdown("<hr>", unsafe_allow_html=True)

# Instruções na barra lateral
st.sidebar.image("psicologia.jpg", width=200)
with st.sidebar.expander("Instruções"):
    st.markdown("""
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando K-means Clustering, PCA, e Rede Neural. Siga as instruções abaixo:

    **Passos:**
    1. Faça o upload de até 10 imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para segmentação de cores.
    3. Escolha se deseja aplicar PCA ou usar Rede Neural.
    4. Clique no botão "Executar" para processar as imagens.

    **Inovações:**
    - Estatísticas avançadas, incluindo margem de erro.
    - Interface interativa personalizável.

    **Pontos Positivos:**
    - Resultados visuais claros e estatísticas relevantes.

    **Limitações:**
    - Tempo de processamento varia com o tamanho da imagem.
    """)

# Upload das imagens pelo usuário (aceitando de 1 a 10 imagens)
uploaded_files = st.sidebar.file_uploader("Escolha de 1 a 10 imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Selecionar o número de clusters
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)

# Checkbox para aplicar PCA
apply_pca = st.sidebar.checkbox("Aplicar PCA para redução de dimensionalidade", value=True)

# Checkbox para usar Rede Neural
use_nn = st.sidebar.checkbox("Usar Rede Neural para classificação de cores", value=False)

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

            # Converter cores para valores inteiros
            colors = colors.astype(int)

            # Calcular margem de erro e outras estatísticas
            margin_of_error = []
            std_devs = []
            conf_intervals = []
            for i in range(num_clusters):
                cluster_pixels = pixels[labels == i]
                std_dev = np.std(cluster_pixels, axis=0)
                margin_err = std_dev / np.sqrt(cluster_pixels.shape[0]) * 1.96  # 95% CI
                conf_int = stats.norm.interval(0.95, loc=colors[i], scale=std_dev)
                std_devs.append(std_dev)
                margin_of_error.append(margin_err)
                conf_intervals.append(conf_int)

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

            # Exibir as cores dominantes, suas porcentagens e margem de erro
            st.write("Cores dominantes e estatísticas relevantes:")
            for i, (color, percentage, moe, std, ci) in enumerate(zip(colors, percentages, margin_of_error, std_devs, conf_intervals)):
                st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")
                st.write(f"Margem de Erro (95% CI): ±{moe}, Desvio Padrão: {std}")
                st.write(f"Intervalo de Confiança (95%): {ci}")
                st.write("---")
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
