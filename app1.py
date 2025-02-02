import streamlit as st
from sklearn.cluster import KMeans
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
    Este aplicativo permite identificar as cores dominantes em uma pintura utilizando o algoritmo K-means Clustering. Siga as instruções abaixo para usar o aplicativo:

    **Passos:**
    1. Faça o upload de duas imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Clique no botão "Executar" para processar as imagens.

    **Detalhes Técnicos:**
    - **Upload da Imagem:** O aplicativo aceita imagens nos formatos JPG, JPEG e PNG.
    - **Número de Clusters:** Selecione entre 1 e 10 clusters.
    - **Resultados:** Exibe uma barra com as cores dominantes e um gráfico de pizza com a distribuição percentual.
    """)

# Upload das imagens pelo usuário
uploaded_files = st.sidebar.file_uploader("Escolha duas imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Selecionar o número de clusters
num_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 5)

# Botão para executar a análise
if st.sidebar.button("Executar"):
    if len(uploaded_files) != 2:
        st.error("Por favor, faça o upload de exatamente duas imagens.")
    else:
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                # Ler a imagem do upload
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    st.error(f"Erro ao ler a imagem: {uploaded_file.name}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Redimensionar a imagem para acelerar o processamento
                image_small = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)

                # Converter a imagem para um array 2D (cada linha é um pixel com 3 canais RGB)
                pixels = image_small.reshape(-1, 3)

                # Aplicar K-means clustering para identificar as cores dominantes
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_

                # Calcular a porcentagem de cada cor
                counts = np.bincount(labels)
                percentages = counts / len(labels)

                # Converter cores para valores inteiros (para exibição)
                colors = colors.astype(int)

                # Ordenar as cores por ordem decrescente de percentual
                sorted_indices = np.argsort(-percentages)
                colors = colors[sorted_indices]
                percentages = percentages[sorted_indices]

                # Gerar as cores em formato hexadecimal para os gráficos
                hex_colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors]

                st.subheader(f"Análise da Imagem {idx}: {uploaded_file.name}")
                st.image(image, caption='Imagem Analisada', use_column_width=True)

                # Plotar as cores dominantes como uma barra customizada com números
                fig_bar, ax_bar = plt.subplots(figsize=(8, 2))
                ax_bar.axis('off')

                # Desenhar retângulos para cada cor e adicionar o texto centralizado com o percentual
                left = 0
                for i, (color, pct) in enumerate(zip(colors, percentages)):
                    width = pct  # largura proporcional ao percentual
                    # Adicionar retângulo
                    ax_bar.add_patch(plt.Rectangle((left, 0), width, 1, color=hex_colors[i]))
                    
                    # Calcular brilho da cor para escolher a cor do texto (fundo claro: texto preto, fundo escuro: texto branco)
                    r, g, b = color
                    brightness = r * 0.299 + g * 0.587 + b * 0.114
                    text_color = 'white' if brightness < 128 else 'black'
                    
                    # Adicionar o texto com o percentual no centro do retângulo
                    ax_bar.text(left + width/2, 0.5, f'{int(pct*100)}%', 
                                ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)
                    left += width

                ax_bar.set_xlim(0, 1)
                ax_bar.set_ylim(0, 1)
                st.pyplot(fig_bar)
                plt.close(fig_bar)

                # Plotar gráfico de pizza das cores dominantes com aprimoramento na visualização dos números
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax_pie.pie(
                    percentages,
                    labels=[f'{int(pct*100)}%' for pct in percentages],
                    colors=hex_colors,
                    autopct='%1.1f%%',
                    startangle=140,
                    textprops={'fontsize': 12, 'fontweight': 'bold'}
                )
                # Ajustar cores dos textos para melhor legibilidade
                for text in texts:
                    text.set_color('grey')
                for autotext in autotexts:
                    autotext.set_color('white')
                plt.title("Distribuição das Cores Dominantes")
                st.pyplot(fig_pie)
                plt.close(fig_pie)

                # Exibir as cores dominantes e suas porcentagens em formato texto
                st.write("Cores dominantes e suas porcentagens:")
                for color, percentage in zip(colors, percentages):
                    st.write(f"Cor: {color}, Porcentagem: {percentage:.2%}")

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar a imagem {uploaded_file.name}: {e}")

# Informações adicionais na barra lateral
st.sidebar.image("logo.png", width=80)
st.sidebar.write("""
Projeto Arteterapia 
- Professores: Marcelo Claro (Coorientador).

Graduanda: Nadielle Darc Batista Dias  
Whatsapp: (88)981587145

Instagram: [Equipe de Psicologia 5º Semestre](https://www.instagram.com/_psicologias/)
""")