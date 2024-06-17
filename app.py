import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.title('Identificação de Cores Dominantes em Pinturas')

# Carregar a imagem a partir do upload do usuário
st.sidebar.image("logo.png", width=200)
with st.sidebar.expander("Instrução"):
    
    st.markdown("""
    O código do Agente Expert Geomaker é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

    **Inovações:**
    - Suporte a múltiplos modelos de linguagem: O código permite que o usuário escolha entre diferentes modelos de linguagem, como o LLaMA, para gerar respostas mais precisas e personalizadas.
    - Integração com a API Groq: A integração com a API Groq permite que o aplicativo utilize a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
    - Refinamento de respostas: O código permite que o usuário refine as respostas do modelo de linguagem, tornando-as mais precisas e relevantes para a consulta.
    - Avaliação com o RAG: A avaliação com o RAG (Rational Agent Generator) permite que o aplicativo avalie a qualidade e a precisão das respostas do modelo de linguagem.

    **Pontos positivos:**
    - Personalização: O aplicativo permite que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas de acordo com suas necessidades.
    - Precisão: A integração com a API Groq e o refinamento de respostas garantem que as respostas sejam precisas e relevantes para a consulta.
    - Flexibilidade: O código é flexível o suficiente para permitir que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas.

    **Limitações:**
    - Dificuldade de uso: O aplicativo pode ser difícil de usar para os usuários que não têm experiência com modelos de linguagem ou API.
    - Limitações de token: O código tem limitações em relação ao número de tokens que podem ser processados pelo modelo de linguagem.
    - Necessidade de treinamento adicional: O modelo de linguagem pode precisar de treinamento adicional para lidar com consultas mais complexas ou específicas.

    **Importância de ter colocado instruções em chinês:**
    A linguagem chinesa tem uma densidade de informação mais alta do que muitas outras línguas, o que significa que os modelos de linguagem precisam processar menos tokens para entender o contexto e gerar respostas precisas. Isso torna a linguagem chinesa mais apropriada para a utilização de modelos de linguagem com baixa quantidade de tokens. Portanto, ter colocado instruções em chinês no código é um recurso importante para garantir que o aplicativo possa lidar com consultas em chinês de forma eficaz.

    Em resumo, o código é uma aplicação inovadora que combina modelos de linguagem com a API Groq para proporcionar respostas precisas e personalizadas. No entanto, é importante considerar as limitações do aplicativo e trabalhar para melhorá-lo ainda mais.
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
    
st.sidebar.image("psicologia.jpg", width=80)
st.sidebar.write("""
Projeto Arteterapia 
- Professores: Marcelo Claro.

Graduanda: Nadielle Darc Batista Dias
Whatsapp: (88)981587145

Instagram: [https://www.instagram.com/_psicologias/](https://www.instagram.com/_psicologias/)
    """)
