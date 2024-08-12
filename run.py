import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configurações da página
st.set_page_config(
    page_title="Identificação de Cores Dominantes em Pinturas",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título e descrição
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Identificação de Cores Dominantes em Pinturas</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("<div style='text-align: center; font-style: italic;'>\"Conheça todas as teorias, domine todas as técnicas, mas ao tocar uma alma humana, seja apenas outra alma humana.\" - C.G. Jung</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Instruções na barra lateral
st.sidebar.image("psicologia.jpg", width=250)
with st.sidebar.expander("🛈 Instruções", expanded=True):
    st.markdown("""
    Este aplicativo permite identificar as cores dominantes em pinturas utilizando o algoritmo K-Means Clustering.

    **Passos para Utilização:**
    1. **Upload das Imagens:** Faça o upload de até **duas** imagens nos formatos JPG, JPEG ou PNG.
    2. **Configuração dos Clusters:** Selecione o número de clusters (entre 1 e 10) para determinar a quantidade de cores dominantes a serem identificadas.
    3. **Processamento:** Clique no botão **"Analisar Imagens"** para iniciar o processamento.
    
    **Resultados Fornecidos:**
    - Visualização da imagem original.
    - Barra com as cores dominantes identificadas.
    - Gráfico de pizza mostrando a porcentagem de cada cor dominante.
    - Detalhamento textual das cores em formato hexadecimal e suas respectivas porcentagens.

    **Observações:**
    - Imagens de alta resolução podem demandar maior tempo de processamento.
    - Para melhores resultados, utilize imagens com boa iluminação e contraste.
    """)
    
# Upload das imagens pelo usuário
uploaded_files = st.sidebar.file_uploader("📁 Faça o upload de até duas imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Você pode arrastar e soltar as imagens ou clicar para selecionar os arquivos.")

# Selecionar o número de clusters
num_clusters = st.sidebar.slider("🎨 Selecione o Número de Clusters", 1, 10, 5, help="Determina quantas cores dominantes serão identificadas em cada imagem.")

# Botão para executar a análise
if st.sidebar.button("🖼️ Analisar Imagens"):
    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### Análise da Imagem {idx+1}")
            try:
                # Ler a imagem usando PIL
                image = Image.open(uploaded_file)
                st.image(image, caption='Imagem Original', use_column_width=True)

                # Converter a imagem para RGB e redimensionar para acelerar o processamento
                image = image.convert('RGB')
                resized_image = image.resize((250, 250))
                img_array = np.array(resized_image)
                img_flat = img_array.reshape((-1, 3))

                # Aplicar K-Means
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(img_flat)
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_

                # Calcular a porcentagem de cada cor
                counts = np.bincount(labels)
                percentages = counts / len(labels)

                # Ordenar as cores por porcentagem decrescente
                sorted_idx = np.argsort(-percentages)
                colors = colors[sorted_idx]
                percentages = percentages[sorted_idx]

                # Exibir a barra de cores dominantes
                st.markdown("#### Cores Dominantes")
                fig, ax = plt.subplots(figsize=(12, 2))
                for i, (color, percentage) in enumerate(zip(colors, percentages)):
                    ax.barh(0, percentage, left=sum(percentages[:i]), color=tuple(color/255), edgecolor='white')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 1)
                st.pyplot(fig)

                # Exibir gráfico de pizza
                st.markdown("#### Distribuição das Cores Dominantes")
                fig1, ax1 = plt.subplots()
                hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])) for c in colors]
                ax1.pie(percentages, labels=hex_colors, autopct='%1.1f%%', colors=hex_colors, startangle=140, textprops={'color':"w"})
                centre_circle = plt.Circle((0,0),0.70,fc='black')
                fig1.gca().add_artist(centre_circle)
                ax1.axis('equal')  
                st.pyplot(fig1)

                # Detalhamento textual das cores
                st.markdown("#### Detalhes das Cores Dominantes")
                color_details = ""
                for i, (color, percentage) in enumerate(zip(colors, percentages)):
                    hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
                    color_details += f"- **Cor {i+1}:** `{hex_color}` | **Porcentagem:** {percentage*100:.2f}%\n"
                st.markdown(color_details)

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar a imagem: {e}")
    else:
        st.error("Por favor, faça o upload de pelo menos uma imagem.")

# Informações adicionais no rodapé
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("""
    **Projeto Arteterapia**
    - **Professores:** Marcelo Claro (Coorientador)
    - **Graduanda:** Nadielle Darc Batista Dias
    - **Contato:** [WhatsApp](https://wa.me/5588981587145)
    - **Instagram:** [Equipe de Psicologia 5º Semestre](https://www.instagram.com/_psicologias/)
    """)
