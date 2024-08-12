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
    1. Faça o upload de duas imagens utilizando o botão "Browse files".
    2. Escolha o número de clusters para a segmentação de cores utilizando o controle deslizante.
    3. Se desejar, ative a opção de PCA para redução de dimensionalidade.
    4. Clique no botão "Executar" para processar as imagens.

    **PCA (Análise de Componentes Principais):**
    
    Como psicólogo, você sabe que as cores desempenham um papel crucial na percepção e podem influenciar o estado emocional de uma pessoa. No entanto, quando se trata de análise digital de imagens, as cores são representadas por valores numéricos em três dimensões: vermelho, verde e azul (RGB). Cada cor em uma imagem é uma combinação desses três componentes.

    O PCA (Principal Component Analysis), ou Análise de Componentes Principais, é uma técnica estatística que pode ser usada para simplificar esses dados. Ele faz isso reduzindo a quantidade de informação, mantendo ao mesmo tempo o máximo de variação possível nas cores. No entanto, há algo importante que você precisa entender:

    **1. Redução de Dimensionalidade:**
    - Imagine que cada cor seja um ponto em um espaço tridimensional (com eixos vermelho, verde e azul). Quando aplicamos o PCA, estamos essencialmente "achatando" esse espaço tridimensional em duas dimensões. Isso significa que estamos descartando uma parte da informação de cor original, o que pode alterar significativamente a percepção das cores.

    **2. Transformação das Cores:**
    - Após o PCA, as cores são reconfiguradas. Elas não correspondem mais diretamente aos componentes RGB originais, mas são novas combinações das cores, baseadas nos "componentes principais" que o PCA calcula. Como resultado, as cores dominantes identificadas após a aplicação do PCA podem parecer diferentes das cores originais na imagem.

    **3. Mudança de Percepção:**
    - Em termos práticos, isso significa que a cor "vermelha" que você vê após o PCA pode não ser mais o mesmo vermelho vibrante da imagem original; pode se tornar um tom mais apagado ou até mudar completamente para outra cor. Isso pode ser útil em algumas análises, como quando se deseja simplificar a imagem para identificar padrões de cor mais amplos, mas não é ideal quando se quer preservar a fidelidade das cores originais.

    **Como Decidir se Deve Usar o PCA:**
    - **Evite usar PCA se:** A precisão das cores for crucial para a análise. Por exemplo, se você estiver analisando a cor em um contexto terapêutico e a cor específica é essencial para o entendimento do estado emocional ou simbólico.
    - **Use PCA se:** Você quiser reduzir ruído ou simplificar a imagem para focar em padrões de cor gerais, sem se preocupar tanto com a exatidão das cores individuais.

    **Conclusão:**
    Se o seu objetivo é entender as cores exatamente como elas aparecem na imagem original, é melhor não ativar a opção de PCA. Caso contrário, se você deseja uma simplificação que possa destacar tendências gerais de cor, o PCA pode ser uma ferramenta valiosa.

    **Resultados:** O aplicativo exibirá uma barra com as cores dominantes e um gráfico de pizza mostrando a distribuição percentual de cada cor para cada imagem.

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
with st.sidebar.expander("Referências"):
    st.markdown("""
    ### Referências:

    - **Heller, Eva. "A Psicologia das Cores." Editorial Gustavo Gili, 2000.**
    - **Jung, Carl G. "The Archetypes and The Collective Unconscious." Princeton University Press, 1959.**
    - **Yoon, Seungjae, & Chun, Myungsoon. "An Autobiographical Study on the Color Psychology of Art." Journal of Art Therapy, 2022.** [Link](https://typeset.io/papers/an-autobiographical-study-on-the-color-psychology-of-art-2wlwfd5v)
    - **McDonald, Emily. "Jungian Archetypes: A Step Towards Scientific Enquiry." Journal of Analytical Psychology, 2020.** [Link](https://typeset.io/papers/jungian-archetypes-a-step-towards-scientific-enquiry-14yljcno47)
    - **Potash, Jordan. "Archetypal Aesthetics: Viewing Art Through States of Consciousness." Art Therapy Journal, 2015.** [Link](https://typeset.io/papers/archetypal-aesthetics-viewing-art-through-states-of-586grld0zu)
    """)

# Função para garantir que as cores estejam no formato adequado para o matplotlib
def validate_color(color):
    color = np.clip(np.round(color), 0, 255).astype(int)
    return color / 255  # Normaliza as cores para o intervalo [0, 1]

def interpret_color_psychology(color):
    r, g, b = validate_color(color)

    # Interpretação das cores com base na psicologia das cores e arquétipos
    if r > 0.85 and g < 0.15 and b < 0.15:
        return "Vermelho: Representa paixão, energia, e agressividade."
    elif b > 0.85 and g < 0.15 and r < 0.15:
        return "Azul: Simboliza calma, confiança e harmonia."
    elif r > 0.85 and g > 0.85 and b < 0.15:
        return "Amarelo: Ligado à criatividade, intelecto e alegria."
    elif g > 0.75 and r < 0.25 and b < 0.25:
        return "Verde: Associado à natureza, crescimento e estabilidade."
    elif r < 0.15 and g < 0.15 and b < 0.15:
        return "Preto: Representa poder, mistério, e morte."
    elif r > 0.95 and g > 0.95 and b > 0.95:
        return "Branco: Simboliza pureza, inocência, e renovação."
    elif r > 0.5 and g > 0.75 and b > 0.5:
        return "Verde-claro: Associado à tranquilidade, frescor e harmonia."
    elif r < 0.2 and g > 0.5 and b < 0.3:
        return "Verde-escuro: Simboliza maturidade, estabilidade e introspecção."
    elif r > 0.6 and g < 0.4 and b > 0.7:
        return "Roxo: Ligado à espiritualidade, transformação e mistério."
    elif r > 0.9 and g > 0.6 and b < 0.3:
        return "Laranja: Representa energia, criatividade, e entusiasmo."
    elif r > 0.4 and g > 0.4 and b > 0.4 and r < 0.6 and g < 0.6 and b < 0.6:
        return "Cinza: Simboliza neutralidade, sabedoria e maturidade."
    elif r > 0.9 and g < 0.7 and b < 0.7:
        return "Rosa: Representa carinho, afeto e vulnerabilidade."
    elif r > 0.6 and g < 0.5 and b > 0.8:
        return "Violeta: Associado à intuição, inovação, e misticismo."
    elif r > 0.85 and g > 0.75 and b < 0.4:
        return "Dourado: Simboliza poder, riqueza e iluminação."
    elif r > 0.7 and g > 0.7 and b > 0.7 and r < 0.9 and g < 0.9 and b < 0.9:
        return "Prata: Conectado à pureza, precisão, e integridade."
    elif r < 0.3 and g > 0.7 and b > 0.6:
        return "Turquesa: Representa equilíbrio emocional e tranquilidade."
    elif r < 0.1 and g < 0.1 and b < 0.1:
        return "Carvão: Representa seriedade, profundidade e introspecção."
    elif r > 0.75 and g > 0.85 and b < 0.4:
        return "Chartreuse: Associado à criatividade, inovação e inspiração."
    elif r > 0.95 and g > 0.95 and b > 0.85:
        return "Marfim: Simboliza elegância, pureza e sofisticação."
    elif r > 0.8 and g < 0.3 and b < 0.3:
        return "Carmesim: Representa paixão, intensidade e força."
    elif r < 0.4 and g > 0.5 and b > 0.8:
        return "Azul-cobalto: Simboliza clareza, precisão e foco."
    elif r < 0.25 and g < 0.35 and b > 0.55:
        return "Azul-marinho: Representa autoridade, confiança e estabilidade."
    else:
        return "Cor não identificada. Consulte manualmente."








st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes e Psicologia das Cores</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

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
                color = validate_color(color)  # Certifique-se de que a cor esteja no formato correto
                dominant_colors.append((color, percentage))
                interpretations.append(interpret_color_psychology(color))

            fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
            for sp in ax.spines.values():
                sp.set_visible(False)
            bar_width = 1
            index = np.arange(len(colors))
            ax.bar(index, [1] * len(colors), color=[validate_color(color) for color in colors], width=bar_width)
            ax.set_xticks(index)
            ax.set_xticklabels([f'Cor {i+1}' for i in range(num_clusters)])
            plt.title("Cores Dominantes")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages],
                                              colors=[validate_color(color) for color in colors],
                                              autopct='%1.1f%%', startangle=140)
            for text in texts:
                text.set_color('grey')
            for autotext in autotexts:
                autotext.set_color('white')
            plt.title("Distribuição das Cores Dominantes")
            st.pyplot(fig)

            st.write("Cores dominantes e interpretações psicológicas:")
            for i, (color, percentage) in enumerate(dominant_colors):
                st.write(f"**Cor {i+1}:** {color} - {percentage:.2%}")
                st.write(f"**Interpretação Psicológica:** {interpretations[i]}")

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
