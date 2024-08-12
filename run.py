import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    color = np.round(color).astype(int)
    if len(color) == 2:
        color = np.append(color, 0)  # Adiciona o valor do azul como 0 se a cor tiver apenas dois componentes
    elif len(color) == 1:
        color = np.append(color, [0, 0])  # Adiciona valores para verde e azul se a cor tiver apenas um componente
    return np.clip(color, 0, 255) / 255  # Normaliza as cores para o intervalo [0, 1]

def interpret_color_psychology(color):
    r, g, b = validate_color(color)

    # Interpretação das cores com base na psicologia das cores e arquétipos junguianos
    if r > 0.59 and g < 0.39 and b < 0.39:
        return ("Vermelho: Associado ao arquétipo do Guerreiro ou Herói, o vermelho representa paixão, "
                "energia, agressividade, mas também vitalidade e força. É a cor do amor intenso e do ódio, "
                "da coragem e da guerra. Para o psicólogo, observar o uso frequente do vermelho pode indicar "
                "conflitos internos, um estado de alerta constante ou uma expressão de poder e domínio. É relevante "
                "monitorar essa cor em pacientes que lidam com raiva ou impulsos agressivos. [Heller, 2000; Jung, 1959; "
                "Yoon & Chun, 2022 - https://typeset.io/papers/an-autobiographical-study-on-the-color-psychology-of-art-2wlwfd5v]")
    elif b > 0.59 and g < 0.39 and r < 0.39:
        return ("Azul: Relacionado ao arquétipo do Sábio ou Mentor, o azul simboliza calma, confiança, "
                "e harmonia. É a cor da sabedoria, do conhecimento profundo, e da introspecção, conectando-se "
                "com a serenidade e a eternidade. Para o psicólogo, a predominância do azul pode sugerir um estado "
                "de tranquilidade ou um desejo de introspecção. Também pode indicar repressão emocional em casos "
                "onde o azul é usado de forma excessiva. [Heller, 2000; Jung, 1959; McDonald, 2020 - "
                "https://typeset.io/papers/jungian-archetypes-a-step-towards-scientific-enquiry-14yljcno47]")
    elif r > 0.59 and g > 0.59 and b < 0.39:
        return ("Amarelo: Ligado ao arquétipo do Criador, o amarelo representa a criatividade, o intelecto e a "
                "alegria. É uma cor de otimismo, mas também pode sugerir ciúme e traição. Está associado ao ouro, "
                "à riqueza de ideias e ao brilho da inteligência. Em arteterapia, o amarelo pode ser um indicativo "
                "de busca por reconhecimento ou validação intelectual. Contudo, em excesso, pode apontar para inseguranças "
                "e ansiedade. [Heller, 2000; Jung, 1959; Potash, 2015 - https://typeset.io/papers/archetypal-aesthetics-viewing-art-through-states-of-586grld0zu]")
    elif g > 0.59 and r < 0.39 and b < 0.39:
        return ("Verde: Este é o arquétipo do Cuidador ou Mãe, que evoca a fertilidade, esperança e saúde. "
                "O verde é a cor da natureza, do crescimento e da estabilidade. Está relacionado ao conforto e à "
                "proteção, mas também à inveja e ao veneno. Para o psicólogo, o verde pode simbolizar a necessidade "
                "de segurança emocional ou uma conexão profunda com a natureza. No entanto, a escolha frequente desta cor "
                "pode também refletir ciúmes ou ressentimentos ocultos. [Heller, 2000; Jung, 1959; Yoon & Chun, 2022 - "
                "https://typeset.io/papers/an-autobiographical-study-on-the-color-psychology-of-art-2wlwfd5v]")
    elif r < 0.2 and g < 0.2 and b < 0.2:
        return ("Preto: Representando o arquétipo da Sombra, o preto é a cor do poder, da morte e do mistério. "
                "Embora também esteja associado à elegância e à sofisticação, é a cor que esconde, que absorve a luz, "
                "e pode simbolizar o desconhecido e o inconsciente. Em termos terapêuticos, o preto pode ser um indicativo "
                "de luto, depressão ou uma jornada em direção ao autoconhecimento. É crucial para o psicólogo monitorar "
                "o uso de preto, pois pode revelar medos profundos ou resistência ao tratamento. [Heller, 2000; Jung, 1959; "
                "McDonald, 2020 - https://typeset.io/papers/jungian-archetypes-a-step-towards-scientific-enquiry-14yljcno47]")
    elif r > 0.78 and g > 0.78 and b > 0.78:
        return ("Branco: Conectado ao arquétipo do Inocente ou Anjo, o branco simboliza a pureza, a inocência e a "
                "bondade. É a cor da luz, da paz, da renovação espiritual e da verdade absoluta. Também pode representar "
                "a esterilidade ou vazio emocional. O branco pode ser utilizado para indicar um desejo de renascimento ou "
                "purificação, mas em excesso pode sugerir negação emocional ou a busca de uma perfeição inatingível. [Heller, 2000; "
                "Jung, 1959; Potash, 2015 - https://typeset.io/papers/archetypal-aesthetics-viewing-art-through-states-of-586grld0zu]")
    elif r > 0.5 and g < 0.4 and b > 0.4:
        return ("Roxo: Associado ao arquétipo do Mago ou Alquimista, o roxo combina a sabedoria do azul com a paixão "
                "do vermelho. É a cor da transformação, da espiritualidade profunda, do mistério e da magia. Também está "
                "ligada ao poder e à realeza. Em arteterapia, o roxo pode ser visto como um sinal de busca por transformação "
                "ou crescimento espiritual. No entanto, pode também indicar uma luta interna entre o desejo por controle e "
                "a necessidade de libertação. [Heller, 2000; Jung, 1959; Yoon & Chun, 2022 - https://typeset.io/papers/an-autobiographical-study-on-the-color-psychology-of-art-2wlwfd5v]")
    elif r > 0.5 and g < 0.4 and b < 0.39:
        return ("Laranja: Ligado ao arquétipo do Explorador ou Artista, o laranja representa a energia criativa, a aventura "
                "e o entusiasmo. É uma cor de movimento, vitalidade e diversão, mas também pode ser percebida como "
                "extravagante ou excessiva. A utilização do laranja pode indicar um estado de excitação ou a busca por novas "
                "experiências. Psicologicamente, também pode apontar para um desejo de se destacar ou ser notado. [Heller, 2000; "
                "Jung, 1959; McDonald, 2020 - https://typeset.io/papers/jungian-archetypes-a-step-towards-scientific-enquiry-14yljcno47]")
    elif r < 0.5 and g > 0.5 and b < 0.5:
        return ("Cinza: Associado ao arquétipo do Oráculo ou Sábio, o cinza representa neutralidade, sabedoria e maturidade. "
                "É a cor da introspecção, do equilíbrio e da ponderação, mas também pode ser vista como sem vida ou sem emoção. "
                "Em arteterapia, o cinza pode refletir um estado de equilíbrio ou uma tentativa de se distanciar emocionalmente "
                "dos eventos da vida. Monitorar essa cor pode ser crucial para identificar sentimentos de apatia ou distanciamento. "
                "[Heller, 2000; Jung, 1959; Potash, 2015 - https://typeset.io/papers/archetypal-aesthetics-viewing-art-through-states-of-586grld0zu]")
    elif r > 0.6 and g < 0.4 and b > 0.4:
        return ("Rosa: Relacionado ao arquétipo do Amante, o rosa é a cor do carinho, do afeto e da ternura. "
                "Simboliza o amor romântico e a compaixão, mas também pode sugerir superficialidade ou imaturidade. "
                "O rosa pode ser uma expressão de vulnerabilidade ou um desejo de carinho e aceitação. Psicologicamente, é "
                "importante observar se o uso desta cor está associado a um comportamento infantilizado ou a uma busca por "
                "atenção e proteção. [Heller, 2000; Jung, 1959; Yoon & Chun, 2022 - https://typeset.io/papers/an-autobiographical-study-on-the-color-psychology-of-art-2wlwfd5v]")
    elif r > 0.4 and g < 0.4 and b > 0.6:
        return ("Violeta: Ligado ao arquétipo do Visionário, o violeta representa a espiritualidade elevada, a intuição e a inovação. "
                "É a cor do misticismo e da inspiração criativa, evocando o desconhecido e o transcendente. A presença do violeta "
                "pode indicar uma conexão com aspectos espirituais ou uma busca por compreensão além do físico. Pode também refletir "
                "uma personalidade introvertida e reflexiva, que valoriza o pensamento profundo. [Heller, 2000; Jung, 1959; Potash, 2015 - "
                "https://typeset.io/papers/archetypal-aesthetics-viewing-art-through-states-of-586grld0zu]")
    elif r > 0.7 and g > 0.5 and b > 0.2:
        return ("Dourado: Associado ao arquétipo do Rei ou Governante, o dourado simboliza poder, riqueza, e iluminação. "
                "É a cor do sucesso e da majestade, refletindo prestígio e grandeza, mas também pode ser associada à arrogância. "
                "Em contextos terapêuticos, o dourado pode simbolizar a busca por status ou a necessidade de controle sobre o ambiente. "
                "Psicologicamente, também pode indicar um desejo de reconhecimento e afirmação social. [Heller, 2000; Jung, 1959; "
                "McDonald, 2020 - https://typeset.io/papers/jungian-archetypes-a-step-towards-scientific-enquiry-14yljcno47]")
    elif r > 0.7 and g > 0.7 and b > 0.7:
        return ("Prata: Conectado ao arquétipo do Curador, o prata representa pureza, precisão, e integridade. "
                "É a cor da reflexão, da clareza e da feminilidade, relacionada à lua e às marés. Em arteterapia, o prata pode "
                "indicar um processo de cura emocional ou uma busca por clareza mental. Também pode simbolizar uma forte conexão "
                "com a intuição e os ciclos naturais. [Heller, 2000; Jung, 1959; Potash, 2015 - https://typeset.io/papers/archetypal-aesthetics-viewing-art-through-states-of-586grld0zu]")
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
