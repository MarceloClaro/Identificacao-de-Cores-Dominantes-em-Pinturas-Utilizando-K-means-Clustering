import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats
import pandas as pd

# Título e descrição
st.markdown("<h1 style='text-align: center;'>Identificação de Cores Dominantes em Pinturas</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("𝐂𝐨𝐧𝐡𝐞𝐜̧𝐚 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞𝐨𝐫𝐢𝐚𝐬, 𝐝𝐨𝐦𝐢𝐧𝐞 𝐭𝐨𝐝𝐚𝐬 𝐚𝐬 𝐭𝐞́𝐜𝐧𝐢𝐜𝐚𝐬, 𝐦𝐚𝐬 𝐚𝐨 𝐭𝐨𝐜𝐚𝐫 𝐮𝐦𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚, 𝐬𝐞𝐣𝐚 𝐚𝐩𝐞𝐧𝐚𝐬 𝐨𝐮𝐭𝐫𝐚 𝐚𝐥𝐦𝐚 𝐡𝐮𝐦𝐚𝐧𝐚 (𝐂.𝐆. 𝐉𝐮𝐧𝐠)")
st.markdown("<hr>", unsafe_allow_html=True)


# Instruções na barra lateral
st.sidebar.image("psicologia.jpg", width=200)
with st.sidebar.expander("Instruções"):
    st.markdown("""
    **Passos:**
    1. **Upload das Imagens**: Utilize o botão "Browse files" para fazer o upload de até 10 imagens no formato JPG, JPEG ou PNG. Essas imagens serão analisadas para identificar as cores dominantes.
    
    2. **Seleção de Clusters**: Use o controle deslizante para escolher o número de clusters para a segmentação de cores. Esse número representa quantas cores distintas você deseja identificar em cada imagem. O valor padrão é 5, mas você pode selecionar qualquer valor entre 1 e 10.

    3. **Redução de Dimensionalidade com PCA**: Se a opção de PCA (Análise de Componentes Principais) estiver ativada, a dimensionalidade dos dados de cor será reduzida antes da segmentação. Isso pode acelerar o processo, mas pode também sacrificar a precisão das cores. Recomendado apenas se estiver processando imagens muito grandes ou complexas.

    4. **Processamento das Imagens**: Após configurar os parâmetros, clique no botão "Executar" para iniciar a análise das imagens. O algoritmo de K-Means será aplicado para identificar e exibir as cores dominantes, juntamente com suas respectivas porcentagens.

    **Importante:** A ativação da opção PCA é opcional e deve ser usada com cautela. Se a precisão na identificação das cores for essencial para sua análise, é recomendável não utilizar essa opção.

    **Impacto do PCA:**
    - **Precisão das Cores:** O PCA pode alterar as cores ao simplificar os dados, o que pode reduzir a precisão na identificação das cores originais. Isso é particularmente relevante em imagens onde cada detalhe cromático é crucial.
    - **Simplificação e Redução de Ruído:** Em imagens complexas, o PCA pode ajudar a destacar padrões principais e reduzir ruídos desnecessários. No entanto, isso pode ocorrer à custa de uma menor fidelidade de cor.
    - **Tempo de Processamento:** O PCA pode reduzir o tempo de processamento ao simplificar os dados, tornando a análise mais eficiente em cenários com recursos computacionais limitados ou muitas imagens para processar.
    - **Recomendações:** Use o PCA se estiver lidando com imagens muito complexas e precisar otimizar o tempo de processamento. Evite-o se a precisão das cores é fundamental para a sua interpretação psicológica das imagens.

    **Exemplos Práticos de PCA em Imagens:**
    1. **Redução de Ruído em Imagens**: PCA pode ser utilizado para reduzir ruídos cromáticos em imagens detalhadas, destacando cores predominantes e eliminando variações sutis que não são importantes.
    2. **Compressão de Imagens para Análise Rápida**: Utilizar PCA para comprimir informações de cor em imagens de alta resolução, permitindo uma análise mais rápida, embora com potencial perda de nuances.
    3. **Destaque de Padrões Cromáticos em Imagens Artísticas**: PCA pode ajudar a identificar padrões cromáticos dominantes em pinturas abstratas, ressaltando paletas de cores subjacentes.
    4. **Simplificação de Cores em Imagens Médicas**: Em imagens médicas, PCA pode agrupar tons semelhantes, facilitando a segmentação e destacando áreas de interesse, mas com possível perda de detalhes sutis.

    **Tipos de Imagens para Análise**

    1. **Imagens Simples (Com Poucas Cores Distintas)**
    - **Exemplo:** Uma pintura com grandes áreas de cores uniformes, como um céu azul claro com uma área verde de grama.
    - **Impacto na Análise:** 
      - **PCA:** Pode não ser necessário, pois as cores são já bem distintas e o PCA pode simplificar excessivamente, perdendo nuances importantes.
      - **Clusterização:** Um número menor de clusters (2-3) pode ser suficiente para capturar as cores dominantes.

    2. **Imagens Complexas (Com Muitas Variações de Cor)**
    - **Exemplo:** Uma pintura abstrata cheia de variações de cor e textura.
    - **Impacto na Análise:**
      - **PCA:** Pode ser útil para reduzir o ruído e destacar padrões de cor predominantes, mas cuidado para não perder detalhes sutis que podem ser psicologicamente significativos.
      - **Clusterização:** Um número maior de clusters (5-10) pode ser necessário para capturar as nuances da imagem. PCA pode ajudar a acelerar o processamento, mas a precisão pode ser levemente sacrificada.

    3. **Imagens de Alta Resolução**
    - **Exemplo:** Fotos detalhadas de uma tela grande, capturando pinceladas e variações finas de cor.
    - **Impacto na Análise:**
      - **PCA:** Altamente recomendável se você precisa processar várias imagens grandes rapidamente. Isso vai comprimir os dados de cor, permitindo uma análise mais eficiente.
      - **Clusterização:** A análise pode capturar tanto detalhes macro (cores gerais) quanto micro (detalhes das pinceladas). Porém, cuidado com a perda de precisão em cores devido ao PCA.

    **Análise de Imagens de Arte Terapia**

    Quando você analisa imagens de sessões de arte terapia, a interpretação das cores pode fornecer insights sobre o estado emocional ou psicológico do paciente. Aqui está como você pode proceder:

    1. **Escolha do Tipo de Análise**
    - **Sem PCA:** Se o objetivo é capturar cada nuance emocional expressa através das cores, não use PCA. Isso preservará cada detalhe de cor, importante para uma interpretação psicológica precisa.
    - **Com PCA:** Use PCA se as imagens forem extremamente complexas e a quantidade de dados for muito grande, dificultando o processamento. Lembre-se de que isso pode simplificar demais as cores, mas ajudará a identificar padrões gerais.

    2. **Número de Clusters**
    - **Imagens Simples:** Use menos clusters (2-4) para capturar os elementos principais.
    - **Imagens Complexas:** Aumente o número de clusters (5-10) para capturar uma gama mais ampla de emoções e expressões.

    3. **Interpretação Psicológica das Cores**
    - **Aplicação:** Cada cor identificada pode ser relacionada a emoções ou estados mentais específicos. Por exemplo, cores como vermelho podem indicar intensidade emocional ou conflito, enquanto azul pode representar calma e introspecção.
    - **Contexto Terapêutico:** O contexto em que a pintura foi feita deve ser considerado ao interpretar os resultados. As cores e padrões podem refletir o processo terapêutico e as mudanças emocionais ao longo do tempo.

    **Recomendações Específicas para Arte Terapia**
    - **Evite PCA em Análises Críticas:** Para interpretações onde cada detalhe importa, como na análise emocional das pinturas, é melhor evitar PCA.
    - **Documente o Contexto:** A análise deve ser contextualizada com informações sobre a sessão terapêutica, o estado emocional do paciente, e o objetivo da atividade artística.
    - **Use Clusterização Adequada:** Ajuste o número de clusters com base na complexidade da imagem para capturar tanto os padrões gerais quanto as nuances importantes.

    **Dicas:**
    - Para imagens com muitas variações de cores, aumente o número de clusters para capturar melhor os detalhes.
    - Revise as cores dominantes e suas interpretações psicológicas para insights adicionais sobre a composição da imagem.
    """)

# Função para normalizar as cores para o intervalo [0, 1]
def normalize_color(color):
    color = np.clip(color, 0, 255) / 255
    return tuple(color)

# Função para calcular a distância euclidiana
def euclidean_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

# Função para encontrar a cor mais próxima e sua interpretação psicológica
def interpret_color_psychology(color):
    r, g, b = color
    colors_db = [
        {
            'color': (1, 0, 0),
            'name': 'Vermelho',
            'interpretation': (
                'Representa paixão, energia, e agressividade. '
                'No contexto dos arquétipos junguianos, o Vermelho está associado ao arquétipo do Guerreiro, '
                'simbolizando a força vital, a luta pela sobrevivência e a capacidade de agir. '
                'Este arquétipo está relacionado com a energia primal e a determinação, sendo um indicador '
                'de que o indivíduo pode estar enfrentando desafios ou conflitos internos. '
                'Na psicologia das cores, o vermelho pode evocar emoções intensas, como amor ou raiva, e '
                'é uma cor que muitas vezes chama à ação. Estudos indicam que a exposição à cor vermelha '
                'pode influenciar a fisiologia, como o aumento da oxigenação no cérebro e nos músculos, o que '
                'reforça sua associação com energia e ativação emocional.'
                '\n\nReferências: '
                '[Weinzirl et al., 2009](https://typeset.io/papers/color-therapy-changes-blood-oxygenation-in-the-brain-and-3ys19rlbtq?utm_source=chatgpt), '
                '[Cribier, 2011](https://typeset.io/papers/the-red-face-art-history-and-medical-representations-1j3gk638g7?utm_source=chatgpt).'                
            )
        },
        {
            'color': (0, 0, 1),
            'name': 'Azul',
            'interpretation': (
                'Simboliza calma, confiança e harmonia. '
                'Nos arquétipos junguianos, o Azul está associado ao arquétipo do Sábio, '
                'representando sabedoria, introspecção e espiritualidade. '
                'Este arquétipo pode refletir a busca por conhecimento e compreensão, '
                'indicando um período de introspecção e paz interior. '
                'Na psicologia das cores, o azul é amplamente reconhecido por seus efeitos calmantes e é frequentemente utilizado '
                'para promover tranquilidade e reduzir o estresse. Estudos mostram que a cor azul pode ter um efeito sedativo, '
                'ajudando a diminuir a ansiedade e até mesmo a dor. Além disso, a obra de artistas como Picasso, em seu "Período Azul", '
                'destaca como essa cor pode ser utilizada para expressar tristeza e sofrimento, refletindo emoções mais profundas através da arte.'
                '\n\nReferências: '
                '[Azeemi & Raza, 2005](https://doi.org/10.1093/ecam/neh137), '
                '[Gharib, 2020](https://www.scirp.org/journal/paperinformation.aspx?paperid=102013).'
                
            )
        },
        {
            'color': (1, 1, 0),
            'name': 'Amarelo',
            'interpretation': (
                'Ligado à criatividade, intelecto e alegria. O Amarelo está associado ao arquétipo do Herói na teoria junguiana, '
                'representando a busca por iluminação, crescimento e o desenvolvimento do self. Este arquétipo reflete o desejo de '
                'superar obstáculos e alcançar novos níveis de consciência. Na psicologia das cores, o amarelo é frequentemente '
                'relacionado à alegria e ao otimismo, mas também pode indicar ansiedade ou medo do fracasso. Pesquisas sugerem que '
                'o amarelo pode influenciar positivamente o humor e a energia, promovendo sentimentos de felicidade e motivação. '
                'Contudo, sua intensidade pode provocar cansaço visual e até frustração em alguns contextos, especialmente quando usado em excesso.'
                '\n\nReferências: '
                '[Artitude, 2023](https://artitude.ca/incorporating-colour-into-art-therapy), '
                '[Dunn-Edwards, 2015](https://www.dunnedwards.com/colors/specs-spaces/color-psychology-how-the-color-yellow-can-create-an-optimistic-space).'
                
            )
        },
        {
            'color': (0, 1, 0),
            'name': 'Verde',
            'interpretation': (
                'Associado à natureza, crescimento e estabilidade. '
                'No contexto dos arquétipos junguianos, o Verde está ligado ao arquétipo da Mãe Terra, '
                'simbolizando o ciclo de vida, a nutrição e o renascimento. '
                'Este arquétipo reflete uma conexão com a natureza e o desejo de estabilidade e crescimento. '
                'Na psicologia das cores, o verde é visto como calmante e equilibrante, muitas vezes usado '
                'para representar renovação e harmonia.'
                '\n\nReferências: '
                '[Bostan, 2022](https://typeset.io/papers/jungian-approach-to-cinema-archetypal-analysis-of-turning-1wvwee1k?utm_source=chatgpt).'
            )
        },
        {
            'color': (0, 0, 0),
            'name': 'Preto',
            'interpretation': (
                'Representa poder, mistério, e morte. '
                'O Preto está associado ao arquétipo da Sombra na teoria junguiana, '
                'simbolizando os aspectos reprimidos ou desconhecidos do self. '
                'Este arquétipo é crucial no processo de individuação, onde o confronto com a sombra leva '
                'ao autoconhecimento e à integração do self. '
                'Na psicologia das cores, o preto pode evocar sentimentos de luto, mistério, ou poder, '
                'e é frequentemente associado ao desconhecido ou ao inconsciente.'
                '\n\nReferências: '
                '[Suinn, 1966](https://typeset.io/papers/jungian-personality-typology-and-color-dreaming-499qe6u2wp?utm_source=chatgpt).'
            )
        },
        {
            'color': (1, 1, 1),
            'name': 'Branco',
            'interpretation': (
                'Simboliza pureza, inocência, e renovação. '
                'O Branco está ligado ao arquétipo do Inocente, representando a pureza primordial e a busca '
                'pela verdade. Este arquétipo pode indicar um novo começo ou o desejo de simplicidade e clareza. '
                'Na psicologia das cores, o branco é frequentemente associado à limpeza, virtude, e um estado de '
                'paz espiritual.'
                '\n\nReferências: '
                '[Petric, 2023](https://typeset.io/papers/psychological-archetypes-1zi1yvvz?utm_source=chatgpt).'
            )
        },
        # Adicione as outras cores aqui seguindo o mesmo padrão.
        {
            'color': (0.5, 0.5, 0.5),
            'name': 'Cinza',
            'interpretation': (
                'Simboliza neutralidade, sabedoria e maturidade. '
                'O Cinza está associado ao arquétipo do Sábio na teoria junguiana, '
                'representando uma perspectiva equilibrada e ponderada sobre a vida. '
                'Este arquétipo reflete a busca por conhecimento e a capacidade de ver as coisas como realmente são, '
                'sem o viés emocional. '
                'Na psicologia das cores, o cinza é muitas vezes visto como uma cor de compromisso, '
                'posicionada entre o branco e o preto, e pode simbolizar a transição e a ambiguidade. '
                'O cinza também pode sugerir uma forma de proteção, ocultando emoções para manter a paz e a neutralidade.'
                '\n\nReferências: '
                '[Kreitler & Kreitler, 1972](https://typeset.io/papers/personality-traits-and-meanings-of-colors-as-expressed-in-9xw2ljzo73?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.5, 0, 0.5),
            'name': 'Roxo',
            'interpretation': (
                'Ligado à espiritualidade, transformação e mistério. '
                'O Roxo está associado ao arquétipo do Mago, que simboliza a transformação, o poder pessoal e a busca por sabedoria oculta. '
                'Este arquétipo sugere uma conexão com o inconsciente coletivo e o desejo de transcender as limitações da realidade comum. '
                'Na psicologia das cores, o roxo é frequentemente associado à realeza, mistério e espiritualidade, '
                'e é usado para evocar sentimentos de reverência e introspecção. '
                'O roxo pode ser uma cor de mudança profunda, indicando um momento de transformação ou de passagem entre diferentes estados de ser.'
                '\n\nReferências: '
                '[Chae & Ho, 2005](https://typeset.io/papers/a-study-on-the-effect-of-color-preference-on-personality-and-1ltjg0j1kj?utm_source=chatgpt).'
            )
        },
        {
            'color': (1, 0.5, 0),
            'name': 'Laranja',
            'interpretation': (
                'Representa energia, criatividade, e entusiasmo. '
                'O Laranja está ligado ao arquétipo do Amante, que simboliza a paixão, o prazer e a conexão com os outros. '
                'Este arquétipo reflete um desejo de vivacidade e de experiências sensoriais intensas. '
                'Na psicologia das cores, o laranja é visto como uma cor que estimula a criatividade e a comunicação, '
                'sendo uma cor que pode elevar o humor e promover a sociabilidade. '
                'O laranja é uma cor calorosa e encorajadora, que pode impulsionar a motivação e a determinação.'
                '\n\nReferências: '
                '[Heller, 2009](https://typeset.io/papers/the-psychology-of-color-effects-and-symbolism-1x52kfxx2?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.5, 0.75, 0.5),
            'name': 'Verde-claro',
            'interpretation': (
                'Associado à tranquilidade, frescor e harmonia. '
                'O Verde-claro está associado ao arquétipo do Curador, representando o desejo de cura e equilíbrio. '
                'Este arquétipo reflete uma conexão profunda com a natureza e um desejo de restaurar a harmonia tanto interna quanto externa. '
                'Na psicologia das cores, o verde-claro é frequentemente utilizado para criar um ambiente relaxante e refrescante, '
                'sendo uma cor que promove a serenidade e o bem-estar. '
                'O verde-claro pode indicar um período de recuperação e crescimento, incentivando a renovação e a paz.'
                '\n\nReferências: '
                '[Dumbar, 2004](https://typeset.io/papers/color-psychology-and-its-effect-on-behavior-and-mood-8f32d3x0x?utm_source=chatgpt).'
            )
        },
        # Adicione as outras cores aqui seguindo o mesmo padrão.
        {
            'color': (0.4, 0.7, 0.6),
            'name': 'Turquesa',
            'interpretation': (
                'Representa equilíbrio emocional e tranquilidade. '
                'Turquesa está associada ao arquétipo do Curador, simbolizando a cura emocional e a comunicação clara. '
                'Este arquétipo reflete a busca por harmonia e a capacidade de trazer clareza aos pensamentos e sentimentos. '
                'Na psicologia das cores, o turquesa é frequentemente utilizado para criar uma sensação de calma e frescor, '
                'incentivando a cura e a renovação emocional. '
                'É uma cor que promove o equilíbrio entre o intelecto e as emoções, sendo ideal para ambientes que exigem clareza e tranquilidade.'
                '\n\nReferências: '
                '[Birren, 1950](https://typeset.io/papers/the-symbolism-of-colors-9f73hx9qr?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.7, 0.2, 0.2),
            'name': 'Carmesim',
            'interpretation': (
                'Representa paixão, intensidade e força. '
                'Carmesim está associado ao arquétipo do Guerreiro, que simboliza força, coragem e determinação. '
                'Este arquétipo reflete a vontade de enfrentar desafios e de lutar por aquilo que se acredita. '
                'Na psicologia das cores, o carmesim é uma cor poderosa que pode estimular a paixão e a ação. '
                'É frequentemente associada a emoções intensas, como amor e raiva, e pode ser usada para aumentar a motivação e a energia em um ambiente.'
                '\n\nReferências: '
                '[Azeemi, 2005](https://typeset.io/papers/color-therapy-and-color-psychology-2ffx7g39g?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.6, 0.5, 0.8),
            'name': 'Violeta',
            'interpretation': (
                'Associado à intuição, inovação, e misticismo. '
                'Violeta está ligado ao arquétipo do Mago, simbolizando transformação, sabedoria e intuição. '
                'Este arquétipo representa a busca por conhecimento oculto e a capacidade de inovar e transformar a realidade. '
                'Na psicologia das cores, o violeta é frequentemente utilizado para evocar sentimentos de misticismo e espiritualidade, '
                'sendo uma cor que estimula a criatividade e a introspecção. '
                'É uma cor que pode facilitar a meditação e a conexão com o eu interior.'
                '\n\nReferências: '
                '[Chae & Ho, 2005](https://typeset.io/papers/a-study-on-the-effect-of-color-preference-on-personality-and-1ltjg0j1kj?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.7, 0.7, 0.7),
            'name': 'Prata',
            'interpretation': (
                'Conectado à pureza, precisão, e integridade. '
                'Prata está associado ao arquétipo do Justo, que simboliza a busca pela verdade, integridade e equilíbrio. '
                'Este arquétipo reflete o desejo de justiça e de viver de acordo com altos padrões morais e éticos. '
                'Na psicologia das cores, o prata é visto como uma cor que simboliza sofisticação e modernidade, '
                'além de estar associado à clareza de pensamento e à objetividade. '
                'É uma cor que pode trazer uma sensação de calma e de resolução em ambientes de tomada de decisão.'
                '\n\nReferências: '
                '[Heller, 2009](https://typeset.io/papers/the-psychology-of-color-effects-and-symbolism-1x52kfxx2?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.8, 0.4, 0),
            'name': 'Âmbar',
            'interpretation': (
                'Simboliza calor, segurança e aconchego. '
                'Âmbar está ligado ao arquétipo do Protetor, representando a segurança, proteção e o cuidado. '
                'Este arquétipo reflete o desejo de criar um ambiente seguro e acolhedor para si mesmo e para os outros. '
                'Na psicologia das cores, o âmbar é uma cor calorosa que pode estimular sentimentos de segurança e conforto, '
                'sendo ideal para espaços que precisam de uma atmosfera acolhedora e reconfortante. '
                'É uma cor que promove a estabilidade emocional e a conexão com o lar.'
                '\n\nReferências: '
                '[Birren, 1950](https://typeset.io/papers/the-symbolism-of-colors-9f73hx9qr?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.2, 0.8, 0.2),
            'name': 'Verde-oliva',
            'interpretation': (
                'Representa paz, diplomacia e harmonia. '
                'Verde-oliva está associado ao arquétipo do Pacificador, simbolizando a diplomacia, a paz e a resolução de conflitos. '
                'Este arquétipo reflete a capacidade de mediar e resolver situações difíceis com calma e equilíbrio. '
                'Na psicologia das cores, o verde-oliva é frequentemente utilizado para criar uma sensação de estabilidade e equilíbrio, '
                'sendo uma cor que promove a paz interior e a harmonia em situações de estresse. '
                'É uma cor ideal para ambientes de negociação e de cooperação.'
                '\n\nReferências: '
                '[Azeemi, 2005](https://typeset.io/papers/color-therapy-and-color-psychology-2ffx7g39g?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.4, 0.2, 0.2),
            'name': 'Marrom',
            'interpretation': (
                'Simboliza estabilidade, confiabilidade e segurança. '
                'Marrom está associado ao arquétipo do Provedor, que simboliza a estabilidade, a segurança e a confiabilidade. '
                'Este arquétipo reflete a capacidade de prover e de criar uma base sólida para os outros. '
                'Na psicologia das cores, o marrom é visto como uma cor que traz uma sensação de solidez e de terra, '
                'sendo frequentemente utilizado para criar ambientes que transmitem conforto e segurança. '
                'É uma cor que sugere estabilidade e praticidade, ideal para espaços que exigem uma base firme.'
                '\n\nReferências: '
                '[Heller, 2009](https://typeset.io/papers/the-psychology-of-color-effects-and-symbolism-1x52kfxx2?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.5, 0.4, 0.4),
            'name': 'Bege',
            'interpretation': (
                'Representa simplicidade, confiabilidade e tradição. '
                'Bege está associado ao arquétipo do Tradicionalista, que simboliza a simplicidade, a confiabilidade e a valorização das tradições. '
                'Este arquétipo reflete a importância das raízes e da continuidade em um mundo em constante mudança. '
                'Na psicologia das cores, o bege é frequentemente utilizado para criar uma sensação de calma e de atemporalidade, '
                'sendo uma cor que promove a simplicidade e a confiabilidade em ambientes que exigem estabilidade. '
                'É uma cor que evoca uma sensação de conforto e de familiaridade, ideal para espaços que valorizam a tradição.'
                '\n\nReferências: '
                '[Kreitler & Kreitler, 1972](https://typeset.io/papers/personality-traits-and-meanings-of-colors-as-expressed-in-9xw2ljzo73?utm_source=chatgpt).'
            )
        },
        # Adicione as outras cores aqui seguindo o mesmo padrão.
        {
            'color': (1, 0.4, 0.7),
            'name': 'Rosa',
            'interpretation': (
                'Representa carinho, afeto e vulnerabilidade. '
                'Rosa está associado ao arquétipo do Amante, que simboliza o amor, o afeto e a ternura. '
                'Este arquétipo reflete a capacidade de expressar emoções de maneira suave e de cuidar dos outros com carinho. '
                'Na psicologia das cores, o rosa é frequentemente utilizado para evocar sentimentos de compaixão e de calor emocional, '
                'sendo uma cor que promove o afeto e a vulnerabilidade em relações interpessoais. '
                'É uma cor ideal para criar ambientes que exigem uma atmosfera acolhedora e gentil.'
                '\n\nReferências: '
                '[Heller, 2009](https://typeset.io/papers/the-psychology-of-color-effects-and-symbolism-1x52kfxx2?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.6, 0.4, 0.2),
            'name': 'Sépia',
            'interpretation': (
                'Evoca nostalgia e antiguidade. '
                'Sépia está ligado ao arquétipo do Sábio, que simboliza a busca por conhecimento, experiência e profundidade. '
                'Este arquétipo reflete a valorização da memória e da sabedoria acumulada ao longo do tempo. '
                'Na psicologia das cores, o sépia é frequentemente utilizado para criar uma sensação de nostalgia e de conexão com o passado, '
                'sendo uma cor que pode evocar memórias e trazer à tona sentimentos de tradição e de história. '
                'É uma cor ideal para ambientes que valorizam a história e a continuidade.'
                '\n\nReferências: '
                '[Azeemi, 2005](https://typeset.io/papers/color-therapy-and-color-psychology-2ffx7g39g?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.4, 0.2, 0.6),
            'name': 'Lavanda',
            'interpretation': (
                'Representa serenidade, graça e elegância. '
                'Lavanda está associada ao arquétipo da Donzela, que simboliza a inocência, a pureza e a delicadeza. '
                'Este arquétipo reflete a busca por beleza, harmonia e a expressão de sentimentos suaves e elegantes. '
                'Na psicologia das cores, a lavanda é frequentemente utilizada para criar uma atmosfera de paz e tranquilidade, '
                'sendo uma cor que promove a calma e a introspecção. '
                'É ideal para espaços que necessitam de serenidade e uma estética graciosa.'
                '\n\nReferências: '
                '[Chae & Ho, 2005](https://typeset.io/papers/a-study-on-the-effect-of-color-preference-on-personality-and-1ltjg0j1kj?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.3, 0.3, 0.7),
            'name': 'Índigo',
            'interpretation': (
                'Associado a pensamentos profundos e espiritualidade. '
                'Índigo está ligado ao arquétipo do Místico, que simboliza a busca pela verdade interior e pela conexão espiritual. '
                'Este arquétipo reflete o desejo de explorar os mistérios do universo e de se conectar com o espiritual. '
                'Na psicologia das cores, o índigo é frequentemente utilizado para estimular a intuição e a percepção espiritual, '
                'sendo uma cor que promove a introspecção e a meditação. '
                'É ideal para ambientes voltados ao estudo, à reflexão profunda e à busca pelo conhecimento espiritual.'
                '\n\nReferências: '
                '[Azeemi, 2005](https://typeset.io/papers/color-therapy-and-color-psychology-2ffx7g39g?utm_source=chatgpt).'
            )
        },
        {
            'color': (0.3, 0.6, 0.3),
            'name': 'Verde-musgo',
            'interpretation': (
                'Representa resiliência, endurance e equilíbrio. '
                'Verde-musgo está associado ao arquétipo do Guardião, que simboliza a proteção, a preservação e a resiliência. '
                'Este arquétipo reflete a capacidade de resistir às adversidades e de manter o equilíbrio em meio aos desafios. '
                'Na psicologia das cores, o verde-musgo é frequentemente utilizado para criar uma sensação de estabilidade e de conexão com a natureza, '
                'sendo uma cor que promove a força interior e a capacidade de adaptação. '
                'É uma cor ideal para ambientes que necessitam de uma base sólida e um foco na resiliência.'
                '\n\nReferências: '
                '[Heller, 2009](https://typeset.io/papers/the-psychology-of-color-effects-and-symbolism-1x52kfxx2?utm_source=chatgpt).'
            )
        }        
    ]
    
    closest_color = min(colors_db, key=lambda c: euclidean_distance((r, g, b), c['color']))
    return closest_color  # Retorna o dicionário inteiro



    
    closest_color = min(colors_db, key=lambda c: euclidean_distance((r, g, b), c['color']))
    return closest_color  # Retorna o dicionário inteiro

# Configuração do streamlit
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
        all_results = []  # Lista para armazenar os resultados de todas as imagens

        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_small = cv2.resize(image, (100, 100))
            pixels = image_small.reshape(-1, 3)

            if apply_pca:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                pixels = pca.fit_transform(pixels)
                st.write("PCA aplicada para redução de dimensionalidade.")

            kmeans = kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_

            counts = np.bincount(labels)
            percentages = counts / len(labels)

            statistics = calculate_statistics(pixels, labels, colors)
            
            # Se PCA foi aplicado, as cores precisam ser transformadas de volta
            if apply_pca:
                colors_rgb = pca.inverse_transform(colors)
            else:
                colors_rgb = colors

            colors_rgb = np.clip(np.round(colors_rgb), 0, 255).astype(int)
            colors_normalized = [normalize_color(color) for color in colors_rgb]

            st.image(image, caption='Imagem Analisada', use_column_width=True)

            dominant_colors = []
            interpretations = []
            for color, percentage in zip(colors_normalized, percentages):
                dominant_colors.append((color, percentage))
                closest_color_info = interpret_color_psychology(color)
                interpretations.append(closest_color_info)

            # Salvar os resultados em uma lista de dicionários para cada cor dominante
            for i, (color, percentage) in enumerate(dominant_colors):
                color_info = interpretations[i]
                r, g, b = [int(c*255) for c in color]
                result = {
                    "Imagem": uploaded_file.name,
                    "Cor": f"RGB({r}, {g}, {b})",
                    "Nome da Cor": color_info['name'],
                    "Porcentagem": f"{percentage:.2%}",
                    "Interpretação Psicológica": color_info['interpretation']
                }
                all_results.append(result)

            # Visualização das cores dominantes - Gráfico de Barras com porcentagens
            fig, ax = plt.subplots(1, 1, figsize=(8, 2))
            bar_width = 0.9
            for i, (color, percentage) in enumerate(dominant_colors):
                ax.bar(i, percentage, color=color, width=bar_width)
            ax.set_xticks(range(len(dominant_colors)))
            ax.set_xticklabels([f'{percentage:.1%}' for color, percentage in dominant_colors])
            ax.set_yticks([])
            plt.title("Cores Dominantes")
            st.pyplot(fig)

            # Gráfico de pizza das cores dominantes
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages],
                                              colors=colors_normalized,
                                              autopct='%1.1f%%', startangle=140, textprops={'color':"w"})
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(centre_circle)
            plt.title("Distribuição das Cores Dominantes")
            st.pyplot(fig)

            # Exibir cores dominantes e suas interpretações psicológicas
            st.write("**Cores dominantes e interpretações psicológicas:**")
            for i, (color, percentage) in enumerate(dominant_colors):
                color_info = interpretations[i]
                r, g, b = [int(c*255) for c in color]
                st.write(f"**Cor {i+1}:** {color_info['name']} (RGB: {r}, {g}, {b}) - {percentage:.2%}")
                st.write(f"**Interpretação Psicológica:** {color_info['interpretation']}")
                st.markdown("<hr>", unsafe_allow_html=True)

        # Converter a lista de resultados para um DataFrame e salvar como CSV
        results_df = pd.DataFrame(all_results)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar resultados como CSV",
            data=csv,
            file_name='cores_dominantes_resultados.csv',
            mime='text/csv',
        )

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
