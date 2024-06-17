# Identificação de Cores Dominantes em Pinturas Utilizando K-means Clustering

Este projeto utiliza o algoritmo de clustering K-means para identificar as cores dominantes em pinturas, facilitando a análise e interpretação dessas cores no contexto da arteterapia.

## Importância para a Arteterapia

A análise de cores em obras de arte pode fornecer insights valiosos sobre o estado emocional e psicológico dos participantes em sessões de arteterapia. As cores escolhidas pelos indivíduos podem refletir suas emoções, pensamentos e experiências internas. Este projeto busca automatizar a identificação dessas cores, permitindo que terapeutas e pesquisadores analisem os dados de maneira mais eficiente e precisa.

## Funcionalidades

- Carregar uma imagem de uma pintura
- Aplicar o algoritmo K-means clustering para identificar as cores dominantes
- Exibir as cores dominantes e suas porcentagens
- Plotar gráficos de barras e gráficos de pizza para visualização das cores dominantes

## Requisitos

Certifique-se de que você tem as seguintes dependências instaladas antes de executar o projeto:

```plaintext
streamlit==1.10.0
scikit-learn==0.24.2
numpy==1.21.0
matplotlib==3.4.2
opencv-python==4.5.2.52
