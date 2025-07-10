### TRABALHO DE REDES NEURAIS
# POR CASSIANO DE SENA, EDUARDO WEBER E RODRIGO CARRARD


para rodar o codigo, instale essas bibliotecas:

se pip não funcionar, vc provavelmente não tem python instalado OU não está pronto para ser usado no ambiente/IDE

pip install pandas

pip install numpy

pip install torch

pip install scikit-learn

# PARÂMETROS:
# "age";"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"
# exemplo: 30;"unemployed";"married";"primary";"no";1787;"no";"no";"cellular";19;"oct";79;1;-1;0;"unknown";"no"

# INFORMAÇÕES IMPORTANTES:
# Backpropagation - algoritmo que ajusta os pesos da rede neural propagando o erro da saída para as camadas anteriores, 
#                   calculando gradientes e atualizando os parâmetros para minimizar a função de perda.
# Pipeline        - sequência de etapas (pré‑processamento, criação de dataset, treino e avaliação) organizadas de forma 
#                   modular, garantindo reprodutibilidade e facilitando a manutenção e a experimentação.
#
# pandas    : biblioteca para leitura, manipulação e análise de dados em tabelas (DataFrames).
# numpy     : biblioteca para cálculo numérico eficiente em arrays multidimensionais.
# torch     : framework de deep learning que fornece tensores, autograd e construção de redes neurais em Python.
# sklearn   : coleção de algoritmos e ferramentas de machine learning, usada aqui para codificação (OneHotEncoder), 
#             normalização (StandardScaler) e cálculo de métricas.
#
# Batches   : pequenos segmentos do dataset (aqui de tamanho 32) usados em cada iteração de treino. 
#             Eles reduzem o ruído das atualizações de gradiente e controlam o uso de memória.
# Época     : uma passagem completa por todos os batches do conjunto de treino (por exemplo, ~142 batches para 4521 exemplos).
#             Cada época refina os pesos da rede em vários passos de backpropagation.
#
# Device Agnostic   - o código detecta GPU (CUDA) ou CPU e roda na melhor opção disponível.
# 
# DataLoader        - classe do PyTorch que faz o batching, shuffle e paralelismo no carregamento dos dados.
# 
# Otimizador Adam   - algoritmo de otimização que adapta a taxa de aprendizado para cada parâmetro, acelerando a convergência.
#
# Weight Decay      - forma de L2‑regularização que penaliza pesos grandes, ajudando a prevenir overfitting.
# 
# Função de Perda   - BCELoss (Binary Cross‑Entropy) mede o erro de classificação binária entre saída (probabilidade) e rótulo.
# 
# Dropout           - camada que “desliga” aleatoriamente uma fração de neurônios durante o treino, reduzindo co‑adaptação e overfitting.
# 
# Batch Evaluation  - na etapa de avaliação, os dados também são processados em batches (ex.: 1024) para economizar memória.
# 
# Logging Detalhado - prints por batch e por época mostram progressão de loss e tempo, ajudando a monitorar problemas de convergência.

"""
flowchart LR
  A[Leitura CSV] --> B[Pré-processamento]
  B --> C[TensorDataset/DataLoader]
  C --> D[Treino (épocas & batches)]
  D --> E[Modelo final]
  E --> F[Avaliação & Métricas]"""

"""
[Input] → Linear(32) → ReLU → Dropout →
         Linear(16) → ReLU → Dropout →
         Linear(1) → Sigmoid → [Saída binária]
"""

## Outros pontos importantes:
#   - Não usamos `y` nas features de entrada.
#     `y` é o valor alvo que queremos **prever**, e deve ficar separado das variáveis de entrada.
#
#   - Não usamos `duration` no treinamento.
#     Esse campo só é conhecido **após** a ligação e causaria *data leakage*,
#     inflando métricas e gerando previsões irreais em produção.
