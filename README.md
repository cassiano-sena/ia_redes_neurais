# Trabalho de Redes Neurais
### Por Cassiano de Sena, Eduardo Weber e Rodrigo Carrard
###### UNIVALI 07/2025 - INTELIGÊNCIA ARTIFICIAL I - CIÊNCIA DA COMPUTAÇÃO

---

## Como executar o código

Instale as bibliotecas necessárias:

```bash
pip install pandas numpy torch scikit-learn
```

> Se `pip` não funcionar, verifique se o Python está corretamente instalado e disponível no terminal ou IDE.

---

## Sobre os dados

Os campos utilizados são:

```
"age", "job", "marital", "education", "default", "balance", "housing", "loan",
"contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
```

Exemplo de entrada:

```
30, "unemployed", "married", "primary", "no", 1787, "no", "no",
"cellular", 19, "oct", 79, 1, -1, 0, "unknown", "no"
```

Outros dados em main() podem ser editados para diferentes resultados:
```
def main():
    aplicar_smote = True
    valor_epocas = 300
    valor_learning_rate = 0.0005
    valor_weight_decay = 1e-4
    valor_threshold = 0.4
```
---

## Principais conceitos utilizados

- **SMOTE(opcional)**: técnica que gera novos exemplos sintéticos da classe minoritária para balancear os dados e melhorar o desempenho da rede em previsões menos frequentes.
- **Backpropagation**: algoritmo que ajusta os pesos da rede com base no erro da saída, retropropagando-o pelas camadas.
- **Pipeline**: sequência organizada de etapas (pré-processamento, treino e avaliação) que garante reprodutibilidade e modularidade.
- **MLP (Multi-Layer Perceptron)**: rede com camadas densas, funções de ativação e aprendizado supervisionado.

### Bibliotecas

- **pandas**: leitura e manipulação de tabelas.
- **numpy**: operações numéricas com arrays.
- **torch**: framework de machine learning com suporte a GPU.
- **scikit-learn**: ferramentas de codificação, normalização e métricas.

### Outros termos

- **Batch**: pequenos grupos de dados usados em cada passo do treino (aqui: 32 exemplos).
- **Época**: uma varredura completa por todos os batches.
- **Adam**: otimizador adaptativo que acelera o aprendizado.
- **Dropout**: técnica que desativa aleatoriamente neurônios para evitar overfitting.
- **BCELoss**: função de perda para classificação binária.
- **Weight Decay**: penalização de pesos altos para regularização.
- **Avaliação por batch**: usa divisões de tamanho fixo para reduzir uso de memória na avaliação.
- **Device Agnostic**: o código detecta se há GPU disponível (CUDA) e se adapta.
- **Learning Rate**: controla o tamanho do passo na atualização dos pesos da rede. Valores menores tornam o treino mais lento e estável; valores maiores tornam mais rápido, mas instável.
- **Threshold (limiar)**: valor de corte usado para converter a saída da rede (probabilidade) em 0 ou 1. Por padrão é 0.5. Ajustar esse valor afeta o equilíbrio entre **precisão** e **revocação**.

---

## Estrutura da rede neural

```text
Entrada
  ↓
Linear (64) → ReLU → Dropout
  ↓
Linear (32) → ReLU → Dropout
  ↓
Linear (16) → ReLU → Dropout
  ↓
Linear (1) → Sigmoid
  ↓
Saída (0 ou 1)
```

---

## Etapas do processo

```mermaid
flowchart LR
  A[Leitura dos dados] --> B[Pré-processamento]
  B --> C[Criação do Dataset]
  C --> D[Treino com épocas e batches]
  D --> E[Modelo treinado]
  E --> F[Avaliação com métricas]
```

---

## Observações

- A coluna `y` é a saída desejada, por isso não deve ser usada como entrada no modelo.
- A coluna `duration` é descartada porque só é conhecida após a chamada — seu uso causaria *vazamento de dados* (data leakage).
