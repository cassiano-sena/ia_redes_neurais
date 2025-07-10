# UNIVALI - Inteligência Artificial - Redes Neurais
# Cassiano de Sena Crispim, Eduardo da Rocha Weber e Rodrigo Rey de Oliveira Carrard
# 08/07/2025 - Rede Neural MLP com Backpropagation.


""" 
# PROPOSTA: 

    Utilizando o dataset bank realizar as seguintes tarefas:

    - Treinar uma rede neural com o conjunto de treinamento "bank.csv" visando prever se o cliente ai fazer a aplicação.

    - Validar o treinamento com o dataset completo "bank-full.csv"

    - Calcular as estatísticas: Acurácia, Precisão, Revocação e F1 score

    - Lembrando que vocês estão sendo contratados para fazer a previsão, focando em otimizar a campanha de marqueting. Assim quanto melhor for o acerto, mais o cliente economiza e fica feliz."""


"""
# Acurácia       - proporção de previsões corretas.
# Precisão       - de todas as previsões “positivas”, quantas eram verdadeiras.
# Revocação      - de todos os positivos reais, quantos foram capturados.
# F1-Score       - média harmônica entre precisão e revocação.
"""

# O campo 'duration' não é usado pois representa o tempo da ligação, uma informação que só existe após a campanha.
# Usá-lo causaria vazamento de dados (data leakage), fazendo a IA aprender com base em algo que ela não teria no mundo real.

# A coluna 'job' (e outras categóricas) é convertida via One-Hot Encoding.
# Isso faz com que cada valor (ex: 'admin', 'technician') ocupe uma posição no vetor de entrada.
# Assim, cada profissão é representada por um neurônio de entrada diferente.


import os
import time
import pandas
import numpy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Esta é uma MLP (Multi-Layer Perceptron) com 3 camadas ocultas, ativação ReLU, e saída com função Sigmoid.
# O treinamento é feito usando o algoritmo de backpropagation com o otimizador Adam.
#
# Etapas do funcionamento:
# - Os dados são carregados com pandas a partir de arquivos CSV.
# - O pré-processamento transforma dados categóricos em vetores (one-hot encoding)
#   e normaliza os dados numéricos (standard scaler).
# - A função de preprocessamento também separa X (entradas) e y (saídas esperadas).
# - A rede neural MLP é definida usando a biblioteca PyTorch (classe nn.Module).
# - Um DataLoader organiza os dados em batches (lotes) para o treino iterativo.
# - Durante o treino, a rede processa os dados em múltiplas épocas, ajustando seus pesos
#   com base no erro (função de perda) calculado entre a previsão e a saída real.
# - Como o indicado na proposta, a avaliação final utiliza métricas como Acurácia, Precisão, Revocação e F1-score.
# - Os resultados são exibidos no terminal.

# ----------------------
# PRÉ-PROCESSAMENTO / PIPELINE
# ----------------------
def preprocess_data(dataframe, onehot_encoder=None, scaler=None, fit=False):
    dataframe = dataframe.drop("duration", axis=1) # duration nao eh utilizado
    X = dataframe.drop("y", axis=1) # y nao eh utilizado
    y = dataframe["y"].map({"yes": 1, "no": 0}).values.astype(numpy.float32)

    numerico   = X.select_dtypes(include="int64").columns.tolist()
    categorico = X.select_dtypes(include="object").columns.tolist()

    X_num = X[numerico].values
    X_cat = X[categorico].values

    if fit:
        scaler = StandardScaler() # valores numericos sao padronizados
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) # categorias viram vetores
        X_num = scaler.fit_transform(X_num)
        X_cat = onehot_encoder.fit_transform(X_cat)
    else:
        X_num = scaler.transform(X_num)
        X_cat = onehot_encoder.transform(X_cat)

    X_final = numpy.hstack([X_num, X_cat]).astype(numpy.float32)
    return X_final, y, onehot_encoder, scaler

# ----------------------
# MODELO: TRES CAMADAS 64-32-16 PRA 1 (SAÍDA BINÁRIA)
# ----------------------
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim): # input_dim garante um vetor variavel de entrada
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, 64),  # 1ra camada com 64 neuronios
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), # 2da camada com 32 neuronios
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16), # 3ra camada com 16 neuronio
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1), # camada de saida com 1 neuronio
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.camadas(x)

# ----------------------
# TREINAMENTO
# ----------------------
def treinar_modelo(model, dataloader, criterio, otimizador, device, epocas=20):
    model.train()
    total_loss_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_update_time = 0.0

    for epoca in range(1, epocas+1):
        start = time.time()
        soma_loss = 0.0
        batches = len(dataloader)

        epoch_loss_time = 0.0
        epoch_forward = 0.0
        epoch_backward = 0.0
        epoch_update = 0.0

        print(f"\n=== Época {epoca}/{epocas} ===")
        for i, (batch_x, batch_y) in enumerate(dataloader, 1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            t0 = time.time()
            saida = model(batch_x)
            t1 = time.time()
            epoch_forward += (t1 - t0)

            t2 = time.time()
            perda = criterio(saida, batch_y)
            t3 = time.time()
            epoch_loss_time += (t3 - t2)

            otimizador.zero_grad()
            t4 = time.time()
            perda.backward()
            t5 = time.time()
            epoch_backward += (t5 - t4)

            t6 = time.time()
            otimizador.step()
            t7 = time.time()
            epoch_update += (t7 - t6)

            soma_loss += perda.item()
            if i % max(1, batches//10) == 0 or i == batches:
                pct = i / batches * 100
                print(f"  Batch {i}/{batches} ({pct:5.1f}%) — loss: {perda.item():.4f}")

        dur = time.time() - start
        loss_medio = soma_loss / batches

        total_forward_time += epoch_forward
        total_loss_time    += epoch_loss_time
        total_backward_time+= epoch_backward
        total_update_time  += epoch_update

        print(f"→ Época {epoca} concluída em {dur:.1f}s — Loss médio: {loss_medio:.4f}")
        print(f"   Forward: {epoch_forward:.3f}s | Loss calc: {epoch_loss_time:.3f}s | "
              f"Backward: {epoch_backward:.3f}s | Update: {epoch_update:.3f}s")

    print("\n— Tempos cumulativos durante o treinamento —")
    print(f"Total forward pass : {total_forward_time:.3f}s ({total_forward_time/epocas:.3f}s/época)")
    print(f"Total loss calc    : {total_loss_time:.3f}s ({total_loss_time/epocas:.3f}s/época)")
    print(f"Total backward pass: {total_backward_time:.3f}s ({total_backward_time/epocas:.3f}s/época)")
    print(f"Total update step  : {total_update_time:.3f}s ({total_update_time/epocas:.3f}s/época)")

# ----------------------
# AVALIAÇÃO
# ----------------------
def avaliar_modelo(model, X, y, device, batch_size=1024, threshold=0.5):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            out = model(batch).cpu().numpy()
            preds.append(out)
    preds = numpy.vstack(preds)
    pred_bin = (preds > threshold).astype(int)

    # Matriz de confusão (TN, FP, FN, TP)
    cm = confusion_matrix(y, pred_bin)
    print("\nMatriz de Confusão:")
    print(cm)

    # Falsos negativos: y=1, pred=0
    falsos_negativos = [i for i in range(len(y)) if y[i] == 1 and pred_bin[i] == 0]
    print(f"\nTotal de falsos negativos: {len(falsos_negativos)}")
    if len(falsos_negativos) > 0:
        print(f"Exemplos de falsos negativos (máx. 10): {falsos_negativos[:10]}")

    # Falsos positivos: y=0, pred=1
    falsos_positivos = [i for i in range(len(y)) if y[i] == 0 and pred_bin[i] == 1]
    print(f"\nTotal de falsos positivos: {len(falsos_positivos)}")
    if len(falsos_positivos) > 0:
        print(f"Exemplos de falsos positivos (máx. 10): {falsos_positivos[:10]}")

    ac  = accuracy_score(y, pred_bin)
    pr  = precision_score(y, pred_bin, zero_division=0)
    re  = recall_score(y, pred_bin)
    f1s = f1_score(y, pred_bin)
    return ac, pr, re, f1s


# ----------------------
# EXECUÇÃO PRINCIPAL
# ----------------------
def main():
    # epocas diz quantas vezes o modelo vai ler o arquivo de treino inteiro
    # learning rate define o tamanho da atualização dos pesos (padrão Adam = 1e-3)
    # weight decay aplica penalidade a pesos grandes, reduzindo overfitting (padrão = 1e-4)
    # threshold é o valor de corte da saída da rede (entre 0 e 1) usado para decidir se a previsão é "sim" (1) ou "não" (0).
    #           por padrão, usa-se 0.5: se a saída da rede for maior que 0.5, o resultado é considerado "sim".
    #           ajustar o threshold pode melhorar o balanceamento entre precisão e recall, especialmente em datasets desbalanceados.

    aplicar_smote = True
    valor_epocas = 1000
    valor_learning_rate = 5e-4
    valor_weight_decay = 1e-4
    valor_threshold = 0.4

    start_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)
    print("Usando o dispositivo:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))


    print("\n\n\nCarregando dados e pré-processando…\n\n\n")
    df_train = pandas.read_csv("dataset_bank/bank.csv", sep=";")
    df_test  = pandas.read_csv("dataset_bank/bank-full.csv", sep=";")

    X_train, y_train, onehot, scaler = preprocess_data(df_train, fit=True)

    if aplicar_smote:
        print("Aplicando SMOTE para balancear as classes do conjunto de treino…")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        valores, contagem = numpy.unique(y_train, return_counts=True)
        print("Distribuição após SMOTE:", dict(zip(valores, contagem)))

    X_test,  y_test,  _, _ = preprocess_data(df_test, onehot, scaler, fit=False)

    dataset    = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model     = MultiLayerPerceptron(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()

    optimizer= torch.optim.Adam(model.parameters(), lr=valor_learning_rate, weight_decay=valor_weight_decay) 
    
    print("\n" + "─"*100)
    print("Iniciando treinamento")
    print("─"*100)
    start_train = time.time()
    treinar_modelo(model, dataloader, criterion, optimizer, device, epocas=valor_epocas)
    train_time = time.time() - start_train
    print("\n" + "─"*100)
    print("Avaliação")
    print("─"*100)
    start_eval = time.time()
    ac, pr, re, f1s = avaliar_modelo(model, X_test, y_test, device, threshold=valor_threshold)
    eval_time = time.time() - start_eval

    # --- Detectar quantidade de camadas ocultas e neurônios ---
    hidden_layers = [m for m in model.camadas if isinstance(m, nn.Linear)][:-1]  # ignorar camada final
    num_camadas_ocultas = len(hidden_layers)
    neuronios_por_camada = [layer.out_features for layer in hidden_layers]

    # --- Exibição das métricas ---
    print("┌" + "─"*46 + "┐")
    print("│ Métrica               │ Valor                │")
    print("├" + "─"*46 + "┤")
    print(f"│ Acurácia              │ {ac*100:6.2f}% ({ac:.4f})     │")
    print(f"│ Precisão              │ {pr*100:6.2f}% ({pr:.4f})     │")
    print(f"│ Revocação             │ {re*100:6.2f}% ({re:.4f})     │")
    print(f"│ F1-score              │ {f1s*100:6.2f}% ({f1s:.4f})     │")
    print("└" + "─"*46 + "┘")
    print(f"  Camadas ocultas       │ {num_camadas_ocultas}")
    print(f"  Neurônios por camada  │ {neuronios_por_camada}")
    print(f"  Número de épocas      │ {valor_epocas}")
    print(f"  Threshold usado       │ {valor_threshold}")
    print(f"  Learning Rate         │ {valor_learning_rate}")
    print(f"  Weight Decay          │ {valor_weight_decay}")
    print(f"  SMOTE aplicado        │ {'Sim' if aplicar_smote else 'Não'}")
    total_time = time.time() - start_total
    print(f"\n  Tempo total de treinamento: {train_time:.1f}s")
    print(f"  Tempo de avaliação: {eval_time:.3f}s")
    print(f"  Tempo total de execução: {total_time:.1f}s")
    print("─"*100)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()