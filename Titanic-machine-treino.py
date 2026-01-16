#%% Bibliotecas usadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# %% Funções de normalização e padronização
def normalizar(x): #x será uma lista de valores numéricos
  return (x - np.min(x))/(np.max(x) - np.min(x))

def padronizar(x):
  return (x -np.mean(x))/np.std(x)

# %% Carregamento dos dados 
df = pd.read_csv('/Users/eduardosantos/Downloads/Titanic-Dataset.csv')
df.head()
# %% Renomeando as Colunas 
df = df.rename(columns = {'PassengerId': 'ID', 'Survived': 'Sobreviveu', 'Pclass': 'Classe', 'Name': 'Nome', 'Sex': 'Sexo', 'Age': 'Idade', 'SibSp': 'Parentes', 'Parch': 'Dependentes', 'Ticket': 'Bilhete', 'Fare': 'Tarifa', 'Cabin': 'Cabine', 'Embarked': 'Embarque'})

# %% vizualizando as 5 primeiras linhas do dataset
df.head()
# %% Dropando colunas desnecessárias
df = df.drop(columns=['ID', 'Nome', 'Bilhete', 'Cabine'])
#%%
df.shape

# %% vendo dataset
df

# %% Verificando os tipos de dados
print(df.dtypes)
df.info()
# %% Verificando valores nulos
print(df.isnull().sum())

# %% For para a coluna para vizualizar os dados das colunas 
for coluna in df.columns:
    print(f"{coluna} valores únicos: {df[coluna].unique()}")
    print("------------------------------")
# %% Verificando valores nulos (NOVAMENTE)
df.isna().sum()

# %% DROPANDO VALORES NULOS 

df.dropna(inplace=True)

# %% Verificando valores nulos (APÓS DROP)
df.isna().sum()
# %%
df.shape
df.info()
df


#%%     Visualizando a distribuição da Idade
df
sb.pairplot(df[['Idade', 'Parentes', 'Dependentes', 'Tarifa', 'Sobreviveu']], hue='Sobreviveu')


# %% Função para verificar se os dados seguem uma distribuição normal
# # Verificar normalidade usando o teste de Shapiro-Wilk
from scipy.stats import shapiro
def verfificar_normalidade(dataframe, coluna):
    coluna_data = dataframe[coluna]
    # Teste do Shapiro-Wilk
    statistic, p_valor = shapiro(coluna_data)
    # Valor de significância
    alpha = 0.05
    # Verificando se a hipotese nula será rejeitada
    if p_valor > alpha:
        print(f"A coluna '{coluna}' parece seguir uma distribuição normal)")
        return True
    else:
        print(f"A coluna '{coluna}' não parece seguir uma distribuição normal)")
        return False

#%% For para verificar a normalidade de todas as colunas numéricas
for coluna in ['Idade', 'Parentes', 'Dependentes', 'Tarifa']:
    if verfificar_normalidade(df, coluna): #se for distribuição normal, padroniza
        df[coluna] = padronizar(df[coluna])

    else:
        df[coluna] = normalizar(df[coluna]) #se não for, não faz nada

# %% Aqui vamos usar o Dummies para colocarmos 0 e 1 na colunas categoricas 
df = pd.get_dummies(df, columns=['Sexo', 'Classe', 'Embarque'], drop_first=True)
df
# %%

def distancia_euclidiana(A, B):
    if len(A) != len(B):
        print("Erro de dimensões: os vetores devem ter o mesmo tamanho.")
        return 0
    total = 0
    for i in range(0, len(A)):
        total += (B[i] - A[i])**2
    return total**0.5

#%%  
pontoA = [6, 3, 200, 1795, -50]
pontoB = [12, -7, 2, 355, -9]

distancia_euclidiana(pontoA, pontoB)
# %%
df.head(1)
# %% target = df['Sobreviveu']
# %% Separando as variáveis de entrada e saída
# A variável de saída é 'Sobreviveu' e as variáveis de entrada são todas as outras colunas
# 'Sobreviveu' é a classe que queremos prever
# As demais colunas são as características (features) usadas para fazer a previsão
# A variável de saída é a coluna 'Sobreviveu' e as variáveis de entrada são todas as outras colunas
y = df['Sobreviveu']#classe (saída)
X = df.drop('Sobreviveu', axis=1) #entradas

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
#%% Regra para escolher o valor de k
X.shape[0]**0.5
# o Melhor k é 27 (raiz quadrada do número de amostras)

# %% # Criar o modelo kNN com 27 vizinhos
knn = KNeighborsClassifier(n_neighbors=27) # Usando k=27 com o teste acima 
# Treinar o modelo kNN
knn.fit(X_train, y_train)
#%% acurácia do modelo
y_pred = knn.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo kNN: {acuracia:.2f}")
#%%
X_test


# %%
print("n=",len(y))
print("k=", (len(y))**0.5)
# %%

#testando o k ideal
for k in range(3, 51, 2):
  knn = KNeighborsClassifier(n_neighbors=k)
  # Treinar o modelo kNN
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)

  # Avaliar a acurácia do modelo
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Acurácia para k={k}:", accuracy)

# %% Primeiro Exemplo de conjuto de teste

# Selecionar o primeiro exemplo do conjunto de teste
exemplo = X_test.iloc[[22]]
saida_real = y_test.iloc[22]

# Fazer a previsão com o modelo treinado
previsao = knn.predict(exemplo)

# Imprimir a previsão e a saída real
print("Previsão:", previsao[0])
print("Saída Real:", saida_real) 


# %% resultado final

# Fazer previsões para todo o conjunto de teste
previsoes = knn.predict(X_test)
# Criar um DataFrame para comparar as previsões com as saídas reais
comparacao = pd.DataFrame({'Previsão': previsoes, 'Saída Real': y_test})
# Mostrar o DataFrame de comparação
comparacao

comparacao.head()


# %% como eu faço para ver quantos sobreviveram ?
sobreviventes = comparacao['Saída Real'].sum()

# %%
print("Número de sobreviventes:", sobreviventes)
# %%
