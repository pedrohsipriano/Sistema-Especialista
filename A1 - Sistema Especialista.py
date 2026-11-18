# 1. O que é aprendizado de máquina?
# É quando um algoritmo ajusta um modelo a partir de dados, em vez de uma pessoa escrever as regras à mão.
# 2. O que caracteriza o aprendizado supervisionado?
# O modelo aprende com exemplos que já vêm com a resposta certa (o rótulo), como um gabarito, e usa os padrões aprendidos para prever casos novos. No seu sistema, cada peça de treino já vem marcada como aprovada ou defeituosa.
# 3. O que são atributos e rótulos?
# Atributos são as entradas que descrevem o caso: peso_kg, comprimento_mm e ruido_db. Rótulo é a resposta conhecida: qualidade (aprovada/defeituosa).
# 4. Como classificação difere de regressão?
# Classificação prevê uma categoria, como aprovada ou defeituosa. Regressão prevê um número, como quantos dB a peça vai fazer. O seu sistema faz classificação.
# 5. Por que separar treino e teste?
# Para medir o modelo em casos que ele nunca viu. Se você avaliar com os mesmos dados do treino, ele pode só ter “decorado” e parecer melhor do que é. No seu código, são 22 peças para treino e 8 para teste.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Dados SIMULADOS. Peça boa: ~1,20 kg, ~150 mm, até 45 dB.
dados = pd.DataFrame({
    # 15 peças boas
    "peso_kg":        [1.20, 1.22, 1.18, 1.25, 1.19, 1.21, 1.23, 1.17, 1.24, 1.20, 1.16, 1.26, 1.19, 1.22, 1.18,
    # 15 peças ruins: 5 leves, 5 compridas, 5 barulhentas
                       1.08, 1.10, 1.12, 1.06, 1.13, 1.21, 1.19, 1.22, 1.20, 1.18, 1.20, 1.23, 1.18, 1.21, 1.19],
    "comprimento_mm": [150, 149, 151, 148, 150, 149, 151, 148, 150, 149, 151, 150, 148, 149, 151,
                       150, 149, 151, 148, 150, 154, 155, 153, 156, 154, 150, 149, 151, 148, 150],
    "ruido_db":       [40, 38, 41, 39, 42, 37, 40, 43, 36, 41, 44, 39, 42, 38, 40,
                       40, 39, 41, 38, 42, 40, 38, 42, 41, 39, 49, 52, 47, 55, 48],
    "qualidade":      ["aprovada"] * 15 + ["defeituosa"] * 15,
})

X = dados[["peso_kg", "comprimento_mm", "ruido_db"]]   # atributos
y = dados["qualidade"]                                 # rótulo
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste)
print("Acurácia:", accuracy_score(y_teste, previsoes))
print(confusion_matrix(y_teste, previsoes, labels=["aprovada", "defeituosa"]))

# Base de conhecimento: (qualidade, defeito_repetido) -> (regra, ação)
regras = {
    ("defeituosa", False): ("R1", "Descartar a peça."),
    ("defeituosa", True):  ("R2", "Parar a linha e chamar o operador."),
    ("aprovada", True):    ("R3", "Montar e avisar o operador para revisar a máquina."),
    ("aprovada", False):   ("R4", "Enviar para a montagem."),
}

# Motor de inferência
def inspecionar(peso, comprimento, ruido, defeito_repetido):
    if peso is None or comprimento is None or ruido is None:      # R5
        return "R5", "Chamar o operador para inspeção manual."
    peca = pd.DataFrame([[peso, comprimento, ruido]], columns=X.columns)
    qualidade = modelo.predict(peca)[0]            # fase 1: ML prevê
    return regras[(qualidade, defeito_repetido)]   # fase 2: regra decide

print(inspecionar(1.20, 150, 40, True))   # positivo
print(inspecionar(1.10, 150, 40, True))    # negativo
print(inspecionar(1.00, 151, 20, True)) # informação ausente
print(inspecionar(1.19, 151, 35, True)) # informação ausente
print(inspecionar(1.19, 151, 40, True)) # informação ausente
print(inspecionar(1.19, 170, None, True)) # informação ausente
print(inspecionar(1.19, 151, 46, True)) # informação ausente
