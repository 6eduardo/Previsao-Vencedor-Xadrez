from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import pickle

# Carregar o dataset
df = pd.read_csv("games.csv")

# Remover colunas desnecessárias
df.drop(columns=["id", "created_at", "last_move_at", "white_id", "black_id",
                 "increment_code", "moves", "opening_eco"], inplace=True)

# Remover valores nulos
df.dropna(inplace=True)

# Copiar dados
data = df.copy()

# Codificação
le_victory = LabelEncoder()
le_opening = LabelEncoder()
le_winner = LabelEncoder()

data["victory_status"] = le_victory.fit_transform(data["victory_status"])
data["opening_name"] = le_opening.fit_transform(data["opening_name"])
data["rated"] = data["rated"].astype(int)
data["winner"] = le_winner.fit_transform(data["winner"])

#separar classes
empate = data[data["winner"] == 1]
branco = data[data["winner"] == 2]
preto = data[data["winner"] == 0]

#esquilibrar
baixar_branco = resample(branco, replace=False, n_samples=950, random_state=42)
baixar_preto = resample(preto, replace=False, n_samples=950, random_state=42)

#unir
dados_balanciados = pd.concat([empate, baixar_branco, baixar_preto])

#embaralhar
dados_balanciados = dados_balanciados.sample(frac=1, random_state=42)

# Separar features e target
X = dados_balanciados[['rated', 'turns', 'victory_status', 'white_rating', 'black_rating', 'opening_name', 'opening_ply']]
y = dados_balanciados["winner"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

'''
# Treinar o modelo
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)
'''

model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)


# Avaliar modelo
y_pred = model.predict(X_test)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salvar modelo e label encoder
with open("modelo_chess.pkl", "wb") as f:
    pickle.dump((model, le_winner, le_victory, le_opening), f)

