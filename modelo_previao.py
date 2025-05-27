from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
le = LabelEncoder()
data["victory_status"] = le.fit_transform(data["victory_status"])
data["opening_name"] = le.fit_transform(data["opening_name"])
data["rated"] = data["rated"].astype(int)
data["winner"] = le.fit_transform(data["winner"])

# Separar features e target
X = data[['rated', 'turns', 'victory_status', 'white_rating', 'black_rating', 'opening_name', 'opening_ply']]
y = data["winner"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Prever com feature names
sample = X_test.iloc[[0]]
pred = model.predict(sample)
print("Predição:", pred)

# Salvar modelo
with open("modelo_chess.pkl", "wb") as f:
    pickle.dump(model, f)
