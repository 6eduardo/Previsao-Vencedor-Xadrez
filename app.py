import streamlit as st
import pandas as pd
import pickle

# Carregar o modelo treinado
with open("modelo_chess.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Previsão do Vencedor de Xadrez")

st.markdown("Preencha os dados da partida abaixo:")

rated = st.selectbox("Rated (0 = Não, 1 = Sim)", [0, 1])
turns = st.number_input("Número de jogadas", min_value=1, value=40)
victory_status = st.selectbox("Status da vitória", [0, 1, 2, 3])
white_rating = st.number_input("Rating do jogador branco", min_value=100, value=1500)
black_rating = st.number_input("Rating do jogador preto", min_value=100, value=1500)
opening_name = st.number_input("Código da abertura (LabelEncoder)", min_value=0, value=100)
opening_ply = st.number_input("Número de jogadas na abertura", min_value=1, value=5)

# Previsão
if st.button("Prever Vencedor"):
    entrada = [[rated, turns, victory_status, white_rating, black_rating, opening_name, opening_ply]]
    resultado = model.predict(entrada)[0]

    if resultado == 0:
        vencedor = "Branco"
    elif resultado == 1:
        vencedor = "Empate"
    else:
        vencedor = "Preto"

    st.success(f"Previsão: **{vencedor}**")
