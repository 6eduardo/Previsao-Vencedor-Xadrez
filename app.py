import streamlit as st
import pickle

# Carregar modelo e encoders
with open("modelo_chess.pkl", "rb") as f:
    model, le_winner, le_victory, le_opening = pickle.load(f)

st.title("Previsão do Vencedor de Xadrez")
st.markdown("Preencha os dados da partida abaixo:")

# Entradas
rated = st.selectbox("Rated (0 = Não, 1 = Sim)", [0, 1])
turns = st.number_input("Número de jogadas", min_value=1, value=40)

# Usar as classes originais do encoder para seleção
victory_status_text = st.selectbox("Status da vitória", le_victory.classes_)
white_rating = st.number_input("Rating do jogador branco", min_value=100, value=150)
black_rating = st.number_input("Rating do jogador preto", min_value=100, value=1500)

# Nome original da abertura
opening_name_text = st.selectbox("Nome da abertura", le_opening.classes_)
opening_ply = st.number_input("Número de jogadas na abertura", min_value=1, value=5)

# Converter textos para os mesmos valores codificados
victory_status = le_victory.transform([victory_status_text])[0]
opening_name = le_opening.transform([opening_name_text])[0]

# Previsão
if st.button("Prever Vencedor"):
    entrada = [[rated, turns, victory_status, white_rating, black_rating, opening_name, opening_ply]]
    resultado = model.predict(entrada)[0]
    vencedor = le_winner.inverse_transform([resultado])[0].upper()
    st.success(f"Previsão: **{vencedor}**")
