import joblib
import streamlit as st
model = joblib.load('../regression.joblib')

size = st.number_input('Size')
nb_rooms = st.number_input('Number of rooms')
garden = st.number_input('Garden')

print(model.predict([[size, nb_rooms, garden]]))

value = model.predict([[size, nb_rooms, garden]])[0]
value = round(value, 2)
st.write(f"The price of the house is {value}$")
