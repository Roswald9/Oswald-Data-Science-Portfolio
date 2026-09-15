import streamlit as st

st.title("Hello, Streamlit!")
st.write("This is my first Streamlit app.")

if st.button("Click me!"):
    st.write("🎉 You clicked the button! Nice work! 🚀")
else:
    st.write("Click the button to see what happens...")

color = st.color_picker("Pick a color", "#00f900")
st.write(f"You picked: {color}")

import pandas as pd

st.subheader("Exploring Our Dataset")

df = pd.read_csv("data/sample_data.csv")