import pandas as pd
import numpy as np
import streamlit as st
import joblib

model= joblib.load("galaxy_model.pkl")
st.title("Galaxy Classification App")
st.markdown("""
This app classifies galaxies into **Spiral** or **Elliptical**
based on structural features using a Machine Learning model.
""")
st.write("Enter features of galaxy:")

# Input fields
# K = st.number_input("K (Concentration)", value=10.0)
# C = st.number_input("C", value=0.3)
# A = st.number_input("A", value=0.7)
# S = st.number_input("S", value=0.8)
# G2 = st.number_input("G2", value=1.2)
# H = st.number_input("H", value=0.6)

K = st.slider("Concentration(K)", 0.0, 50.0, 10.0)
C = st.slider("Compactness(C)", 0.0, 1.0, 0.3)
A = st.slider("Asymmetry(A)", 0.0, 1.0, 0.7)
S = st.slider("Smoothness(S)", 0.0, 1.0, 0.8)
G2 = st.slider("Light distribution(G2)", 0.0, 3.0, 1.2)
H = st.slider("Structural complexity(H)", 0.0, 1.5, 0.6)


#Prediction
if st.button("Predict Galaxy Type"):

    features = np.array([[K, C, A, S, G2, H]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("Prediction: Spiral Galaxy")
    else:
        st.success("Prediction: Elliptical Galaxy")