import streamlit as st
import os
import joblib


this_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(this_dir, 'hobbies_model.pkl'))

with st.sidebar:
    st.title('Hobbies Prediction')
    st.write('This app predicts hobbies based on age and gender. It depends ond small ficticious data built')
    st.write('To show how we can use sklearn and streamlit to build a simple app')
st.title('Hobbies Prediction')
st.subheader('Enter Age and Gender and We will predict your hobby')

age = st.number_input('Enter Age', min_value=1, max_value=100)
gender = st.radio('Select Gender:', options=['Male', 'Female'])

if st.button('Predict'):
    input = [1 if gender == 'Male' else 0, age]
    prediction = model.predict([input])
    st.write(prediction)
