import streamlit as st
from langchainhelper import get_advise

st.title('Traveller Advisor')
st.subheader('Welcome to the Traveller Advisor!  Just give me details of the type of place you want to visit and I will advise you.')

details = st.text_input('Enter the type of place you want to visit')

if details:
    country, preps, attractions, avoid = get_advise(details)
    st.header(f'You should visit {country}')
    st.subheader('Make your self Prepared with:')
    st.write(preps)
    st.subheader('Top Attractions to visit:')
    st.write(attractions)
    st.subheader('Things to avoid:')
    st.write(avoid)
