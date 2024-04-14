import streamlit as st
import joblib
import numpy as np

# input = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2', 'carat',
#    'depth', 'table', 'x', 'y', 'z']
# carat : 0.2 - 3
# depth : 55 - 70
# table : 52 - 68
# x : 3 - 9
# y : 3 - 9
# z : 2 - 5
# Output: Price
model = joblib.load('./DiamondsPrices_model.pkl')


def predict_price(clarity, carat, depth, table, x, y, z):
    columns = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1',
               'VVS2', 'carat', 'depth', 'table', 'x', 'y', 'z']
    x_input = [0, 0, 0, 0, 0, 0, 0, 0, carat, depth, table, x, y, z]

    clarity_index = columns.index(clarity)
    x_input[clarity_index] = 1
    x_input = np.array(x_input).reshape(1, -1)

    price = model.predict(x_input)
    return price[0]


st.image("logo.png", width=64)
st.title('Diamonds Prices Prediction')
st.subheader(
    'Please read about the dataset that we used to build the model on the side menu.')


col1, col2, col3 = st.columns(3)


with col1:
    st.write('Diamond Clarity')
    clarity = st.selectbox(
        'Clarity', ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'])
with col2:
    st.write('Carat, Table and Depth')
    carat = st.number_input('Carat', min_value=0.2, max_value=3.0, value=0.2)
    depth = st.number_input('Depth', min_value=55, max_value=70, value=55)
    table = st.number_input('Table', min_value=52, max_value=68, value=52)

with col3:
    st.write('Diamond Dimensions')
    x = st.number_input('X', min_value=3, max_value=9, value=3)
    y = st.number_input('Y', min_value=3, max_value=9, value=3)
    z = st.number_input('Z', min_value=2, max_value=5, value=2)


if st.button('Predict'):
    price = predict_price(clarity, carat, depth, table, x, y, z)
    st.subheader('The price of the diamond is Around:')
    st.subheader(f'${price:.2f}')


with st.sidebar:
    st.image("logo.png", width=64)
    st.title('About Diamonds Prices Prediction App.')

    st.write("This App is built using the dataset from kaggle by: VITTORIO GIATTI.")
    dataset_url = 'https://www.kaggle.com/datasets/vittoriogiatti/diamondprices'
    st.markdown('Dataset [link](%s)' % dataset_url)

    st.subheader("About the DataSet")
    st.write('Each record is a random diamond 💎 💍 with its own characteristics. It should be easy to create a model to predict its price on the market basing on objective variables or try to cluster the observations avoiding the use of clarity, which is the "pureness" of the stone.')

    st.write("We tried from scikit-learn The following algorithms for regression:")
    st.write("1. Linear Regression: 0.8964 accuracy.")
    st.write("2. Lasso: 0.8961 accuracy.")
    st.write("3. DecisionTreeRegressor: 0.8649 accuracy.")

    st.write("We Also we tried from XGBoost:")
    st.write("1. XGBRegressor: 0.9301 accuracy.")

    st.write("So we saved the model of XGBRegressor algorithm in a pickle file and we are using it here.")
