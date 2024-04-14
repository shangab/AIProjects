import streamlit as st
import joblib
import numpy as np

# input = ['clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
#    'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2', 'cut_Fair', 'cut_Good',
#    'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_D', 'color_E',
#    'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'carat', 'depth',
#    'table', 'x', 'y', 'z']
# carat : 0.20 -> 3.24
# depth : 43 - 80
# table : 50 - 80

# x : 3.73 - 9.54
# y : 3.68 - 9.46
# z : 1.07 - 5.98
# Output: Price
model = joblib.load('./DiamondsPrices_model.pkl')


def predict_price(clarity, cut, color, carat, depth, table, x, y, z):
    columns = ['clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
               'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2', 'cut_Fair', 'cut_Good',
               'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_D', 'color_E',
               'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'carat', 'depth',
               'table', 'x', 'y', 'z']
    x_input = np.zeros(len(columns))
    x_input = x_input.tolist()
    x_input[20] = carat
    x_input[21] = depth
    x_input[22] = table
    x_input[23] = x
    x_input[24] = y
    x_input[25] = z

    clarity_index = columns.index(clarity)
    x_input[clarity_index] = 1

    cut_index = columns.index(cut)
    x_input[cut_index] = 1

    color_index = columns.index(color)
    x_input[color_index] = 1

    x_input = np.array(x_input).reshape(1, -1)

    price = model.predict(x_input)
    return price[0]


st.image("logo.png", width=64)
st.title('Diamonds Prices Prediction')
st.subheader(
    'Please read about the dataset that we used to build the model on the side menu. Model on: XGBRegressor: 0.9812 accuracy.")')


col1, col2, col3 = st.columns(3)


with col1:
    st.write('Diamond Clarity')
    clarity = st.selectbox(
        'Clarity', ['clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
                    'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2'])
    cut = st.selectbox(
        'Cut', ['cut_Fair', 'cut_Good',
                'cut_Ideal', 'cut_Premium', 'cut_Very Good'])
    color = st.selectbox(
        'Color', ['color_D', 'color_E',
                  'color_F', 'color_G', 'color_H', 'color_I', 'color_J'])
with col2:
    st.write('Carat, Depth, & Table')
    carat = st.number_input('Carat', min_value=0.20,
                            max_value=3.50, value=1.20)
    depth = st.number_input('Depth', min_value=52.0,
                            max_value=70.0, value=55.0)
    table = st.number_input('Table', min_value=52.0,
                            max_value=70.0, value=52.0)

with col3:
    st.write('Diamond Dimensions')
    x = st.number_input('X', min_value=3.0, max_value=10.0, value=3.20)
    y = st.number_input('Y', min_value=3.0, max_value=10.0, value=3.73)
    z = st.number_input('Z', min_value=1.0, max_value=6.0, value=1.5)


if st.button('Predict'):
    price = predict_price(clarity, cut, color, carat, depth, table, x, y, z)
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
    st.subheader(
        "Updated the App with the following changes:")

    st.write("1. Did outliers removing with accurate values.")
    st.write("2. So I retained around ~ 53700 rows, while the original dataset has 53940 rows. But the first version took around ~1000 as outliers.")
    st.write("3. I also retained categorigcal columns (cut, color and clarity)  and used them in the model.")
    st.write("The result was :")

    st.write("scikit-learn algorithms:")
    st.write("1. Linear Regression: 0.9227 accuracy.")
    st.write("2. Lasso: 0.9224 accuracy.")
    st.write("3. DecisionTreeRegressor: 0.9674 accuracy.")

    st.write("XGBoost:")
    st.write("1. XGBRegressor: 0.9812 accuracy.")
    st.write("So we saved the model of XGBRegressor algorithm in a pickle file and we are using it here.")
    st.write(
        "New code is in files : video12-B.py and video13-B.py in github, link below:")
    github_url = 'https://github.com/shangab/PracticalAI/tree/main/STAT_ML_DS'
    st.markdown('Github [link](%s)' % github_url)
