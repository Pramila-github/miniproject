import numpy as np
import streamlit as st
import pandas as pd
import tensorflow as tf
import keras
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size:cover;


    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('windmill.gif')



old_models = tf.keras.models.load_model('model.h5')

def home():
    return "welcome"


def predict(temperature,pressure,wind_speed,wind_direction):
    values=np.array([[temperature,pressure,wind_speed,wind_direction]])
    prediction=old_models.predict(values.reshape(-1,1,4), batch_size=1)
    print(prediction)
    return prediction

def main():
    st.sidebar.header('User Input Parameters 💻️')
    st.title(" DEEP WINDS ⚒️")
    st.write("Made by Quad Techies with ❤️")
    st.write("### WIND POWER PREDICTION DL WEB-APP ")


    temperature = st.sidebar.slider('Temperature ⛅🌞🌧️ [°C]', -15, 1, 50)
    pressure = st.sidebar.slider('Pressure  ⚡ [atm]️', 0.9, 1.0, 1.0)
    wind_speed = st.sidebar.slider('Wind Speed  🌬️ [m/s]', 1, 1, 27)
    wind_direction = st.sidebar.slider('Wind Direction  🚩🌀 [deg]', 0, 1, 360)
    result = ""
    if st.button("Predict"):
        result = predict(temperature,pressure,wind_speed,wind_direction)
    st.success('Predicted Power is {}'.format(result))



if __name__ == "__main__":
    main()


