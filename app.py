import numpy as np
import streamlit as st
import pandas as pd
import datetime
import keras

old_models = keras.models.load_model('model.h5')

def home():
    return "welcome"


#@app.route('/predict', methods=['POST'])
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
    if st.button(""):
        result = predict(temperature,pressure,wind_speed,wind_direction)
    st.success('Predicted Power is {}'.format(result))


#@app.route('/')


if __name__ == "__main__":
    main()


