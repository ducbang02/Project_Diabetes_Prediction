import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np


st.title("Ứng Dụng Dự Đoán Điabetes")

image = Image.open('DiabetesPredictioninMachineLearningusingPython.jpg')
st.image(image)

input = open('diabetes.pkl', 'rb')
model = pkl.load(input)

st.header('Input admission information')
gl = st.number_input('Glucose')
bp = st.number_input('BloodPressure')
skinThickness = st.number_input('SkinThickness')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
dpf = st.number_input('DiabetesPedigreeFunction')
age = st.number_input('Age')


if gl is not None and bp is not None and skinThickness is not None and insulin is not None and bmi is not None and dpf is not None and age is not None:
    if st.button('Predict'):
        input_data_reshaped = np.array([gl, bp, skinThickness, insulin, bmi, dpf, age]).reshape(1,-1)
        # Chuẩn hóa dữ liệu đầu vào
        std_data = model.scaler.transform(input_data_reshaped)
       
        # Máy làm và cho ra kết quả 0 hoặc 1
        prediction = model.predict(std_data)
    
        st.header('Result')
        
        if (prediction[0] == 0):
            st.text('The person is not diabetic')
        else:
            st.text('The person is diabetic')