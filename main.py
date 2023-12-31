import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np


st.title("Diabetes Prediction Application")

image = Image.open('DiabetesPredictioninMachineLearningusingPython.jpg')
st.image(image)

input = open('diabetes.pkl', 'rb')
model = pkl.load(input)

st.header('Input admission information')
# 4	110	92	0	0	37.6	0.191	30	0
# 9	165	88	0	0	30.4	0.302	49	1


pre = st.number_input('Pregnancies',step=1, format='%d')
gl = st.number_input('Glucose',step=1, format='%d')
bp = st.number_input('BloodPressure', step=1, format='%d')
skinThickness = st.number_input('SkinThickness', step=1, format='%d')
insulin = st.number_input('Insulin', step=1, format='%d')
bmi = st.number_input('BMI')
dpf = st.number_input('DiabetesPedigreeFunction')
age = st.number_input('Age', step=1, format='%d')


if pre is not None and gl is not None and bp is not None and skinThickness is not None and insulin is not None and bmi is not None and dpf is not None and age is not None:
    if st.button('Predict'):
        feature_vector = np.array([pre, gl, bp, skinThickness, insulin, bmi, dpf, age]).reshape(1,-1)

        std_data = model.scaler.transform(feature_vector)
        # Máy làm và cho ra kết quả 0 hoặc 1
        prediction = model.predict(std_data)
    
        st.header('Result')
        
        if (prediction[0] == 0):
            st.text('The person is not diabetic')
        else:
            st.text('The person is diabetic')