import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model


st.title('Arabic number recognizer')
st.write('Web app for handwritten digit recognition')
model = load_model('model')

SIZE = 192
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')


if st.button('Predict'):
    
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    img = img.reshape(1, 28, 28, 1)
    img /= 255
    digit = model.predict(img)
    classes=np.argmax(digit[0])
    st.write(f'Predicted Result: {classes}')
