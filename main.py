from helper import *
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from keras.models import load_model

sns.set_style('dark')
sns.set()
st.title('Adult Mosquito Classifier')

model_option = st.selectbox(
    'Select fine-tuned model',
    ('ResNet50', 'VGG16'))

if model_option == 'ResNet50':
    predictor_model = load_model('ResNet50.model')

if model_option == 'VGG16':
    predictor_model = load_model('VGG16.model')

input_option = st.selectbox(
    "Image input method",
    ("Upload an image", "Camera"))

if input_option == "Upload an image":
    uploaded_file = st.file_uploader("Upload Image")

if input_option == "Camera":
    uploaded_file = st.camera_input("Take a picture")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('images', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1

    except:
        return 0


if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        prediction = predictor(os.path.join('images', uploaded_file.name), predictor_model)
        os.remove('images/'+uploaded_file.name)
        st.text('Predictions:')
        fig, ax = plt.subplots()
        ax = sns.barplot(y='name', x='values', data=prediction,
                         order=prediction.sort_values('values', ascending=False).name, palette='Paired')
        ax.set(xlabel='Confidence %', ylabel='Species')
        show_values(ax, "h", space=1)
        st.pyplot(fig)