from helper import *
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_style('dark')
sns.set()
st.title('Adult Mosquito Classifier')

option = st.selectbox(
    "Image input method",
    ("Upload an image", "Camera"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload Image")

if option == "Camera":
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
        prediction = predictor(os.path.join('images', uploaded_file.name))
        os.remove('images/'+uploaded_file.name)
        st.text('Predictions:')
        fig, ax = plt.subplots()
        ax = sns.barplot(y='name', x='values', data=prediction,
                         order=prediction.sort_values('values', ascending=False).name, palette='Paired')
        ax.set(xlabel='Confidence %', ylabel='Species')
        show_values(ax, "h", space=1)
        st.pyplot(fig)