import io
import requests
import streamlit as st
from PIL import Image


def load_image(image_file):
    img = Image.open(image_file)
    return img


st.sidebar.header("Select service")
name = st.sidebar.selectbox("Service", ["ImageNet"])

st.title("imagenet classification web app")
with st.form('my_form'):
    image_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Submit")
    if submitted:
        files = {"file": image_file.getvalue()}
        st.write("Result")
        result = requests.post("http://127.0.0.1:8000/csf", files=files)
        img = load_image(image_file)
        st.image(img)
        if result:
            st.text(f"{result.json()}")
