import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from img_classifications import breast_classification
st.title("Image Classification Artificial Intelligence")
st.header("Breast Cancer Classification")
st.text("Upload a breast cancer Image for image classification as benign or malignant")

uploaded_file = st.file_uploader("Choose a image ...", type="png")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Scan.', use_column_width=True)
	st.write("")
	st.write("Classifying...")
	label = teachable_machine_classification(image, 'model.h5')
	if label == 0:
	    st.write("There is cancer present in the scan")
	else:
	    st.write("The scan is healthy")