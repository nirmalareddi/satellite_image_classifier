import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model (update the path if needed)
model = load_model("cnn_model.h5")

# Define class names used during training
class_names = ["barren", "forest", "urban", "water"]

st.title("🌍 Satellite Image Classifier")
st.write("Upload a satellite image to predict its land cover class.")

uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Predicted Class: `{predicted_class}`")
    st.markdown("### 🔍 Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name}: {prediction[0][i]:.2f}")