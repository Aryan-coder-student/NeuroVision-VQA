import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="VQA with BLIP", layout="centered", page_icon="🤖")
st.title("Visual Question Answering (VQA) with BLIP")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
question = st.text_input("Ask a question about the image:")

if uploaded_file and question:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Get Answer"):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"question": question}
        response = requests.post("http://127.0.0.1:5000//predict/", files=files, data=data)
        
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer received")
            st.success(f"Answer: {answer}")
        else:
            st.error("Error in fetching the answer. Please try again.")
