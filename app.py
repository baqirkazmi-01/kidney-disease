import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KIDNEY-Diseases.h5")
# Same order as train.class_indices in the notebook (alphabetical)
CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
INFO = {
    "Cyst": "Fluid-filled sac in or on the kidney. Often benign, but needs clinical follow-up.",
    "Normal": "No cyst, stone or tumor pattern detected in this CT slice.",
    "Stone": "Hard mineral deposit (calculus) in the kidney or urinary tract.",
    "Tumor": "Abnormal mass of tissue. Requires urgent specialist evaluation.",
}
ICON = {"Cyst": "🔵", "Normal": "🟢", "Stone": "🟡", "Tumor": "🔴"}

st.set_page_config(page_title="Kidney CT Classifier", page_icon="🩻", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import keras
    return keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(img: Image.Image, size):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(size)
    x = np.asarray(img, dtype=np.float32)
    x = x / 127.5 - 1.0  # identical to mobilenet_v2.preprocess_input used in training
    return np.expand_dims(x, 0)


st.title("🩻 Kidney Disease Classifier")
st.caption("CNN trained on CT kidney images · Cyst / Normal / Stone / Tumor")

if not os.path.exists(MODEL_PATH):
    st.error(f"`{MODEL_PATH}` not found. Put it in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model.")
    st.exception(e)
    st.stop()
h, w = model.input_shape[1:3]

files = st.file_uploader(
    "Upload one or more CT images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if not files:
    st.info("Upload a kidney CT image to get a prediction.")

for f in files or []:
    img = Image.open(f)
    probs = model.predict(preprocess(img, (w, h)), verbose=0)[0]
    top = int(np.argmax(probs))
    label = CLASSES[top]

    st.divider()
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.image(img, caption=f.name, use_container_width=True)
    with c2:
        st.subheader(f"{ICON[label]} {label}")
        st.metric("Confidence", f"{probs[top] * 100:.1f}%")
        st.write(INFO[label])
        for name, p in zip(CLASSES, probs):
            st.progress(float(p), text=f"{name}: {p * 100:.1f}%")

st.divider()
st.warning(
    "For educational/research use only. This is not a medical device and must not replace "
    "diagnosis by a qualified radiologist or physician."
)
