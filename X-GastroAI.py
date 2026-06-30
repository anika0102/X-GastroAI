import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_PATH = "best_resnet50_gastric_finetuned.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    state_dict = torch.load(
    path,
    map_location=device,
    weights_only=True
)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# def download_model(url, path):
    import requests
    with st.spinner("Downloading model (this may take a while)..."):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def generate_gradcam(image, model, device):
    img = image.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    outputs = model(input_tensor)
    pred_class = outputs.argmax(dim=1).item()

    heatmap = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_class)]
    )[0]

    img_np = np.array(img.resize((224, 224))) / 255.0
    cam_vis = show_cam_on_image(img_np, heatmap, use_rgb=True)

    return pred_class, img_np, cam_vis

st.set_page_config(page_title="🩺X-GastricAI")
st.title("🩺Gastric Cancer Screening")
st.write("Upload a histopathology image, and our model will analyze it to detect whether the tissue is cancerous (Abnormal) or Normal. You will also see a Grad-CAM heatmap overlay showing which regions influenced the model’s decision.")

model = None
device = None

if not os.path.exists(MODEL_PATH) and MODEL_URL:
    st.warning("Model file not found locally. You can download it now (requires internet).")
    if st.button("Download model now"):
        try:
            download_model(MODEL_URL, MODEL_PATH)
            st.success("Model downloaded. Reload the app (F5) or re-run the cell block below.")
        except Exception as e:
            st.error(f"Download failed: {e}")

if os.path.exists(MODEL_PATH):
    with st.spinner("Loading model..."):
        try:
            model, device = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            model = None
st.markdown("## Choose an Image")


if "image" not in st.session_state:
    st.session_state.image = None

option = st.radio(
    "Image Source",
    ["Upload your own image", "Use a sample image"],
    horizontal=True
)

if option == "Upload your own image":

    uploaded_file = st.file_uploader(
        "Upload Histopathology Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            st.session_state.image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Unable to open the uploaded image.")

else:

    sample = st.radio(
        "Choose a sample image",
        [
            "Normal Tissue",
            "Cancer Tissue"
        ],
        horizontal=True
    )

    if sample == "Normal Tissue":
        st.session_state.image = Image.open(
            "sample_images/normal.png"
        ).convert("RGB")

    else:
        st.session_state.image = Image.open(
            "sample_images/cancer.png"
        ).convert("RGB")

image = st.session_state.image

if image is not None:

    st.image(
        image,
        caption="Selected Image",
        width=300
    )

    if st.button("Run Prediction"):

        if model is None:
            st.error("Model could not be loaded.")

        else:

            with st.spinner("Running inference..."):

                try:

                    input_tensor = transform(image).unsqueeze(0).to(device)

                    outputs = model(input_tensor)

                    probabilities = torch.softmax(outputs, dim=1)

                    confidence = probabilities.max().item() * 100

                    pred_class, _, cam_img = generate_gradcam(
                        image,
                        model,
                        device
                    )

                    if pred_class == 1:
                        prediction = "Cancerous Tissue Detected"
                    else:
                        prediction = "Normal Tissue"

                    st.subheader("Prediction")

                    st.success(prediction)

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.write("Class Probabilities")

                    st.write(f"Normal Tissue: {probabilities[0][0] * 100:.2f}%")

                    st.write(f"Cancer Tissue: {probabilities[0][1] * 100:.2f}%")

                    st.subheader("Grad-CAM Visualization")

                    st.image(
                        cam_img,
                        caption="Model Attention Heatmap",
                        width=350
                    )

                except Exception as e:

                    st.error(f"Prediction failed: {e}")




st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#888;">
        Developed by <b>Anika0102</b><br>
        Gastric Cancer Screening using Deep Learning and Grad-CAM<br><br>
        <a href="https://github.com/anika0102/X-GastroAI" target="_blank">
            View Source Code
        </a>
    </div>
    """,
    unsafe_allow_html=True
)