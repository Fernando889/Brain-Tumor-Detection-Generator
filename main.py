import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models

st.set_page_config(
    page_title="Brain Tumor Detection Generator",
    layout="centered"
)

st.title("Introducing Brain Tumor Detection Generator (Learning Portfolio Project)")
st.divider()

st.markdown("""
## 🧠 Brain Tumor Detection Generator
### *(Learning Portfolio Project)*

This application is a **deep learning-based brain tumor classifier** built using
**PyTorch** with **EfficientNet-B0 (EffNetB0)** as the backbone model, and **Streamlit**
for the user interface (UI). It classifies **MRI brain scan images** into four categories:

- 🧠 **Glioma** — a type of brain tumor that originates from glial cells, which are the
  supportive tissues of the brain and spinal cord.
  _Source: [Google](https://www.google.com/search?q=glioma)_

- 🧠 **Meningioma** — a tumor that begins in the meninges, the membranes surrounding the
  brain and spinal cord.
  _Source: [Google](https://www.google.com/search?q=meningioma)_

- 🧠 **Pituitary** — an abnormal growth in the pituitary gland, a small gland at the base
  of the brain. Most pituitary tumors are benign (non-cancerous) and slow-growing, but
  they can still cause issues by affecting hormone production or pressing on nearby
  structures in the brain.
  _Source: [Google](https://www.google.com/search?q=pituitary+tumor)_

- ✅ **No Tumor**

The model is trained on the **BRISC2025** dataset from Kaggle.
📦 Dataset: [BRISC2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)

---

⚠️ **Disclaimer**
This application is developed **strictly for educational and research purposes** as part of a personal portfolio.
It is **not a diagnostic tool** and must **not be used for clinical or medical decision-making**.
""")

st.subheader("📤 Upload MRI Image for Prediction")
uploaded_file = st.file_uploader(
    "Choose an MRI scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_models_path = "saved_models/effnetb0.pth"

    checkpoint = torch.load(saved_models_path, map_location=device)

    manual_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = manual_transforms(image).unsqueeze(0)

    # Load Models
    if checkpoint["model_name"] == "effnetb0":
        model = models.efficientnet_b0(weights=None)
        num_classes = len(checkpoint["class_names"])
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        class_names = checkpoint["class_names"]
    else:
        st.error(f"Unsupported model: {checkpoint['model_name']}")
        st.stop()

    # Prediction
    with torch.inference_mode():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        predicted_idx = outputs.argmax(dim=1).item()
        prediction = class_names[predicted_idx]

    # Show Result
    st.success(f"🧠 ** Predicted Class: ** {prediction}")
