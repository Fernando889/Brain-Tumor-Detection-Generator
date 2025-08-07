# 🧠 Brain Tumor Detection Generator

> A personal **Machine Learning Portfolio Project** built using PyTorch and Streamlit.  
> This app classifies **MRI brain scans** into one of four categories using a deep learning model trained on the **BRISC2025** dataset.

---

## 📌 Project Overview

The **Brain Tumor Detection Generator** is a **deep learning-based classifier** for MRI brain images.  
It distinguishes between the following classes:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The project is built using **EfficientNet-B0** as a feature extractor with a custom classifier head, trained in **PyTorch**, and deployed using **Streamlit** for an interactive user interface.

> ⚠️ **Disclaimer**  
> This application is intended **solely for educational and research purposes**.  
> It is **not a diagnostic tool** and must **not be used for clinical or medical decision-making**.

🌐 **Try it live now**: [Brain Tumor Detection Generator on Streamlit](https://brain-tumor-detection-generator.streamlit.app/)

## 📤 How to Use

1. Open the app in your browser via Streamlit.
2. Upload an MRI scan (`.jpg`, `.jpeg`, or `.png`).
3. The model processes the image and predicts one of the four classes.
4. See the result immediately displayed with a success message.

---

## 🧠 Key Features

- 🔍 Upload MRI brain scans for prediction
- 🤖 Trained using EfficientNet-B0 (transfer learning)
- 📊 Tracks experiments using the `runs/` directory (PyTorch)
- 📈 Reports training/validation accuracy and loss curves
- 🧪 Based on the BRISC2025 public dataset
- 🚀 Streamlit-powered clean and interactive UI

---

## 🛠️ Tech Stack

| Component | Tool/Library                |
| --------- | --------------------------- |
| Framework | PyTorch                     |
| UI        | Streamlit                   |
| Model     | EfficientNet-B0             |
| Image I/O | Pillow (`PIL`)              |
| Dataset   | BRISC2025 (Kaggle)          |
| Tracking  | TensorBoard Logs in `/runs` |

---

## 🧬 Dataset

- **Name**: BRISC2025 Brain Tumor MRI Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)
- **Used Models**: Classification Task Models (classification_task)
- **Structure**: Labeled MRI images organized into folders per class.

---

## 🧪 Model & Training

- **Architecture**: EfficientNet-B0 (from `torchvision.models`)
- **Input Size**: 224x224
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Training Epochs**: 40 Epochs

> Training logs, loss/accuracy plots, and metrics are stored in the `runs/` directory using TensorBoard.

```bash
tensorboard --logdir=runs

### EffNet Training & Testing Accuracy (EffNetB0 and EffNetB2)

| Model    | Epochs | Test Acc (Smooth) | Test Acc (Value) | Train Acc (Smooth) | Train Acc (Value) | Time (min) |
|----------|--------|-------------------|-------------------|---------------------|--------------------|-------------|
| effnetb0 | 5      | 86.0998           | 88.0859           | 88.8972             | 89.6895            | 1.07        |
| effnetb0 | 10     | 89.6197           | 90.4297           | 90.8752             | 91.1823            | 2.543       |
| effnetb0 | 15     | 90.2780           | 90.8203           | 92.0363             | 92.1576            | 3.989       |
| effnetb0 | 20     | 91.1530           | 90.6250           | 92.2964             | 92.0581            | 5.498       |
| effnetb0 | 25     | 92.4001           | 92.4805           | 92.6719             | 92.7548            | 6.957       |
| effnetb0 | 30     | 92.1730           | 92.3828           | 92.5784             | 92.2970            | 8.032       |
| effnetb0 | 35     | 91.6349           | 91.0156           | 92.4511             | 92.6154            | 9.798       |
| effnetb0 | 40     | 92.9048           | 93.2617           | 92.6272             | 92.5358            | 11.45       |
| effnetb0 | 45     | 92.2852           | 92.0898           | 92.8850             | 93.1728            | 12.88       |
| effnetb0 | 50     | 92.6200           | 92.4805           | 93.2973             | 93.4713            | 14.27       |
| effnetb2 | 5      | 83.1341           | 84.8633           | 86.9153             | 87.8185            | 1.612       |
| effnetb2 | 10     | 86.4770           | 87.5000           | 88.7919             | 88.9331            | 3.664       |
| effnetb2 | 15     | 88.0551           | 88.8672           | 89.3348             | 89.1919            | 5.819       |
| effnetb2 | 20     | 88.8764           | 88.7695           | 89.7564             | 89.5303            | 7.907       |
| effnetb2 | 25     | 88.1791           | 87.5000           | 90.0262             | 89.6895            | 9.598       |
| effnetb2 | 30     | 88.8803           | 89.2578           | 89.7474             | 89.6099            | 12.11       |
| effnetb2 | 35     | 88.8640           | 88.9648           | 90.0556             | 90.3065            | 14.39       |
| effnetb2 | 40     | 88.7683           | 88.5742           | 90.5119             | 90.7842            | 16.55       |
| effnetb2 | 45     | 89.7335           | 89.2578           | 90.2967             | 90.8041            | 18.32       |
| effnetb2 | 50     | 89.3347           | 89.4531           | 90.6009             | 91.0629            | 20.44       |

### EffNet Training & Testing Loss (EffNetB0 and EffNetB2)

| Model    | Epochs | Test Loss (Smooth) | Test Loss (Value) | Train Loss (Smooth) | Train Loss (Value) | Time (min) |
|----------|--------|---------------------|---------------------|----------------------|----------------------|-------------|
| effnetb0 | 5      | 0.3547              | 0.3187              | 0.3212               | 0.2897               | 1.07        |
| effnetb0 | 10     | 0.2701              | 0.2526              | 0.2529               | 0.2413               | 2.543       |
| effnetb0 | 15     | 0.2475              | 0.2487              | 0.2184               | 0.2098               | 3.989       |
| effnetb0 | 20     | 0.2249              | 0.2373              | 0.2089               | 0.2140               | 5.498       |
| effnetb0 | 25     | 0.2103              | 0.2144              | 0.1991               | 0.1961               | 6.957       |
| effnetb0 | 30     | 0.2080              | 0.2038              | 0.1963               | 0.1964               | 8.032       |
| effnetb0 | 35     | 0.2109              | 0.2199              | 0.1960               | 0.1926               | 9.798       |
| effnetb0 | 40     | 0.1923              | 0.1916              | 0.1930               | 0.1927               | 11.45       |
| effnetb0 | 45     | 0.1985              | 0.2032              | 0.1897               | 0.1871               | 12.88       |
| effnetb0 | 50     | 0.1906              | 0.1870              | 0.1830               | 0.1821               | 14.27       |
| effnetb2 | 5      | 0.4215              | 0.3776              | 0.3689               | 0.3322               | 1.612       |
| effnetb2 | 10     | 0.3378              | 0.3150              | 0.3074               | 0.2950               | 3.664       |
| effnetb2 | 15     | 0.3123              | 0.3062              | 0.2844               | 0.2831               | 5.819       |
| effnetb2 | 20     | 0.2949              | 0.2921              | 0.2694               | 0.2651               | 7.907       |
| effnetb2 | 25     | 0.2946              | 0.3055              | 0.2634               | 0.2657               | 9.598       |
| effnetb2 | 30     | 0.2801              | 0.2766              | 0.2656               | 0.2682               | 12.11       |
| effnetb2 | 35     | 0.2832              | 0.2878              | 0.2640               | 0.2653               | 14.39       |
| effnetb2 | 40     | 0.2822              | 0.2766              | 0.2552               | 0.2507               | 16.55       |
| effnetb2 | 45     | 0.2653              | 0.2709              | 0.2570               | 0.2457               | 18.32       |
| effnetb2 | 50     | 0.2656              | 0.2652              | 0.2515               | 0.2412               | 20.44       |

```

---

## 📦 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/brain-tumor-detection
cd brain-tumor-detection
```

2. **Install Requirements**

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

3. **Launch Streamlit App**

```bash
streamlit run main.py
```

---

## 🙏 Acknowledgments

- 📚 **Dataset**: [BRISC2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)
- 🧠 **EfficientNet-B0**: Google AI
- 💬 **Streamlit**: Open-source app framework
