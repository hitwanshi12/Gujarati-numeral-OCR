# Gujarati Numeral OCR

*Handwritten Gujarati Digit Recognition using Ensemble Learning with Deep Learning-based Feature Fusion*

---

## 🧠 Overview

This project presents an Optical Character Recognition (OCR) system for handwritten **Gujarati numerals (0–9)**.
It employs deep learning architectures—**ResNet50, VGG16, VGG19, and InceptionV3**—for feature extraction, followed by **feature fusion** and classification using **XGBoost**.

The approach enhances recognition accuracy and robustness for Gujarati digits, contributing to multilingual OCR research and regional language digit recognition.

---

## ✨ Features

* Recognizes handwritten Gujarati numerals (0–9)
* Uses feature fusion from multiple CNN architectures
* High-accuracy classification via XGBoost
* Includes preprocessing (thresholding, inversion, normalization)
* Extensible to other Indian language numeral datasets

---

## 🧩 Methodology

**Workflow:**

```
Input Image → Preprocessing → Feature Extraction (VGG16, VGG19, ResNet50, InceptionV3)
→ Feature Fusion → XGBoost Classifier → Output (Predicted Digit)
```

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/32b191e1-29bb-4aef-bd6e-55032390d175" />


---
## 📊 Results and Performance

### Model Performance Summary
| Model | Testing Accuracy | Precision | Recall | F1-Score |
|--------|------------------|-----------|--------|-----------|
| VGG-16 | 97.42% | 97.49% | 97.42% | 97.41% |
| VGG-19 | 97.04% | 97.07% | 97.04% | 97.03% |
| ResNet50 | 97.71% | 97.72% | 97.71% | 97.71% |
| InceptionV3 | 99.42% | 99.42% | 99.42% | 99.42% |
| **Proposed Model** | **99.68%** | **99.69%** | **99.68%** | **99.68%** |

---
### 🔍 Confusion Matrix

<p align="center">
  <img src="https://github.com/user-attachments/assets/acef1e12-4845-4689-9fad-fbaead9e7c15" width="45%" alt="Confusion Matrix"/>
</p>

---

### ⚙️ Installation & Usage

**1️⃣ Clone the Repository**
```bash
git clone https://github.com/hitwanshi12/gujarati-numeral-ocr.git
cd gujarati-numeral-ocr
```
**2️⃣ Training**

Run the following Jupyter notebooks individually to extract features from each CNN model:

**vgg16.ipynb**
**vgg19.ipynb**
**inceptionv3.ipynb**
**resnet.ipynb**

These notebooks generate feature vectors for the dataset using pre-trained models.

**3️⃣ Feature Fusion, Classification & Testing**

After extracting all features, run the classification notebook:

**classification.ipynb**

This notebook performs feature fusion, trains the XGBoost classifier, and evaluates the final model performance on the test dataset.

---

## 🧰 Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* XGBoost
* NumPy, Pandas, Matplotlib


---

## 📚 Research Paper

📄 **Paper Title:** *Handwritten Digit Recognition using Ensemble Learning with Deep Learning-based Feature Fusion*
📍*2024 8th International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC), Kirtipur, Nepal, 2024*

🔗 **Link:** https://ieeexplore.ieee.org/document/10714854/

---

## 🙌 Acknowledgements

Special thanks to **Parth Goel Sir**, **Hetvi Bhadani**, and **Arjav Ankoliya** for their valuable guidance and collaboration.

---

## 🏆 Citation

If you use this work, please cite:

```bibtex
@INPROCEEDINGS{10714854,
  author={Ankoliya Arjav, Bhadani Hetvi, Dalsania Hitwanshi and Goel Parth},
  booktitle={2024 8th International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC)}, 
  title={Handwritten Digit Recognition using Ensemble Learning with Deep Learning-based Feature Fusion}, 
  year={2024},
  volume={},
  number={},
  pages={1961-1966},
  keywords={Deep learning;Handwriting recognition;Analytical models;Accuracy;Writing;Linguistics;Feature extraction;Ensemble learning;Residual neural networks;Testing;Gujrati handwritten digits;classification;pre-trained CNN networks;ensemble learning;deep learning},
  doi={10.1109/I-SMAC61858.2024.10714854}}

```

---

## 📬 Contact

**Hitwanshi Dalsania**
📧 [hitwanshidalsania@gmail.com](mailto:hitwanshidalsania@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/hitwanshi-dalsania/) | [GitHub](https://github.com/hitwanshi12)

---
