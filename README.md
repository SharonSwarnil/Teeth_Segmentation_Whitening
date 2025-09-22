# Teeth Detection/Segmentation and Realistic Whitening Pipeline


#  Teeth Segmentation & Whitening

This project implements **teeth detection/segmentation** and applies a **realistic whitening filter** using deep learning.  
Built with **PyTorch + torchvision** (DeepLabV3 with MobileNetV2 backbone) and a **Gradio UI** for easy demos.


## Solution Approach

###  Dataset Preparation
- Used the Dental AI Dataset (images + JSON annotations).
- Preprocessed JSON to extract only 'teeth' class polygons.
- Converted polygons to binary masks using
- Run training to generate checkpoints.

### Features
- Teeth segmentation using DeepLabV3 (MobileNetV2 backbone).
- Whitening filter applied only on segmented teeth area.
- Natural whitening effect (avoids lips/tongue).
- Batch inference and Gradio interactive app.
- Quantization support for fast CPU inference.
- Supports **batch inference** and **Gradio demo UI**

---