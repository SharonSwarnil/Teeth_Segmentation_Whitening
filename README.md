# 🦷 Teeth Segmentation & Whitening with DeepLabV3

This repository contains a **teeth segmentation & whitening pipeline** using **PyTorch + DeepLabV3 (MobileNetV2 backbone)**.  
It segments teeth from dental images and applies **natural whitening** using LAB color space.

## 📂 Project Structure
- \models/\ → model architectures (DeepLabV3, MobileNetV2 wrapper)
- \inference/\ → inference scripts (\infer.py\, \infer_all.py\)
- \	raining/\ → training loop, dataset, loss functions
- \data_preprocessing/\ → JSON annotation parser → masks
- \esults/\ → saved inference outputs (masks, overlays, whitening)

## 🚀 Usage
### Training
\\\ash
python training/train.py
\\\

### Inference
\\\ash
python inference/infer.py --image_path ./data/test/img/sample.jpg
\\\

### Web Demo (Gradio)
\\\ash
python models/app.py
\\\

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

## ✅ Features
- Teeth segmentation (binary mask)
- Whitening post-processing in LAB space
- Inference with **CPU quantization for faster speed**
- Gradio web app for quick demo

## 📜 License
MIT License.
