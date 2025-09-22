# Teeth Segmentation & Whitening – Final Report

## Problem Statement
The goal of this project was to detect or segment teeth from human face images and apply a realistic whitening filter on the segmented regions.  
- Segmentation should exclude lips, gums, and tongue.  
- Whitening must look natural, not overexposed or artificial.  
- Model should work in near real-time with lightweight architecture.  

---

## Approach

### 1. Dataset Preparation
- Dataset: [Dental AI Dataset](https://datasetninja.com/dentalai).  
- Structure:  
  - `data/train/img` → training images.  
  - `data/train/ann` → corresponding `.json` annotations.  
  - `data/val/img`, `data/val/ann` for validation.  
- Classes used: Only `teeth` class kept. Other labels ignored.  
- Preprocessing:  
  - Custom parser `data_preprocessing/parse_annotations.py` converted polygon annotations to binary masks.  
  - Images and masks resized to `320x320`.  
  - Masks stored as `.png` for compatibility.  

---

### 2. Model Architectures Attempted
- **U-Net with ResNet18 encoder**: worked but slower and heavier. Not chosen.  
- **DeepLabV3 + ResNet50 backbone**: high accuracy but very slow on CPU. Not chosen.  
- **DeepLabV3 + MobileNetV2 backbone (final choice)**: lightweight, efficient, and supported by TorchVision. Balanced speed vs accuracy.  

---

### 3. Training Strategy
- Loss function: **Binary Cross Entropy (BCE) + Dice Loss**.  
- Optimizer: **Adam (learning rate = 1e-4)**.  
- Scheduler: **ReduceLROnPlateau**.  
- Batch size: **8**.  
- Epochs: **50** (with early stopping).  
- Evaluation metric: **Dice coefficient, IoU**.  

---

### 4. Post-Processing
- **Thresholding**: Sigmoid output thresholded at 0.5.  
- **Mask resizing**: Predictions resized to original image size.  
- **Whitening filter**:  
  - Converted RGB → LAB color space.  
  - Increased L channel (lightness) in teeth regions only.  
  - Blended with Gaussian mask for natural results.  
- **Overlay**: Red overlay mask for debugging.  

---

### 5. Quantization and Inference
- Used **dynamic quantization** for faster CPU inference.  
- GPU preferred for batch predictions.  
- Inference scripts:  
  - `inference/infer.py` → single image.  
  - `inference/infer_all.py` → batch directory.  
- Debug outputs:  
  - `mask.png` → binary mask.  
  - `overlay.png` → overlay mask.  
  - `whitened.png` → final whitening result.  

---

### 6. Gradio Web App
- Implemented in `models/app.py`.  
- Interactive UI: upload an image, see mask, overlay, and whitened result.  
- Supports local testing and public link with `share=True`.  

---

## Final Chosen Approach
- Architecture: **DeepLabV3 + MobileNetV2**.  
- Preprocessing: **JSON to masks, resizing to 320x320**.  
- Training: **BCE + Dice Loss, Adam optimizer**.  
- Post-processing: **LAB whitening + blending**.  
- Deployment: **Gradio app + GitHub repo**.  

---

## Future Improvements
- Replace DeepLabV3 with **UNet++ or SegFormer** for sharper edges.  
- Add **mouth detection** to skip closed-mouth cases.  
- Extend pipeline to **real-time video whitening**.  
- Experiment with **model distillation** for lightweight mobile apps.  
- Use **mixed precision** for faster training.  

---

## Deliverables

1. **Detailed Report** (this document).  
   - Includes failed attempts, final pipeline, and improvements.  

2. **Reproducibility Instructions**  
   - Clone repo:  
     ```bash
     git clone https://github.com/SharonSwarnil/Teeth_Segmentation_Whitening.git
     cd Teeth_Segmentation_Whitening
     pip install -r requirements.txt
     ```
   - Run training:  
     ```bash
     python training/train.py
     ```
   - Run inference:  
     ```bash
     python inference/infer.py --image_path data/test/img/sample.jpg
     ```
   - Launch Gradio app:  
     ```bash
     python models/app.py
     ```

3. **Codebase**  
   - `data_preprocessing` → JSON parser + mask generator.  
   - `training` → training pipeline + loss functions.  
   - `inference` → inference scripts.  
   - `models` → DeepLabV3 backbone + Gradio app.  
   - `results` → sample outputs.  

---

## Conclusion
This project successfully builds an end-to-end teeth detection and whitening pipeline.  
The final solution is **lightweight, efficient, and produces realistic results**, meeting all assignment requirements.  
