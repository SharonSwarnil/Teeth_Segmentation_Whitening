import os
import sys
import time
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2

# ----------------------------
# Project Root (absolute path)
# ----------------------------
ROOT_DIR = r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Teeth_Segmentation_Whitening"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.mobilenetv2_deeplabv3 import get_model


# ----------------------------
# Utility Functions
# ----------------------------
def _ensure_image(input_image):
    """Accepts: filepath (str) or PIL.Image or numpy array.
       Returns: (PIL Image, base filename)"""
    if isinstance(input_image, str):
        img = Image.open(input_image).convert("RGB")
        base = os.path.splitext(os.path.basename(input_image))[0]
    elif isinstance(input_image, Image.Image):
        img = input_image.convert("RGB")
        base = f"img_{int(time.time())}"
    elif isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image.astype("uint8"))
        base = f"img_{int(time.time())}"
    else:
        raise ValueError("❌ Unsupported image type. Use path, PIL.Image, or numpy array.")
    return img, base


def apply_whitening_np(image_np, mask_np, intensity=0.15):
    """Apply whitening effect where mask==255 with smooth blending."""
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Soft mask
    mask = (mask_np > 127).astype(np.float32)
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)

    # Increase brightness in LAB space
    l = l + (mask_blur * intensity * 255.0)
    l = np.clip(l, 0, 255)

    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    whitened_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    whitened_rgb = cv2.cvtColor(whitened_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original
    blend = (whitened_rgb.astype(np.float32) * mask_blur[..., None] +
             image_np.astype(np.float32) * (1 - mask_blur[..., None])).astype(np.uint8)
    return blend


def find_model_path(candidate_paths=None):
    """Find model weights in common directories."""
    if candidate_paths is None:
        candidate_paths = [
            os.path.join(ROOT_DIR, "training", "best_model.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "best_model.pth"),
            os.path.join(ROOT_DIR, "best_model.pth"),
        ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None


# ----------------------------
# Main Inference Function
# ----------------------------
def run_inference(image_input, model_path=None, output_dir=None, return_all=True):
    """
    Run inference on an image.

    Args:
        image_input: path | PIL.Image | numpy.ndarray
        model_path: path to .pth file (auto-search if None)
        output_dir: save directory (default: ROOT_DIR/results)
        return_all: if True returns (overlay_path, mask_path, whitened_path)
                    else only returns whitened_path

    Returns:
        tuple or str: paths to results
    """
    try:
        # Load input image
        image, base_name = _ensure_image(image_input)
        orig_w, orig_h = image.size

        # Setup output dir
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "results")
        os.makedirs(output_dir, exist_ok=True)

        # Find model
        if model_path is None:
            model_path = find_model_path()
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}")

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ Using device: {device}")

        # Load model
        model = get_model(num_classes=1)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        # Quantize for CPU
        if device.type == "cpu":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.to("cpu")
        else:
            model.to(device)

        model.eval()

        # Preprocessing
        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, dict):
                out = out.get("out", out)

            if out.shape[1] > 1:
                out = out[:, 0:1, :, :]  # first channel = teeth

            probs = torch.sigmoid(out).squeeze().cpu().numpy()

        # Debug stats
        print(f"📊 Prob stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")

        # Save raw probability heatmap
        debug_path = os.path.join(output_dir, f"{base_name}_probs.png")
        cv2.imwrite(debug_path, (probs * 255).astype(np.uint8))

        # Resize logits before thresholding
        probs_resized = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = (probs_resized > 0.5).astype(np.uint8) * 255

        # Fallback if no mask detected
        if mask.sum() == 0:
            print("⚠️ Warning: Empty mask, fallback to blank.")
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Overlay visualization
        image_np = np.array(image)
        overlay_mask = image_np.copy()
        overlay_mask[mask > 127] = [255, 0, 0]
        blended = cv2.addWeighted(image_np, 0.7, overlay_mask, 0.3, 0)

        # Whitening effect
        whitened_np = apply_whitening_np(image_np, mask, intensity=5.51)

        # Save outputs
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        whitened_path = os.path.join(output_dir, f"{base_name}_whitened.png")

        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        cv2.imwrite(whitened_path, cv2.cvtColor(whitened_np, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved: {mask_path}, {overlay_path}, {whitened_path}")

        if return_all:
            return overlay_path, mask_path, whitened_path
        return whitened_path

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise e
