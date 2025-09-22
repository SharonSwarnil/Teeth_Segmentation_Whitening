# inference/evaluate.py
import os, sys, json
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mobilenetv2_deeplabv3 import get_model

def load_gt_mask(json_path, img_size):
    W,H = img_size
    mask = Image.new("L", (W,H), 0)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return np.zeros((H,W), dtype=np.uint8)
    for obj in data.get("objects", []):
        pts = obj.get("points", {}).get("exterior", [])
        if not pts:
            continue
        poly = [(float(x), float(y)) for x,y in pts]
        ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def run_model_on_image(image_path, model, device, input_size=(320,320)):
    transform = T.Compose([T.Resize(input_size), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred = torch.sigmoid(out)[0,0].cpu().numpy()
    pred_bin = (pred > 0.5).astype(np.uint8)
    pred_bin = cv2.resize(pred_bin, img.size, interpolation=cv2.INTER_NEAREST)
    return pred_bin, img

def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return (2.0*inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + 1e-6) / (union + 1e-6)

if __name__ == "__main__":
    base_dir = r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Teeth_Segmentation_Whitening"
    test_img_dir = os.path.join(base_dir, "data", "test", "img")
    test_ann_dir = os.path.join(base_dir, "data", "test", "ann")
    model_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    img_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    dice_scores = []
    iou_scores = []

    for name in tqdm(img_files):
        img_path = os.path.join(test_img_dir, name)
        json_path = os.path.join(test_ann_dir, os.path.splitext(name)[0] + ".json")
        pred_mask, img = run_model_on_image(img_path, model, device)
        gt_mask = load_gt_mask(json_path, img.size) if os.path.exists(json_path) else np.zeros((img.height, img.width), dtype=np.uint8)

        d = dice(pred_mask, gt_mask)
        i = iou(pred_mask, gt_mask)
        dice_scores.append(d); iou_scores.append(i)

        # Save overlay
        img_np = np.array(img)
        overlay = img_np.copy(); overlay[pred_mask==1] = [255,0,0]
        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
        out_path = os.path.join(results_dir, os.path.splitext(name)[0] + "_overlay.png")
        Image.fromarray(blended).save(out_path)
        cv2.imwrite(os.path.join(results_dir, os.path.splitext(name)[0] + "_predmask.png"), pred_mask*255)

    print("Avg Dice:", np.mean(dice_scores))
    print("Avg IoU: ", np.mean(iou_scores))
    print("Results saved in:", results_dir)