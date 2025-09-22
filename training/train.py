# training/train.py
import os
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mobilenetv2_deeplabv3 import get_model

# ---------- JSON->mask helper ----------
def load_mask_from_json(json_path, image_size):
    """Return numpy mask (H, W) with 1 for teeth polygons, 0 otherwise.
       JSON format expected: 'objects' list with 'points' -> 'exterior' polygons, 'classTitle' field."""
    W, H = image_size
    mask = Image.new('L', (W, H), 0)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return np.zeros((H, W), dtype=np.uint8)

    for obj in data.get("objects", []):
        pts = obj.get("points", {}).get("exterior", [])
        if not pts:
            continue
        # Fill polygon
        poly = [(float(x), float(y)) for (x, y) in pts]
        ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

# ---------- Dataset ----------
class TeethDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(320, 320)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size  # (H, W)
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(self.images) == 0:
            raise RuntimeError(f"No images in {image_dir}")
        self.img_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.image_dir, name)
        json_name = os.path.splitext(name)[0] + ".json"
        json_path = os.path.join(self.mask_dir, json_name)

        image = Image.open(img_path).convert("RGB")
        if os.path.exists(json_path):
            mask_np = load_mask_from_json(json_path, image.size)
            mask = Image.fromarray((mask_np*255).astype('uint8')).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

        image_t = self.img_transform(image)
        mask_t = self.mask_transform(mask)
        mask_t = (mask_t > 0.5).float()

        return image_t, mask_t

# ---------- Loss ----------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1,2,3))
        denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2.*intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

# ---------- Train ----------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_dir = r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Teeth_Segmentation_Whitening"
    train_img = os.path.join(base_dir, "data", "train", "img")
    train_ann = os.path.join(base_dir, "data", "train", "ann")
    val_img = os.path.join(base_dir, "data", "valid", "img")
    val_ann = os.path.join(base_dir, "data", "valid", "ann")

    input_size = (320, 320)
    batch_size = 8
    epochs = 10
    lr = 1e-3

    train_ds = TeethDataset(train_img, train_ann, image_size=input_size)
    val_ds = TeethDataset(val_img, val_ann, image_size=input_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(num_classes=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            opt.zero_grad()
            outs = model(imgs)
            loss = bce(outs, masks) + dice(outs, masks)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1} train loss: {running/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device); masks = masks.to(device)
                outs = model(imgs)
                preds = (torch.sigmoid(outs) > 0.5).float()
                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice_score = (2*inter + 1e-6) / (union + 1e-6)
                val_dice += dice_score.item()
        avg_val = val_dice / len(val_loader)
        print(f"Epoch {epoch+1} val dice: {avg_val:.4f}")

        if avg_val > best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(base_dir, "checkpoints", "best_model.pth"))
            print("Saved best model:", best_val)

    print("Done training.")

if __name__ == "__main__":
    train_model()

