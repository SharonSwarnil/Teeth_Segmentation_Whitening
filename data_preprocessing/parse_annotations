import json
import os
import cv2
import numpy as np
from PIL import Image

def polygon_to_mask(img_shape, polygons):
    """Convert polygons to binary mask."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:  # Skip invalid polygons
            continue
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask

def create_teeth_masks(image_dir, annotation_dir, output_image_dir, output_mask_dir, img_size=(320, 320)):
    """Parse JSON annotations and create teeth masks only."""
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    processed_count = 0
    for ann_file in os.listdir(annotation_dir):
        if not ann_file.endswith('.json'):
            continue
        
        ann_path = os.path.join(annotation_dir, ann_file)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        
        # Extract teeth polygons only (class: 'teeth')
        teeth_polygons = []
        for obj in ann.get('objects', []):
            if obj.get('label', '').lower() == 'teeth' and 'polygon' in obj:
                teeth_polygons.append(obj['polygon'])  # polygons already a list of points
        
        # Corresponding image
        img_name = ann_file.replace('.json', '.jpg')
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping {ann_file}")
            continue
        
        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img_resized = cv2.resize(img, img_size)
        
        # Scale polygons
        scale_x = img_size[1] / w0  # Note: img_size=(H,W) but resize uses (W,H)
        scale_y = img_size[0] / h0
        scaled_polygons = []
        for poly in teeth_polygons:
            scaled_poly = [[int(x * scale_x), int(y * scale_y)] for x, y in poly]
            if len(scaled_poly) >= 3:
                scaled_polygons.append(scaled_poly)
        
        # Create mask
        mask = polygon_to_mask(img_resized.shape, scaled_polygons)
        
        # Save resized image and mask
        img_path_out = os.path.join(output_image_dir, img_name)
        mask_path_out = os.path.join(output_mask_dir, img_name.replace('.jpg', '_mask.png'))
        Image.fromarray(img_resized).save(img_path_out)
        cv2.imwrite(mask_path_out, mask * 255)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} files")
    
    print(f"Completed processing {processed_count} files. Masks saved to {output_mask_dir}")

if __name__ == "__main__":
    ROOT = r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Teeth_Segmentation_Whitening\data"

    # Train set
    create_teeth_masks(
        image_dir=os.path.join(ROOT, "train", "img"),
        annotation_dir=os.path.join(ROOT, "train", "ann"),
        output_image_dir=os.path.join(ROOT, "images", "train"),
        output_mask_dir=os.path.join(ROOT, "masks", "train"),
        img_size=(320, 320)
    )

    # Validation set
    create_teeth_masks(
        image_dir=os.path.join(ROOT, "valid", "img"),
        annotation_dir=os.path.join(ROOT, "valid", "ann"),
        output_image_dir=os.path.join(ROOT, "images", "valid"),
        output_mask_dir=os.path.join(ROOT, "masks", "valid"),
        img_size=(320, 320)
    )

    # Test set
    create_teeth_masks(
        image_dir=os.path.join(ROOT, "test", "img"),
        annotation_dir=os.path.join(ROOT, "test", "ann"),
        output_image_dir=os.path.join(ROOT, "images", "test"),
        output_mask_dir=os.path.join(ROOT, "masks", "test"),
        img_size=(320, 320)
    )
    print("All datasets processed.")