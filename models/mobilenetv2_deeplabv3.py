import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class TeethSegmenter(nn.Module):
    def __init__(self, num_classes=1):
        super(TeethSegmenter, self).__init__()
        
        # Load pretrained DeepLabV3 with MobileNetV3 backbone
        model = deeplabv3_mobilenet_v3_large(pretrained=True)
        
        # Replace the classifier head
        model.classifier = DeepLabHead(960, num_classes)  # MobileNetV3 has 960 channels
        
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]   # "out" is the segmentation logits

def get_model(num_classes=1):
    """Get the segmentation model."""
    return TeethSegmenter(num_classes)

if __name__ == "__main__":
    model = get_model(num_classes=5)
    model.eval()  # <-- Add this line
    print("Model loaded successfully.")
    import torch
    dummy = torch.randn(1, 3, 320, 320)
    output = model(dummy)
    print("Output shape:", output.shape)