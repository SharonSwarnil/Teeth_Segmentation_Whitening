import os
import gradio as gr
from inference.infer import run_inference

#  UPDATE THIS PATH to where your best_model.pth is actually saved
MODEL_PATH = r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Teeth_Segmentation_Whitening\checkpoints\best_model.pth"

# Directory to save outputs
OUTPUT_DIR = r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Teeth_Segmentation_Whitening\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict(image):
    """Run inference and return overlay, mask, and whitened images."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f" Model file not found. Please place 'best_model.pth' here:\n{MODEL_PATH}"
        )

    overlay_path, mask_path, whitened_path = run_inference(
        image,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        return_all=True
    )

    return overlay_path, mask_path, whitened_path

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[
        gr.Image(type="filepath", label="Overlay"),
        gr.Image(type="filepath", label="Mask"),
        gr.Image(type="filepath", label="Whitened"),
    ],
    title=" Teeth Segmentation + Whitening",
    description="Upload a face image and get segmented teeth + whitening."
)

if __name__ == "__main__":
    demo.launch(share=False)  # set share=True if you want public link
