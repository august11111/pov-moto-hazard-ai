import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Charger GIT
processor = AutoProcessor.from_pretrained("microsoft/git-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large")

def generate_captions(image_folder):
    captions = []
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(image_folder, fname)
        image = Image.open(path).convert("RGB")

        # Préparer et générer la caption
        inputs = processor(images=image, return_tensors="pt")
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=64)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        captions.append((fname, caption))
    return captions
