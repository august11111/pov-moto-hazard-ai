import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Charger GIT (Google Image Transformer)
processor = AutoProcessor.from_pretrained("microsoft/git-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large")

def generate_rich_description(image_path):
    """
    Génère une description riche pour une seule image à l'aide de GIT.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=64)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"[Erreur de génération pour {image_path} : {str(e)}]"
