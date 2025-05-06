import os
import pandas as pd
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# === Charger le modèle BLIP-2 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
model.to(device)

def generate_captions_for_folder(frames_folder, output_csv):
    """
    Génère les captions pour chaque image dans un dossier donné et les sauvegarde dans un CSV.
    """
    captions = []

    for fname in sorted(os.listdir(frames_folder)):
        if fname.endswith(".jpg"):
            img_path = os.path.join(frames_folder, fname)
            raw_image = Image.open(img_path).convert('RGB')
        
            inputs = processor(raw_image, text="Décris le contexte routier, les autres véhicules, et les risques potentiels", return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            captions.append((fname, caption))
            print(f"{fname} → {caption}")
    
    # Sauvegarde dans un fichier CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(captions, columns=["image", "description"])
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Captions sauvegardées dans {output_csv}")
