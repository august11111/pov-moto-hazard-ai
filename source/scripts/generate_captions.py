import os
import pandas as pd
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# === Charger le modèle BLIP ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# === Dossier contenant les images extraites de la vidéo ===
frames_folder = "test/data/frames/ytb_video"
captions = []

# === Boucle sur chaque image ===
for fname in sorted(os.listdir(frames_folder)):
    if fname.endswith(".jpg"):
        img_path = os.path.join(frames_folder, fname)
        raw_image = Image.open(img_path).convert('RGB')
        
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions.append((fname, caption))
        print(f"{fname} → {caption}")
        
        

# === Sauvegarde dans un fichier CSV ===
os.makedirs("test/data/descriptions", exist_ok=True)
df = pd.DataFrame(captions, columns=["image", "description"])
df.to_csv("test/data/descriptions/ytb_video_test.csv", index=False)

print("\n✅ Captions sauvegardées dans test/data/descriptions/ytb_video.csv")
