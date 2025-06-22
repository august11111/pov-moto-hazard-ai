import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from source.utils.llama_utils import ask_llama
from source.utils.yolo_utils import detect_objects_yolo

input_csv = "test/data/descriptions/ytb_video_test.csv"
output_csv = "test/data/analyses/ytb_video_test_analyses.csv"


df = pd.read_csv(input_csv)

results = []

for idx, row in df.iterrows():
    description = row['description']
    # Détection d’objets via YOLO
    image_path = os.path.join("test", "data", "frames", "ytb_video", row["image"])
    objects = detect_objects_yolo(image_path)
    object_str = ", ".join(set(objects))

    prompt = f"""
Tu es un assistant expert en sécurité moto.
Analyse la situation suivante et donne des conseils pour éviter un accident.

Situation : {description}

Réponds de manière synthétique en 1 ou 2 phrases.
"""
    print(f"Frame {row['image']} → Analyse en cours...")
    analyse = ask_llama(prompt)
    print(f"Réponse : {analyse}\n")

    results.append({
        "image": row["image"],
        "description": description,
        "objects_detected": object_str,
        "analyse_llama": analyse
    })

# Sauvegarde dans un CSV
output_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
output_df.to_csv(output_csv, index=False)

print(f"\n✅ Analyse terminée et sauvegardée dans {output_csv}")
